# ruff: noqa: C901, EM101, EM102, PLR0912, PLR0915, TRY003
"""Run the isolated 99-parameter latent-region activation screen.

Non-ring regions store one signed latent vector ``v`` and use

``H = 1.5 v v^T - 0.5 ||v||^2 I``.

The three ring regions retain their prepared cellwise circumferential fiber.
Only the first stored coordinate is free there, and it is the nonnegative
square root of activation amplitude.  The other two ring coordinates remain
zero.  Thus the on-disk array has shape ``(35, 3)`` and 105 scalar slots, while
the feasible model has ``32 * 3 + 3 * 1 = 99`` free parameters.

The optimizer loop, strict equilibrium solver, trial acceptance, snapshots,
and reset-branch check are reused from ``20-run-face-inverse.py`` through an
isolated import.  Shared experiment helpers are not modified.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pyvista as pv
import torch
from experiment_profile import ProfileCometNoCommit

from liblaf import cherries

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
DEFAULT_FIXTURE = EXPERIMENT / "data/10-fixture"
DEFAULT_INITIAL = EXPERIMENT / "data/26-latent-region-initial-v2.npz"
BASE_RUNNER = HERE / "20-run-face-inverse.py"
RING_MUSCLE_IDS = frozenset({97, 98, 254})
INITIAL_ACTIVATION = 0.02
METHOD = "LatentRegionHybrid"
AXIS_AMPLITUDE_TOLERANCE = 64.0 * np.finfo(np.float64).eps


def _load_base_runner() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_face_inverse_20", BASE_RUNNER)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import base runner: {BASE_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = _load_base_runner()
AREF = BASE.AREF


class Config(BASE.Config):
    """Pinned configuration for the latent-region screen."""

    output_dir: Path = cherries.output("26-latent-region-fat049", mkdir=True)
    method: str = METHOD
    initial: Path | None = DEFAULT_INITIAL
    smoothness: float = 0.0
    soft_nu: float = 0.46
    fat_nu: float | None = 0.49


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    if array.dtype.hasobject:
        raise TypeError("object arrays cannot have stable byte receipts")
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def ring_region_mask(region_muscle_ids: np.ndarray) -> np.ndarray:
    muscle_ids = np.asarray(region_muscle_ids, dtype=np.int64).reshape(-1)
    mask = np.isin(muscle_ids, tuple(sorted(RING_MUSCLE_IDS)))
    if set(muscle_ids[mask].tolist()) != RING_MUSCLE_IDS or int(mask.sum()) != 3:
        raise ValueError("fixture must contain exactly the three declared ring regions")
    return mask


def project(q: torch.Tensor, ring_regions: torch.Tensor, amax: float) -> torch.Tensor:
    """Project non-rings to a row ball and rings to one nonnegative scalar."""
    if q.ndim != 2 or q.shape[1] != 3 or not q.is_floating_point():
        raise ValueError("q must be floating point with shape (regions, 3)")
    if ring_regions.shape != (len(q),) or ring_regions.dtype != torch.bool:
        raise ValueError("ring_regions must be a Boolean mask over q rows")
    if not math.isfinite(amax) or amax < 0.0:
        raise ValueError("amax must be finite and nonnegative")
    if amax == 0.0:
        return torch.zeros_like(q)
    radius = q.new_tensor(amax).sqrt()
    norm = torch.linalg.vector_norm(q, dim=1, keepdim=True)
    nonring = q * (radius / norm.clamp_min(radius)).clamp_max(1.0)
    ring = torch.zeros_like(q)
    ring[:, 0] = q[:, 0].clamp(0.0, radius)
    return torch.where(ring_regions[:, None], ring, nonring)


def local_vectors(
    q: torch.Tensor,
    region_ids: torch.Tensor,
    fibers: torch.Tensor,
    ring_regions: torch.Tensor,
) -> torch.Tensor:
    """Map region coordinates to cellwise latent vectors."""
    if q.ndim != 2 or q.shape[1] != 3:
        raise ValueError("q must have shape (regions, 3)")
    if region_ids.ndim != 1 or fibers.shape != (len(region_ids), 3):
        raise ValueError("region_ids and fibers must describe the same cells")
    if ring_regions.shape != (len(q),):
        raise ValueError("ring_regions must describe q rows")
    if torch.any(region_ids < 0) or torch.any(region_ids >= len(q)):
        raise ValueError("region_ids contain an out-of-range region")
    norm = torch.linalg.vector_norm(fibers, dim=1)
    if not torch.allclose(
        norm,
        torch.ones_like(norm),
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError("prepared active fibers must be unit vectors")
    region_q = q[region_ids]
    ring_cell = ring_regions[region_ids]
    ring_v = region_q[:, :1] * fibers
    return torch.where(ring_cell[:, None], ring_v, region_q)


def matrices_from_vectors(vectors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(A_inv, H)`` for cellwise latent vectors."""
    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError("vectors must have shape (cells, 3)")
    amplitude = vectors.square().sum(dim=1)
    identity = torch.eye(3, dtype=vectors.dtype, device=vectors.device).expand(
        len(vectors), 3, 3
    )
    outer = torch.einsum("ni,nj->nij", vectors, vectors)
    H = 1.5 * outer - 0.5 * amplitude[:, None, None] * identity
    return torch.matrix_exp(H), H


def initial_coordinates(
    region_ids: np.ndarray,
    fibers: np.ndarray,
    ring_regions: np.ndarray,
    activation: float = INITIAL_ACTIVATION,
) -> np.ndarray:
    """Build the target-independent nonzero rest-fiber initialization."""
    if not math.isfinite(activation) or activation <= 0.0:
        raise ValueError("initial activation must be finite and positive")
    region_ids = np.asarray(region_ids, dtype=np.int64)
    fibers = np.asarray(fibers, dtype=np.float64)
    if fibers.shape != (len(region_ids), 3):
        raise ValueError("fibers must match region_ids")
    q = np.zeros((len(ring_regions), 3), dtype=np.float64)
    scale = math.sqrt(activation)
    for region_id in range(len(q)):
        rows = region_ids == region_id
        if not np.any(rows):
            raise ValueError(f"region {region_id} has no active cells")
        if ring_regions[region_id]:
            q[region_id, 0] = scale
        else:
            region_fibers = fibers[rows]
            if not np.all(region_fibers == region_fibers[0]):
                raise ValueError(
                    f"non-ring region {region_id} does not have a constant signed fiber"
                )
            q[region_id] = scale * region_fibers[0]
    return q


class Objective:
    """Mixed latent-vector/ring-scalar objective for the imported base loop."""

    def __init__(self, physics: Any, cfg: Config) -> None:
        if cfg.method != METHOD:
            raise ValueError(f"method is pinned to {METHOD!r}")
        self.p, self.cfg = physics, cfg
        self.shared = True
        self.spatial_modes = False
        self.mode = METHOD
        self.shape = (physics.n_regions, 3)
        region_muscle_ids = np.asarray(
            physics.mesh.field_data["ActivationRegionMuscleId"], dtype=np.int64
        )
        if len(region_muscle_ids) != physics.n_regions:
            raise ValueError("activation region metadata and active cells differ")
        self.ring_regions = torch.as_tensor(
            ring_region_mask(region_muscle_ids), dtype=torch.bool
        )
        self.volume = torch.as_tensor(physics.volumes)
        self.mass = torch.as_tensor(physics.region_mass)[:, None]
        self.ei, self.ej, self.ew = (torch.as_tensor(array) for array in physics.graph)
        self.target = torch.as_tensor(physics.target)
        self.scale = physics.D
        self.calls = 0

    def project(self, q: torch.Tensor) -> torch.Tensor:
        return project(q, self.ring_regions, self.cfg.amax)

    def __call__(
        self, value: torch.Tensor, seed: np.ndarray, *, backward: bool = True
    ) -> dict[str, Any]:
        q = value.detach().clone().requires_grad_(backward)
        local = local_vectors(q, self.p.region_t, self.p.fibers, self.ring_regions)
        A, H = matrices_from_vectors(local)
        u = self.p.solve(BASE.am.packed(A), seed)
        detf_min = self.p.detf(u.detach().cpu().numpy()).min()
        ainv_np = A.detach().cpu().numpy()
        activation_eigen_min = float(np.linalg.eigvalsh(ainv_np).min())
        if detf_min < self.cfg.det_floor or activation_eigen_min <= 0.0:
            raise BASE.TrialAdmissibilityError(detf_min, activation_eigen_min)
        fit = (
            self.p.weights_t[:, None]
            * (u[self.p.top_t] - self.target[self.p.top_t]).square()
        ).sum() / self.scale**2
        magnitude = (
            (self.volume * H.square().sum(dim=(1, 2))).sum()
            / self.volume.sum()
            / 1.5
            / AREF**2
        )
        amplitude = local.square().sum(dim=1)
        smooth = (
            self.cfg.smooth_length**2
            * (self.ew * (amplitude[self.ei] - amplitude[self.ej]).square()).sum()
            / self.volume.sum()
            / AREF**2
        )
        # This diagnostic measures only within-muscle scalar-amplitude variation.
        # Its objective weight is pinned to zero for this region-shared screen.
        objective = fit + self.cfg.magnitude * magnitude
        adjoint = None
        if backward:
            objective.backward()
            adjoint = self.p.check_adjoint()
            if q.grad is None or not torch.isfinite(q.grad).all():
                raise RuntimeError(
                    "latent-region objective produced an invalid gradient"
                )
        self.calls += 1
        return {
            "value": float(objective.detach()),
            "fit": float(fit.detach()),
            "magnitude": float(magnitude.detach()),
            "smooth": float(smooth.detach()),
            "grad": q.grad.detach().clone() if backward else None,
            "u": u.detach().cpu().numpy().copy(),
            "A": ainv_np,
            "forward": dict(self.p.last_forward),
            "adjoint": adjoint,
        }


def activation_axes(ainv: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract deterministic unoriented principal axes from active-cell tensors."""
    ainv = np.asarray(ainv, dtype=np.float64)
    if ainv.ndim != 3 or ainv.shape[1:] != (3, 3):
        raise ValueError("A_inv must have shape (active_cells, 3, 3)")
    eigenvalues, eigenvectors = np.linalg.eigh(ainv)
    amplitude = np.maximum(np.log(eigenvalues[:, -1]), 0.0)
    defined = amplitude > AXIS_AMPLITUDE_TOLERANCE
    axis = np.zeros((len(ainv), 3), dtype=np.float64)
    axis[defined] = eigenvectors[defined, :, -1]
    if np.any(defined):
        selected = axis[defined]
        dominant = np.argmax(np.abs(selected), axis=1)
        sign = np.where(selected[np.arange(len(selected)), dominant] < 0.0, -1.0, 1.0)
        axis[defined] *= sign[:, None]
    return axis, defined, amplitude


class LatentFacePhysics(BASE.FacePhysics):
    """Original strict physics with latent-axis annotations on saved VTK files."""

    def save_mesh(
        self, path: Path, u: np.ndarray, ainv: np.ndarray | None = None
    ) -> None:
        super().save_mesh(path, u, ainv)
        if ainv is None:
            return
        active_axis, active_defined, active_amplitude = activation_axes(ainv)
        axis = np.zeros((len(self.tets), 3), dtype=np.float64)
        defined = np.zeros(len(self.tets), dtype=np.int8)
        amplitude = np.zeros(len(self.tets), dtype=np.float64)
        axis[self.ids] = active_axis
        defined[self.ids] = active_defined.astype(np.int8)
        amplitude[self.ids] = active_amplitude
        mesh = pv.read(path)
        mesh.cell_data["LearnedActivationAxis"] = axis
        mesh.cell_data["LearnedActivationAxisDefined"] = defined
        mesh.cell_data["LearnedActivationAmplitude"] = amplitude
        mesh.save(path)


def fixture_arrays(fixture: Path) -> dict[str, np.ndarray]:
    """Read only the rest arrays needed to construct the initialization."""
    mesh = pv.read(fixture / "volume.vtu")
    active_ids = np.flatnonzero(np.asarray(mesh.cell_data["ActivationMask"], bool))
    return {
        "points": np.asarray(mesh.points, dtype=np.float64),
        "active_ids": active_ids,
        "active_region_ids": np.asarray(
            mesh.cell_data["ActivationControlId"], dtype=np.int64
        )[active_ids],
        "active_fibers": np.asarray(
            mesh.cell_data["ActivationFiber"], dtype=np.float64
        )[active_ids],
        "region_muscle_ids": np.asarray(
            mesh.field_data["ActivationRegionMuscleId"], dtype=np.int64
        ),
    }


def prepare_initial(fixture: Path, output: Path) -> dict[str, Any]:
    """Write the nonzero q and zero equilibrium seed expected by base main."""
    volume_path = fixture / "volume.vtu"
    arrays = fixture_arrays(fixture)
    rings = ring_region_mask(arrays["region_muscle_ids"])
    q = initial_coordinates(arrays["active_region_ids"], arrays["active_fibers"], rings)
    u = np.zeros_like(arrays["points"])
    receipt = {
        "fixture_volume_sha256": file_sha256(volume_path),
        "generator_source_sha256": file_sha256(Path(__file__)),
        "active_ids_sha256": array_sha256(arrays["active_ids"]),
        "active_region_ids_sha256": array_sha256(arrays["active_region_ids"]),
        "active_fibers_sha256": array_sha256(arrays["active_fibers"]),
        "region_muscle_ids_sha256": array_sha256(arrays["region_muscle_ids"]),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite initial state: {output}")
    np.savez_compressed(
        output,
        q=q,
        u=u,
        initial_activation=np.asarray(INITIAL_ACTIVATION),
        ring_region_mask=rings,
        free_parameters=np.asarray(99, dtype=np.int64),
        stored_parameters=np.asarray(q.size, dtype=np.int64),
        **{name: np.asarray(value) for name, value in receipt.items()},
    )
    return {
        "output": str(output),
        "output_sha256": file_sha256(output),
        "q_shape": list(q.shape),
        "u_shape": list(u.shape),
        "free_parameters": 99,
        "stored_parameters": int(q.size),
        "inputs": receipt,
    }


def validate(fixture: Path, initial: Path | None = None) -> dict[str, Any]:
    """Run CPU-only actual-fixture, bound, zero, and derivative checks."""
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float64)
    arrays = fixture_arrays(fixture)
    ring_np = ring_region_mask(arrays["region_muscle_ids"])
    ring = torch.as_tensor(ring_np, dtype=torch.bool)
    region = torch.as_tensor(arrays["active_region_ids"], dtype=torch.int64)
    fibers = torch.as_tensor(arrays["active_fibers"], dtype=torch.float64)
    q_np = initial_coordinates(
        arrays["active_region_ids"], arrays["active_fibers"], ring_np
    )
    q = torch.as_tensor(q_np)
    local = local_vectors(q, region, fibers, ring)
    A, H = matrices_from_vectors(local)
    reference_a = torch.full((len(fibers), 1), INITIAL_ACTIVATION)
    reference_A, reference_H = BASE.am.matrices(reference_a, "F", fibers)
    H_error = float(torch.max(torch.abs(H - reference_H)))
    A_error = float(torch.max(torch.abs(A - reference_A)))
    if H_error > 2e-17 or A_error > 2e-15:
        raise AssertionError(
            f"initialization differs from FiberRegion: H={H_error}, A={A_error}"
        )

    amax = -math.log(0.65)
    radius = math.sqrt(amax)
    zero = torch.zeros_like(q)
    zero_local = local_vectors(zero, region, fibers, ring)
    zero_A, zero_H = matrices_from_vectors(zero_local)
    identity = torch.eye(3).expand(len(fibers), 3, 3)
    if not torch.equal(zero_H, torch.zeros_like(zero_H)) or not torch.equal(
        zero_A, identity
    ):
        raise AssertionError("zero q must produce H=0 and A_inv=I")
    if not torch.equal(project(zero, ring, amax), zero):
        raise AssertionError("zero q must be fixed by projection")
    probe = torch.full_like(q, 2.0)
    if not torch.equal(project(probe, ring, 0.0), zero):
        raise AssertionError("the explicit zero bound must project exactly to zero")
    projected = project(probe, ring, amax)
    if torch.linalg.vector_norm(projected[~ring], dim=1).max() > radius + 1e-14:
        raise AssertionError("non-ring row-ball projection exceeded its bound")
    if not torch.all((projected[ring, 0] >= 0.0) & (projected[ring, 0] <= radius)):
        raise AssertionError("ring scalar projection exceeded its interval")
    if not torch.equal(projected[ring, 1:], torch.zeros_like(projected[ring, 1:])):
        raise AssertionError("ring stored constraints must remain zero")

    toy_region = torch.tensor((0, 1), dtype=torch.int64)
    toy_fiber = torch.tensor(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    toy_ring = torch.tensor((False, True))
    toy_q = torch.tensor(((0.18, -0.07, 0.04), (0.21, 0.0, 0.0)), requires_grad=True)
    if not torch.autograd.gradcheck(
        lambda value: matrices_from_vectors(
            local_vectors(value, toy_region, toy_fiber, toy_ring)
        )[0],
        (toy_q,),
    ):
        raise AssertionError("mixed latent map failed gradcheck")
    jacobian = torch.autograd.functional.jacobian(
        lambda value: matrices_from_vectors(
            local_vectors(value, toy_region, toy_fiber, toy_ring)
        )[0],
        toy_q,
    )
    nonring_derivative = float(torch.linalg.vector_norm(jacobian[0, :, :, 0]))
    ring_derivative = float(torch.linalg.vector_norm(jacobian[1, :, :, 1, 0]))
    ring_fixed_derivative = float(torch.linalg.vector_norm(jacobian[1, :, :, 1, 1:]))
    zero_jacobian = torch.autograd.functional.jacobian(
        lambda value: matrices_from_vectors(
            local_vectors(value, toy_region, toy_fiber, toy_ring)
        )[0],
        torch.zeros_like(toy_q),
    )
    if nonring_derivative <= 0.0 or ring_derivative <= 0.0:
        raise AssertionError(
            "free ring and non-ring directions need nonzero derivatives"
        )
    if ring_fixed_derivative != 0.0 or not torch.equal(
        zero_jacobian, torch.zeros_like(zero_jacobian)
    ):
        raise AssertionError("fixed ring and zero-state derivatives must vanish")

    if initial is not None:
        with np.load(initial, allow_pickle=False) as saved:
            if not np.array_equal(saved["q"], q_np):
                raise AssertionError(
                    "saved initial q differs from fixture construction"
                )
            if not np.array_equal(saved["u"], np.zeros_like(arrays["points"])):
                raise AssertionError("saved initial u must be exactly zero")
            expected_hashes = {
                "fixture_volume_sha256": file_sha256(fixture / "volume.vtu"),
                "generator_source_sha256": file_sha256(Path(__file__)),
                "active_ids_sha256": array_sha256(arrays["active_ids"]),
                "active_region_ids_sha256": array_sha256(arrays["active_region_ids"]),
                "active_fibers_sha256": array_sha256(arrays["active_fibers"]),
                "region_muscle_ids_sha256": array_sha256(arrays["region_muscle_ids"]),
            }
            for name, expected in expected_hashes.items():
                if str(saved[name]) != expected:
                    raise AssertionError(f"stale initial receipt: {name}")

    return {
        "active_cells": len(region),
        "regions": len(q),
        "ring_regions": np.flatnonzero(ring_np).tolist(),
        "stored_shape": list(q.shape),
        "stored_parameters": int(q.numel()),
        "free_parameters": int((~ring).sum()) * 3 + int(ring.sum()),
        "initial_H_max_absolute_error": H_error,
        "initial_Ainv_max_absolute_error": A_error,
        "nonring_derivative_norm": nonring_derivative,
        "ring_free_derivative_norm": ring_derivative,
        "ring_fixed_derivative_norm": ring_fixed_derivative,
        "zero_derivative_norm": float(torch.linalg.vector_norm(zero_jacobian)),
        "max_projected_nonring_norm": float(
            torch.linalg.vector_norm(projected[~ring], dim=1).max()
        ),
        "max_projected_ring_scalar": float(projected[ring, 0].max()),
        "zero_bound_projection_is_exact_zero": True,
        "control_radius": radius,
        "initial_file_verified": initial is not None,
    }


def annotate_receipts(cfg: Config) -> None:
    """Add model-specific interpretation to the base run receipts."""
    summary_path = cfg.output_dir / "summary.json"
    provenance_path = cfg.output_dir / "provenance.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["parameterization"] = {
        "stored_shape": [35, 3],
        "stored_parameters": 105,
        "free_parameters": 99,
        "nonring_regions": 32,
        "nonring_free_parameters_per_region": 3,
        "ring_regions": 3,
        "ring_free_parameters_per_region": 1,
        "ring_fixed_stored_coordinates": 6,
        "nonring_bound": "row ball ||v|| <= sqrt(amax)",
        "ring_bound": "0 <= v[0] <= sqrt(amax); v[1:3] = 0",
    }
    summary["initialization"] = {
        "activation": INITIAL_ACTIVATION,
        "target_used": False,
        "source": "prepared rest ActivationFiber",
        "equivalence": "exact FiberRegion a=0.02 tensor field",
        "difference_from_main_screen": (
            "nonzero target-independent initialization; the main screen used zero"
        ),
        "interpretation": "fitted axes are latent; not identified anatomy",
    }
    summary["regularization"] = {
        "magnitude": "muscle-fraction-volume-weighted ||H||_F^2 / 1.5",
        "amplitude_smoothness": (
            "existing active-cell graph metric on ||local_v||^2 only"
        ),
        "amplitude_smoothness_interpretation": (
            "measures scalar-amplitude variation; does not measure fitted-axis variation"
        ),
        "amplitude_smoothness_objective_weight": 0.0,
    }
    summary["vtk_axis_fields"] = {
        "LearnedActivationAxis": "deterministically signed representative of an unoriented axis",
        "LearnedActivationAxisDefined": (
            "one only where amplitude exceeds the declared numerical tolerance"
        ),
        "LearnedActivationAmplitude": "squared latent-vector norm",
        "undefined_axis_storage": "zero vector with Defined=0",
        "amplitude_tolerance": AXIS_AMPLITUDE_TOLERANCE,
        "ring_axis_source": "prepared circumferential ActivationFiber",
    }
    BASE.write_json(summary_path, summary)

    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["latent_region_screen"] = {
        "base_runner": BASE_RUNNER.name,
        "base_runner_sha256": file_sha256(BASE_RUNNER),
        "initial_state": str(cfg.initial),
        "initial_state_sha256": file_sha256(cfg.initial),
        "free_parameters": 99,
        "stored_parameters": 105,
    }
    BASE.write_json(provenance_path, provenance)


def main(cfg: Config) -> None:
    """Run the imported optimizer with only the isolated model seams replaced."""
    if cfg.method != METHOD:
        raise ValueError(f"method is pinned to {METHOD!r}")
    if cfg.initial is None:
        raise ValueError("the target-independent nonzero initial state is required")
    if cfg.soft_nu != 0.46 or cfg.fat_nu != 0.49:
        raise ValueError("initial screen is pinned to muscle nu=0.46 and fat nu=0.49")
    BASE.Objective = Objective
    BASE.FacePhysics = LatentFacePhysics
    BASE.main(cfg)
    annotate_receipts(cfg)


if __name__ == "__main__":
    if sys.argv[1:] == ["--prepare-initial"]:
        print(json.dumps(prepare_initial(DEFAULT_FIXTURE, DEFAULT_INITIAL), indent=2))
    elif sys.argv[1:] == ["--cpu-validate"]:
        print(json.dumps(validate(DEFAULT_FIXTURE, DEFAULT_INITIAL), indent=2))
    else:
        cherries.main(
            main, profile=None if os.getenv("DEBUG") == "1" else ProfileCometNoCommit
        )
