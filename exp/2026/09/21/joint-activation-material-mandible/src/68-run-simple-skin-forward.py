# ruff: noqa: EM101, EM102, TRY003
"""Direct PNCG forward solve with prescribed heterogeneous skin properties.

This experiment deliberately has no inverse solve, adjoint, activation, or bulk
baseline stress.  Complete registered source bones are fixed IPC obstacles and
the only prescribed load is the literature-derived skin stress resultant.
"""

from __future__ import annotations

import copy
import faulthandler
import json
import logging
import os
import platform
import signal
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, override

import attrs
import numpy as np
import pydantic_settings as ps
import torch
from joint_common import (
    HISTORICAL,
    ProfileJoint,
    archive_sources,
    sha256,
    write_json,
)
from joint_data import PreparedInputs, array_sha256, file_sha256
from joint_equilibrium import configure_cuda
from joint_fields import BULK_TISSUES, research_informed_material_config
from joint_full_skull_contact import (
    FullSkullJointPhysics,
    load_admitted_initialization,
    load_full_skull_geometry,
)

from liblaf import cherries
from liblaf.apple.forward._problem import ForwardProblem

sys.path.insert(0, str(HISTORICAL))
from face_physics import (
    ForwardConvergenceError,
    StrictLineSearch,
    StrictPncg,
)

LOG = logging.getLogger(__name__)


class ForwardWallTimeExceededError(RuntimeError):
    """The declared forward wall budget was reached after an accepted step."""


class ForwardInterruptedError(RuntimeError):
    """A termination signal requested a checkpointed stop."""


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)

    prepared_dir: Path
    geometry: Path
    geometry_audit: Path
    admission: Path
    skin_field: Path
    skin_field_manifest: Path
    output_dir: Path = cherries.output("simple-skin-forward", mkdir=True)

    rtol: float = 1e-5
    atol: float = 1e-11
    max_steps: int = 5000
    line_search_max_steps: int = 40
    line_search_armijo: float = 1e-4
    directional_curvature: Literal["approximate", "exact_hvp"] = "approximate"
    max_step_norm_m: float = 5e-5
    volume_step_safety: float = 0.8
    allow_inverted_cells: bool = False
    common_poisson: float | None = None
    restart_checkpoint: Path | None = None
    force_threshold_override: float | None = None
    contact_collision_set_type: Literal["IPC", "IMPROVED_MAX_APPROX"] = (
        "IMPROVED_MAX_APPROX"
    )
    ccd_max_iterations: int = 10000000
    ccd_min_distance_m: float = 0.0
    pncg_restart_interval_steps: int = 0
    hessian_damping_initial: float = 0.0
    wall_cap_seconds: float = 1800.0
    watchdog_seconds: int = 60
    heartbeat_seconds: float = 15.0
    telemetry_interval_steps: int = 10
    checkpoint_interval_steps: int = 50


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def append_jsonl(path: Path, value: object) -> None:
    line = json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n"
    with path.open("a") as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())


def load_skin_field(  # noqa: C901, PLR0912
    field_path: Path,
    manifest_path: Path,
    *,
    prepared: PreparedInputs,
    expected_triangles: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load one exact per-face field in the model's frozen triangle order."""
    field_path = field_path.resolve()
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest["schema"] != "joint-prescribed-skin-field-v1":
        raise ValueError(f"unexpected skin-field schema: {manifest['schema']}")
    if manifest["success"] is not True:
        raise ValueError("skin-field producer did not report success")
    if manifest["artifact"]["sha256"] != file_sha256(field_path):
        raise ValueError("skin-field NPZ hash differs from its manifest")
    prepared_binding = manifest["prepared_inputs"]
    if prepared_binding["npz_sha256"] != file_sha256(prepared.npz_path):
        raise ValueError("skin field was produced for another prepared NPZ")
    if prepared_binding["manifest_sha256"] != file_sha256(prepared.manifest_path):
        raise ValueError("skin field was produced for another prepared manifest")
    with np.load(field_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    required = {"E_mpa", "nu", "h_m", "baseline_N_per_m", "skin_triangles"}
    if set(arrays) != required:
        raise ValueError(f"skin-field arrays must be exactly {sorted(required)}")
    triangles = arrays["skin_triangles"]
    if triangles.dtype.kind not in "iu" or not np.array_equal(
        triangles.astype(np.int64, copy=False), expected_triangles
    ):
        raise ValueError("skin-field triangle order differs from the model")
    count = len(expected_triangles)
    expected_shapes = {
        "E_mpa": (count,),
        "nu": (count,),
        "h_m": (count,),
        "baseline_N_per_m": (count, 2, 2),
        "skin_triangles": (count, 3),
    }
    for name, shape in expected_shapes.items():
        value = arrays[name]
        if value.shape != shape:
            raise ValueError(f"{name} has shape {value.shape}, expected {shape}")
        record = manifest["arrays"][name]
        if (
            record["shape"] != list(shape)
            or np.dtype(record["dtype"]).str != value.dtype.str
        ):
            raise ValueError(f"{name} layout differs from its manifest")
        if record["sha256"] != array_sha256(value):
            raise ValueError(f"{name} hash differs from its manifest")
    for name in ("E_mpa", "nu", "h_m", "baseline_N_per_m"):
        if arrays[name].dtype != np.float64 or not np.isfinite(arrays[name]).all():
            raise ValueError(f"{name} must be finite float64")
    if np.any(arrays["E_mpa"] <= 0):
        raise ValueError("skin E_mpa must be positive")
    if np.any((arrays["nu"] < 0) | (arrays["nu"] >= 0.5)):
        raise ValueError("skin nu must lie in [0,0.5)")
    if np.any(arrays["h_m"] <= 0):
        raise ValueError("skin h_m must be positive")
    baseline = arrays["baseline_N_per_m"]
    if not np.array_equal(baseline, baseline.transpose(0, 2, 1)):
        raise ValueError("baseline_N_per_m must be exactly symmetric")
    contract = manifest["constitutive_contract"]
    if contract != {
        "law": "exact plane-stress polynomial stable Neo-Hookean membrane",
        "baseline_units": "N/m",
        "baseline_frame": "frozen neutral per-triangle tangent frame",
        "elastic_modulus_units": "MPa",
        "thickness_units": "m",
        "triangle_order": "skin_triangles exact global FEM point ids",
    }:
        raise ValueError("skin-field constitutive contract changed")
    return arrays, manifest


def heterogeneous_materials(
    physics: FullSkullJointPhysics, skin: dict[str, np.ndarray]
) -> dict[str, dict[str, torch.Tensor]]:
    """Canonical passive bulk plus the prescribed per-face skin field."""
    zero_bulk = torch.zeros((3, 3, 3), dtype=torch.float64, device="cuda")
    zero_skin = torch.zeros((2, 2), dtype=torch.float64, device="cuda")
    values = physics.materials(
        zero_bulk,
        zero_skin,
        torch.ones((), dtype=torch.float64, device="cuda"),
        active_stress=None,
    )
    young = torch.as_tensor(skin["E_mpa"], dtype=torch.float64, device="cuda")
    nu = torch.as_tensor(skin["nu"], dtype=torch.float64, device="cuda")
    mu = young / (2 * (1 + nu))
    classical_lambda = young * nu / ((1 + nu) * (1 - 2 * nu))
    # This polynomial SNH implementation requires lambda_code=lambda_classical+mu
    # to reproduce the requested small-strain isotropic E and nu.
    values["skin"]["mu"] = mu
    values["skin"]["lmbda"] = classical_lambda + mu
    values["skin"]["thickness"] = torch.as_tensor(
        skin["h_m"], dtype=torch.float64, device="cuda"
    )
    values["skin"]["baseline_stress"] = torch.as_tensor(
        skin["baseline_N_per_m"] * 1e-6,
        dtype=torch.float64,
        device="cuda",
    )
    for name in BULK_TISSUES:
        active = values[name]["active_stress"]
        if torch.count_nonzero(active):
            raise AssertionError(f"nonzero bulk baseline stress in {name}")
    return values


class Heartbeat:
    def __init__(self, path: Path, *, interval: float, wall_started: float) -> None:
        self.path = path
        self.interval = interval
        self.wall_started = wall_started
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.current_operation = "setup"
        self.operation_started = time.perf_counter()
        self.accepted_steps = 0
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=2 * self.interval)

    def operation(self, name: str) -> None:
        with self.lock:
            self.current_operation = name
            self.operation_started = time.perf_counter()

    def accepted(self, step: int) -> None:
        with self.lock:
            self.accepted_steps = step

    def snapshot(self) -> dict[str, Any]:
        now = time.perf_counter()
        with self.lock:
            return {
                "schema": "joint-simple-forward-heartbeat-v1",
                "pid": os.getpid(),
                "wall_elapsed_seconds": now - self.wall_started,
                "accepted_steps": self.accepted_steps,
                "current_operation": self.current_operation,
                "current_operation_elapsed_seconds": now - self.operation_started,
                "unix_time": time.time(),
            }

    def _run(self) -> None:
        while not self.stop_event.wait(self.interval):
            write_json(self.path, self.snapshot())


class VolumeStepGuard:
    """Conservative no-inversion bound for every tetrahedron along a trial path.

    For current deformation gradient ``F`` and proposed increment ``G``, the
    path is ``F(t)=F+tG=F(I+t F^-1 G)``.  Requiring
    ``t ||F^-1 G||_F <= safety < 1`` keeps ``I+t F^-1 G`` nonsingular for the
    entire interval, so an initially positive determinant cannot change sign.
    """

    def __init__(self, physics: FullSkullJointPhysics, *, safety: float) -> None:
        if not 0 < safety < 1:
            raise ValueError("volume step safety must lie in (0,1)")
        self.safety = safety
        self.fem_node_count = physics.full_skull.geometry.fem_node_count
        self.points = torch.as_tensor(
            physics.points, dtype=torch.float64, device="cuda"
        )
        self.tets = torch.as_tensor(physics.tets, dtype=torch.int64, device="cuda")
        self.dm_inv = torch.as_tensor(
            physics.dm_inv, dtype=torch.float64, device="cuda"
        )
        self.last_receipt: dict[str, Any] | None = None

    def max_step_size(self, u: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        if u.shape != p.shape or u.shape[0] < self.fem_node_count:
            raise ValueError("volume guard requires matching full displacement arrays")
        with torch.no_grad():
            current = self.points + u[: self.fem_node_count]
            increment = p[: self.fem_node_count]
            x = current[self.tets]
            dx = increment[self.tets]
            ds = (x[:, 1:] - x[:, :1]).transpose(1, 2)
            dp = (dx[:, 1:] - dx[:, :1]).transpose(1, 2)
            deformation = ds @ self.dm_inv
            increment_gradient = dp @ self.dm_inv
            determinants = torch.linalg.det(deformation)
            minimum_det = torch.min(determinants)
            if not torch.isfinite(minimum_det) or minimum_det <= 0:
                message = "volume guard received a nonpositive current determinant"
                raise ForwardConvergenceError(
                    message,
                    receipt={"detF_min": float(minimum_det)},
                )
            relative = torch.linalg.solve(deformation, increment_gradient)
            maximum_norm = torch.amax(torch.linalg.matrix_norm(relative, ord="fro"))
            if not torch.isfinite(maximum_norm):
                message = "volume guard relative increment is nonfinite"
                raise ForwardConvergenceError(
                    message,
                    receipt={"relative_increment_frobenius_max": float(maximum_norm)},
                )
            fraction = torch.clamp(
                maximum_norm.new_tensor(self.safety) / maximum_norm,
                max=1.0,
            )
            if maximum_norm == 0:
                fraction = torch.ones_like(maximum_norm)
            self.last_receipt = {
                "method": "relative deformation Frobenius sufficient bound",
                "safety": self.safety,
                "current_detF_min": float(minimum_det),
                "relative_increment_frobenius_max": float(maximum_norm),
                "fraction": float(fraction),
                "guarantee": "positive determinant throughout the scaled linear trial path",
            }
            return fraction


class TimedForwardProblem(ForwardProblem):
    """ForwardProblem with synchronized operation timing and accepted callbacks."""

    def __init__(
        self,
        *args: Any,
        monitor: ForwardMonitor,
        volume_guard: VolumeStepGuard | None,
        directional_curvature: str = "approximate",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.monitor = monitor
        self.volume_guard = volume_guard
        self.directional_curvature = directional_curvature
        self.counts: defaultdict[str, int] = defaultdict(int)
        self.seconds: defaultdict[str, float] = defaultdict(float)

    def _timed(self, name: str, operation: Any) -> Any:
        self.monitor.heartbeat.operation(name)
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = operation()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        self.counts[name] += 1
        self.seconds[name] += elapsed
        self.monitor.last_operation = {"name": name, "seconds": elapsed}
        return result

    @override
    def max_step_size(self, state: Any, p: torch.Tensor) -> torch.Tensor:
        p_full = self.model.dof_map.to_full_grad(p)
        volume_fraction = (
            self._timed(
                "volume_step_bound",
                lambda: self.volume_guard.max_step_size(state.u, p_full),
            )
            if self.volume_guard is not None
            else p.new_tensor(1.0)
        )
        collision_fraction = self._timed(
            "ccd_max_step_size",
            lambda: self.model.max_step_size(state, volume_fraction * p_full),
        )
        return volume_fraction * collision_fraction

    @override
    def update(self, state: Any, u: torch.Tensor) -> None:
        parent = super().update
        self._timed("state_update", lambda: parent(state, u))

    @override
    def fun(self, state: Any) -> torch.Tensor:
        parent = super().fun
        return self._timed("energy", lambda: parent(state))

    @override
    def grad(self, state: Any) -> torch.Tensor:
        parent = super().grad
        return self._timed("gradient", lambda: parent(state))

    @override
    def hess_diag(self, state: Any) -> torch.Tensor:
        parent = super().hess_diag
        return self._timed("hessian_diagonal", lambda: parent(state))

    @override
    def hess_quad(self, state: Any, p: torch.Tensor) -> torch.Tensor:
        if self.directional_curvature == "exact_hvp":
            parent_hvp = super().hess_prod
            return self._timed(
                "hessian_quadratic_exact_hvp",
                lambda: torch.dot(p, parent_hvp(state, p)),
            )
        parent = super().hess_quad
        return self._timed("hessian_quadratic", lambda: parent(state, p))

    @override
    def callback(self, model_state: Any, opt_state: Any) -> None:
        self.monitor.accepted(self, model_state, opt_state)

    def timing_receipt(self) -> dict[str, Any]:
        return {
            name: {"count": self.counts[name], "seconds": self.seconds[name]}
            for name in sorted(self.counts)
        }


@attrs.define(kw_only=True)
class MonitoredPncg(StrictPncg):
    monitor: ForwardMonitor = attrs.field()
    restart_interval: int = 0

    @override
    def terminate(self, problem: Any, model_state: Any, opt_state: Any) -> Any:
        # Peach records the gradient before the accepted update. Convergence
        # must describe the state that will be saved, including when a step
        # crosses the tolerance in either direction.
        gradient = problem.grad(model_state)
        norm = torch.linalg.vector_norm(gradient)
        opt_state.convergence_state.grad_norm = norm
        self.monitor.last_force_norm = float(norm)
        return super().terminate(problem, model_state, opt_state)

    @override
    def step(self, problem: Any, model_state: Any, opt_state: Any) -> None:
        self.monitor.step_started = time.perf_counter()
        if self.restart_interval and opt_state.step % self.restart_interval == 0:
            # The PNCG direction implementation uses this flag to restart with
            # preconditioned steepest descent; Armijo is still enforced.
            opt_state.line_search_state.ok = False
        try:
            super().step(problem, model_state, opt_state)
        except ForwardConvergenceError as error:
            self.monitor.failed_trial = {
                "accepted": False,
                "optimizer_step_before_failure": int(opt_state.step),
                "failure": str(error),
                "receipt": copy.deepcopy(error.receipt),
            }
            raise


class ForwardMonitor:
    def __init__(
        self,
        *,
        output_dir: Path,
        physics: FullSkullJointPhysics,
        wall_started: float,
        wall_cap_seconds: float,
        telemetry_interval: int,
        checkpoint_interval: int,
        heartbeat: Heartbeat,
    ) -> None:
        self.output_dir = output_dir
        self.physics = physics
        self.wall_started = wall_started
        self.wall_cap_seconds = wall_cap_seconds
        self.telemetry_interval = telemetry_interval
        self.checkpoint_interval = checkpoint_interval
        self.heartbeat = heartbeat
        self.trace_path = output_dir / "trace.jsonl"
        self.step_started = wall_started
        self.initial_force_norm: float | None = None
        self.last_force_norm: float | None = None
        self.last_operation: dict[str, Any] | None = None
        self.last_accepted_u: torch.Tensor | None = None
        self.last_accepted_step = 0
        self.failed_trial: dict[str, Any] | None = None
        self.checkpoints: list[dict[str, Any]] = []

    def accepted(
        self, problem: TimedForwardProblem, state: Any, opt_state: Any
    ) -> None:
        step = int(opt_state.step)
        self.last_accepted_u = state.u.detach().clone()
        self.last_accepted_step = step
        self.heartbeat.accepted(step)
        row: dict[str, Any] = {
            "schema": "joint-simple-forward-step-v1",
            "accepted": True,
            "step": step,
            "wall_elapsed_seconds": time.perf_counter() - self.wall_started,
            "step_seconds": time.perf_counter() - self.step_started,
            "energy": float(opt_state.fun),
            "pre_step_free_force_norm": float(opt_state.convergence_state.grad_norm),
            "pre_step_force_evaluation": (
                "PNCG gradient at the state from which this accepted step started"
            ),
            "stagnation_count": int(opt_state.convergence_state.stagnation_count),
            "directional_slope": float(opt_state.slope),
            "direction_inf_norm": float(
                torch.linalg.vector_norm(opt_state.direction, ord=torch.inf)
            ),
            "direction_hessian_quadratic": float(opt_state.hess_quad),
            "hessian_damping_factor": float(opt_state.hess_damping_state.factor),
            "line_search": {
                "ok": bool(opt_state.line_search_state.ok),
                "step": int(opt_state.line_search_state.step),
                "alpha": float(opt_state.line_search_state.alpha),
                "f0": float(opt_state.line_search_state.f0),
                "f_alpha": float(opt_state.line_search_state.f_alpha),
            },
            "last_operation": self.last_operation,
            "volume_guard": (
                copy.deepcopy(problem.volume_guard.last_receipt)
                if problem.volume_guard is not None
                else None
            ),
        }
        if step % self.telemetry_interval == 0:
            accepted_force = problem.grad(state)
            self.last_force_norm = float(torch.linalg.vector_norm(accepted_force))
            row["accepted_state_free_force_norm"] = self.last_force_norm
            row["contact"] = contact_receipt(self.physics)
            LOG.info(
                "Accepted PNCG step %d: energy %.9g, exact free-force %.6g, wall %.1fs",
                step,
                row["energy"],
                self.last_force_norm,
                row["wall_elapsed_seconds"],
            )
        append_jsonl(self.trace_path, row)
        if step == 1 or step % self.checkpoint_interval == 0:
            self.checkpoint(problem, state, opt_state, label=f"step-{step:05d}")
        if "contact" in row and row["contact"]["contact_numerically_valid"] is not True:
            raise ForwardConvergenceError(
                "accepted state violates the repulsive bone-contact contract",
                receipt=row["contact"],
            )
        if time.perf_counter() - self.wall_started >= self.wall_cap_seconds:
            if step % self.checkpoint_interval:
                self.checkpoint(
                    problem, state, opt_state, label=f"wall-cap-step-{step:05d}"
                )
            raise ForwardWallTimeExceededError(
                f"forward wall cap {self.wall_cap_seconds:g}s reached after accepted step {step}"
            )

    def checkpoint(
        self,
        _problem: TimedForwardProblem | None,
        state: Any,
        opt_state: Any | None,
        *,
        label: str,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        fem_count = self.physics.full_skull.geometry.fem_node_count
        displacement = np.ascontiguousarray(
            state.u[:fem_count].detach().cpu().numpy(), dtype=np.float64
        )
        path = self.output_dir / f"checkpoint-{label}.npz"
        atomic_npz(path, displacement_m=displacement)
        receipt = {
            "label": label,
            "accepted": True,
            "step": int(opt_state.step)
            if opt_state is not None
            else self.last_accepted_step,
            "path": str(path.resolve()),
            "sha256": sha256(path),
            "array_sha256": array_sha256(displacement),
            "wall_elapsed_seconds": time.perf_counter() - self.wall_started,
            "write_seconds": time.perf_counter() - started,
            "metrics": self.physics.metrics(state.u[:fem_count]),
            "contact": contact_receipt(self.physics),
        }
        write_json(path.with_suffix(".json"), receipt)
        self.checkpoints.append(receipt)
        LOG.info("Persisted accepted checkpoint %s", path.name)
        return receipt


def contact_receipt(physics: FullSkullJointPhysics) -> dict[str, Any]:
    model = physics.runtime.forward.model
    assert model.collision is not None
    assert physics.runtime.forward.state.collision is not None
    return model.collision.diagnostics(
        physics.runtime.forward.state.collision, physics.runtime.forward.state.u
    )


def force_component_receipt(physics: FullSkullJointPhysics) -> dict[str, Any]:
    """Separate tissue and contact gradients on the free displacement DOFs."""
    model = physics.runtime.forward.model
    state = physics.runtime.forward.state
    collision = model.collision
    assert collision is not None
    assert state.collision is not None
    torch.cuda.synchronize()
    started = time.perf_counter()
    tissue_full = torch.zeros_like(state.u)
    model.warp_model.grad(state.u, tissue_full)
    torch.cuda.synchronize()
    tissue_seconds = time.perf_counter() - started
    started = time.perf_counter()
    contact_full = torch.zeros_like(state.u)
    collision.grad(state.collision, state.u, contact_full)
    torch.cuda.synchronize()
    contact_seconds = time.perf_counter() - started
    tissue = model.dof_map.to_free_grad(tissue_full)
    contact = model.dof_map.to_free_grad(contact_full)
    total = tissue + contact
    tissue_norm = float(torch.linalg.vector_norm(tissue))
    contact_norm = float(torch.linalg.vector_norm(contact))
    total_norm = float(torch.linalg.vector_norm(total))
    return {
        "gradient_sign_convention": "reported norms are energy gradients; mechanical force is their negative",
        "tissue_free_gradient_norm": tissue_norm,
        "contact_free_gradient_norm": contact_norm,
        "total_free_gradient_norm": total_norm,
        "contact_to_tissue_norm_ratio": contact_norm / max(tissue_norm, 1e-300),
        "tissue_gradient_seconds": tissue_seconds,
        "contact_gradient_seconds": contact_seconds,
    }


def runtime_identity() -> dict[str, Any]:
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gpu": props.name,
        "gpu_total_memory_bytes": props.total_memory,
        "torch_threads": torch.get_num_threads(),
        "pid": os.getpid(),
    }


def main(cfg: Config) -> None:  # noqa: C901, PLR0912, PLR0915
    if (
        min(
            cfg.rtol,
            cfg.atol,
            cfg.wall_cap_seconds,
            cfg.max_step_norm_m,
            cfg.volume_step_safety,
            cfg.heartbeat_seconds,
            cfg.telemetry_interval_steps,
            cfg.checkpoint_interval_steps,
        )
        <= 0
    ):
        raise ValueError("all tolerances, intervals, and budgets must be positive")
    if cfg.max_steps <= 0 or cfg.line_search_max_steps <= 0:
        raise ValueError("step budgets must be positive")
    if not 0 < cfg.line_search_armijo < 1:
        raise ValueError("Armijo decrease coefficient must lie in (0, 1)")
    if cfg.ccd_max_iterations <= 0:
        raise ValueError("CCD iteration budget must be positive")
    if not 0 <= cfg.ccd_min_distance_m < 0.0001:
        raise ValueError("CCD clearance must lie in [0, dhat)")
    if cfg.pncg_restart_interval_steps < 0 or cfg.hessian_damping_initial < 0:
        raise ValueError("PNCG restart interval and Hessian damping cannot be negative")
    if cfg.common_poisson is not None and not 0 < cfg.common_poisson < 0.5:
        raise ValueError("common Poisson ratio must lie in (0, 0.5)")
    if cfg.force_threshold_override is not None and cfg.force_threshold_override <= 0:
        raise ValueError("force threshold override must be positive")
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    archive_sources(cfg.output_dir)
    configure_cuda()
    wall_started = time.perf_counter()
    heartbeat = Heartbeat(
        cfg.output_dir / "heartbeat.json",
        interval=cfg.heartbeat_seconds,
        wall_started=wall_started,
    )
    heartbeat.start()
    completed = False
    status = "setup_failed"
    failure: dict[str, Any] | None = None
    monitor: ForwardMonitor | None = None
    timed_problem: TimedForwardProblem | None = None
    physics: FullSkullJointPhysics | None = None
    watchdog_stream = (cfg.output_dir / "watchdog-tracebacks.log").open("a")
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def request_stop(signum: int, _frame: Any) -> None:
        raise ForwardInterruptedError(f"received signal {signum}")

    signal.signal(signal.SIGTERM, request_stop)
    faulthandler.dump_traceback_later(
        cfg.watchdog_seconds, repeat=True, file=watchdog_stream
    )
    try:
        prepared = PreparedInputs.load(
            cfg.prepared_dir / "inputs.npz", cfg.prepared_dir / "manifest.json"
        )
        geometry = load_full_skull_geometry(cfg.geometry, cfg.geometry_audit)
        admission = json.loads(cfg.admission.read_text())
        admitted_seed = load_admitted_initialization(admission, geometry)
        canonical = research_informed_material_config()["materials"]
        skin_reference = canonical["skin"]
        contact = {
            "schema": "joint-full-source-bone-contact-v1",
            "enabled": True,
            "surface_selection": "pure-soft-vs-complete-source-bones",
            "attachment_policy": "no-source-triangle-exclusions",
            "friction": "frictionless",
            "dhat_m": 0.0001,
            "stiffness_mpa": 0.01,
            "collision_set_type": cfg.contact_collision_set_type,
            "ccd_max_iterations": cfg.ccd_max_iterations,
            "ccd_min_distance_m": cfg.ccd_min_distance_m,
            "barrier_dmin_m": 0.0,
            "ccd_clearance_definition": (
                "Numerical swept-path feasible-set buffer only; barrier dmin, energy and derivatives stay unchanged. Final exact unprojected force must meet the original tolerance."
            ),
        }
        physics = FullSkullJointPhysics(
            prepared.volume_path,
            prepared.skin_path,
            prepared.arrays,
            bulk_young_mpa={
                name: canonical[name]["young_mpa"] for name in BULK_TISSUES
            },
            bulk_nu={
                name: cfg.common_poisson
                if cfg.common_poisson is not None
                else canonical[name]["poisson"]
                for name in BULK_TISSUES
            },
            skin_young_mpa=skin_reference["reference_map"]["young_mpa"],
            skin_nu=cfg.common_poisson
            if cfg.common_poisson is not None
            else skin_reference["poisson"],
            thickness_m=skin_reference["thickness_m"],
            full_skull_geometry=geometry,
            full_skull_admission=admission,
            full_skull_contact_config=contact,
            rtol=cfg.rtol,
            atol=cfg.atol,
            max_steps=cfg.max_steps,
            forward_method="pncg",
            adjoint_rtol=1e-7,
        )
        skin, skin_manifest = load_skin_field(
            cfg.skin_field,
            cfg.skin_field_manifest,
            prepared=prepared,
            expected_triangles=physics.skin_tri,
        )
        if cfg.common_poisson is not None:
            assert np.all(skin["nu"] == cfg.common_poisson), (
                "skin artifact must match common Poisson ratio"
            )
        materials = heterogeneous_materials(physics, skin)
        model = physics.runtime.forward.model
        model.set_materials(materials)
        pose = torch.zeros(6, dtype=torch.float64, device="cuda")
        fixed = physics.boundary(pose)
        model.dof_map.fixed_values = fixed.detach().clone()
        fem_seed = torch.as_tensor(admitted_seed, dtype=torch.float64, device="cuda")
        if cfg.restart_checkpoint is not None:
            restart_receipt = json.loads(
                cfg.restart_checkpoint.with_suffix(".json").read_text()
            )
            assert sha256(cfg.restart_checkpoint) == restart_receipt["sha256"]
            with np.load(cfg.restart_checkpoint, allow_pickle=False) as archive:
                restored = archive["displacement_m"]
            assert restored.shape == admitted_seed.shape
            assert np.isfinite(restored).all()
            fem_seed = torch.as_tensor(restored, dtype=torch.float64, device="cuda")
        full_seed = physics.full_skull.extend_seed(fem_seed, pose)
        projected = model.dof_map.to_full(model.dof_map.to_free(full_seed)).detach()
        collision = model.collision
        assert collision is not None
        prior_contact = collision.state_at(full_seed)
        boundary_fraction = float(
            collision.max_step_size(prior_contact, full_seed, projected - full_seed)
        )
        if boundary_fraction != 1.0:
            raise ValueError(
                f"zero-pose boundary projection fails CCD: {boundary_fraction}"
            )
        forward = physics.runtime.forward
        forward.state.u = projected.clone()
        forward.state.collision = collision.state_at(forward.state.u)
        forward.state.collision.boundary_ccd_fraction = boundary_fraction
        monitor = ForwardMonitor(
            output_dir=cfg.output_dir,
            physics=physics,
            wall_started=wall_started,
            wall_cap_seconds=cfg.wall_cap_seconds,
            telemetry_interval=cfg.telemetry_interval_steps,
            checkpoint_interval=cfg.checkpoint_interval_steps,
            heartbeat=heartbeat,
        )
        monitor.last_accepted_u = forward.state.u.detach().clone()
        volume_guard = (
            None
            if cfg.allow_inverted_cells
            else VolumeStepGuard(physics, safety=cfg.volume_step_safety)
        )
        timed_problem = TimedForwardProblem(
            model=model,
            monitor=monitor,
            volume_guard=volume_guard,
            directional_curvature=cfg.directional_curvature,
        )
        default = forward.default_optimizer(
            max_steps=cfg.max_steps,
            rtol=0.0 if cfg.force_threshold_override is not None else cfg.rtol,
            atol=cfg.force_threshold_override
            if cfg.force_threshold_override is not None
            else cfg.atol,
        )
        optimizer = MonitoredPncg(
            criteria=default.criteria,
            hess_damping=StrictPncg.HessianDamping(initial=cfg.hessian_damping_initial),
            line_search=StrictLineSearch(
                armijo=cfg.line_search_armijo,
                max_steps=cfg.line_search_max_steps,
                max_step_norm=cfg.max_step_norm_m,
            ),
            monitor=monitor,
            restart_interval=cfg.pncg_restart_interval_steps,
        )
        forward.optimizer = optimizer
        protocol = {
            "schema": "joint-simple-prescribed-skin-forward-protocol-v1",
            "solver": {
                "method": "direct strict PNCG only",
                "rtol": cfg.rtol,
                "atol": cfg.atol,
                "max_steps": cfg.max_steps,
                "hessian_damping_initial": cfg.hessian_damping_initial,
                "pncg_restart_interval_steps": cfg.pncg_restart_interval_steps,
                "line_search": "strict Armijo",
                "line_search_armijo": cfg.line_search_armijo,
                "directional_curvature": cfg.directional_curvature,
                "termination_force": "exact free gradient recomputed at each accepted state",
                "line_search_max_steps": cfg.line_search_max_steps,
                "max_step_norm_m": cfg.max_step_norm_m,
                "max_step_norm_definition": (
                    "configured absolute infinity-norm cap on each proposed free-displacement step before CCD"
                ),
                "max_step_norm_over_dhat": cfg.max_step_norm_m / contact["dhat_m"],
                "volume_step_guard": {
                    "enabled": not cfg.allow_inverted_cells,
                    "method": "relative deformation Frobenius sufficient bound",
                    "safety": cfg.volume_step_safety,
                    "order": "scale proposed displacement for positive determinant, then run complete-source CCD",
                    "endpoint_only_check": False,
                },
                "inversion_policy": "diagnostic only; user permits inverted cells"
                if cfg.allow_inverted_cells
                else "positive determinant required",
                "force_threshold_override": cfg.force_threshold_override,
                "fallback": None,
                "adjoint": False,
                "inverse": False,
            },
            "monitoring": {
                "telemetry_interval_steps": cfg.telemetry_interval_steps,
                "checkpoint_interval_steps": cfg.checkpoint_interval_steps,
                "wall_cap_seconds": cfg.wall_cap_seconds,
                "watchdog_seconds": cfg.watchdog_seconds,
                "heartbeat_seconds": cfg.heartbeat_seconds,
                "operation_timing": "CUDA-synchronized calls; instrumentation changes wall overhead but not numerical operations",
            },
            "mechanics": {
                "bulk": "canonical passive SNH with exactly zero additive stress",
                "activation": "none",
                "skin": "prescribed heterogeneous plane-stress SNH stiffness, thickness, and signed baseline resultant",
                "jaw_pose_rad_m": [0.0] * 6,
                "contact": contact,
                "bone_bone_contact": False,
                "poisson_ratios": {
                    **{
                        name: cfg.common_poisson
                        if cfg.common_poisson is not None
                        else canonical[name]["poisson"]
                        for name in BULK_TISSUES
                    },
                    "skin_range": [float(skin["nu"].min()), float(skin["nu"].max())],
                },
            },
            "inputs": {
                "prepared_npz": str(prepared.npz_path),
                "prepared_npz_sha256": file_sha256(prepared.npz_path),
                "prepared_manifest": str(prepared.manifest_path),
                "prepared_manifest_sha256": file_sha256(prepared.manifest_path),
                "geometry": physics.full_skull_receipt(),
                "admission_path": str(cfg.admission.resolve()),
                "admission_sha256": sha256(cfg.admission),
                "skin_field_path": str(cfg.skin_field.resolve()),
                "skin_field_sha256": sha256(cfg.skin_field),
                "skin_field_manifest_path": str(cfg.skin_field_manifest.resolve()),
                "skin_field_manifest_sha256": sha256(cfg.skin_field_manifest),
                "skin_field_manifest": skin_manifest,
                "restart_checkpoint": str(cfg.restart_checkpoint.resolve())
                if cfg.restart_checkpoint is not None
                else None,
                "restart_checkpoint_sha256": sha256(cfg.restart_checkpoint)
                if cfg.restart_checkpoint is not None
                else None,
            },
            "runtime": runtime_identity(),
        }
        write_json(cfg.output_dir / "protocol.json", protocol)
        initial_energy = timed_problem.fun(forward.state)
        initial_gradient = timed_problem.grad(forward.state)
        initial_force_norm = float(torch.linalg.vector_norm(initial_gradient))
        monitor.initial_force_norm = initial_force_norm
        monitor.last_force_norm = initial_force_norm
        initial_components = force_component_receipt(physics)
        component_match = abs(
            initial_components["total_free_gradient_norm"] - initial_force_norm
        ) / max(initial_force_norm, 1e-300)
        if component_match > 1e-12:
            raise AssertionError(
                f"initial force component norm mismatch: {component_match}"
            )
        initial_contact = contact_receipt(physics)
        if initial_contact["contact_numerically_valid"] is not True:
            message = "initial complete-source contact state is numerically invalid"
            raise ForwardConvergenceError(  # noqa: TRY301
                message, receipt=initial_contact
            )
        append_jsonl(
            monitor.trace_path,
            {
                "schema": "joint-simple-forward-step-v1",
                "accepted": True,
                "step": 0,
                "wall_elapsed_seconds": time.perf_counter() - wall_started,
                "energy": float(initial_energy),
                "accepted_state_free_force_norm": initial_force_norm,
                "force_components": initial_components,
                "contact": initial_contact,
            },
        )
        monitor.checkpoint(timed_problem, forward.state, None, label="step-00000")
        LOG.info(
            "Starting direct PNCG: initial free-force %.6g, threshold %.6g",
            initial_force_norm,
            cfg.force_threshold_override
            if cfg.force_threshold_override is not None
            else max(cfg.atol, cfg.rtol * initial_force_norm),
        )
        solution = optimizer.minimize(timed_problem, forward.state, forward.free)
        forward.last_solution = solution
        final_gradient = timed_problem.grad(forward.state)
        final_force_norm = float(torch.linalg.vector_norm(final_gradient))
        monitor.last_force_norm = final_force_norm
        threshold = (
            cfg.force_threshold_override
            if cfg.force_threshold_override is not None
            else max(cfg.atol, cfg.rtol * initial_force_norm)
        )
        converged = bool(solution.success) and final_force_norm <= threshold
        if not converged:
            raise ForwardConvergenceError(  # noqa: TRY301
                "PNCG result failed the independent terminal force check",
                receipt={
                    "result": str(solution.result),
                    "reported_success": bool(solution.success),
                    "final_force_norm": final_force_norm,
                    "force_threshold": threshold,
                },
            )
        completed = True
        status = "converged_simple_skin_forward"
    except (
        ForwardConvergenceError,
        ForwardWallTimeExceededError,
        ForwardInterruptedError,
    ) as error:
        status = {
            ForwardConvergenceError: "numerical_failure",
            ForwardWallTimeExceededError: "wall_cap_reached",
            ForwardInterruptedError: "interrupted",
        }[type(error)]
        failure = {
            "type": type(error).__name__,
            "message": str(error),
            "receipt": copy.deepcopy(getattr(error, "receipt", None)),
        }
        if isinstance(error, ForwardInterruptedError):
            LOG.warning("Simple forward checkpointed stop: %s", failure)
        else:
            LOG.exception("Simple forward stopped: %s", failure)
    except KeyboardInterrupt as error:
        status = "interrupted"
        failure = {"type": type(error).__name__, "message": "keyboard interrupt"}
        LOG.warning("Simple forward interrupted")
    finally:
        faulthandler.cancel_dump_traceback_later()
        signal.signal(signal.SIGTERM, previous_sigterm)
        heartbeat.stop()
        write_json(cfg.output_dir / "heartbeat.json", heartbeat.snapshot())
        watchdog_stream.close()

    final: dict[str, Any] = {
        "schema": "joint-simple-prescribed-skin-forward-v1",
        "success": completed,
        "status": status,
        "failure": failure,
        "wall_seconds": time.perf_counter() - wall_started,
        "accepted_steps": monitor.last_accepted_step if monitor is not None else 0,
        "initial_free_force_norm": (
            monitor.initial_force_norm if monitor is not None else None
        ),
        "final_free_force_norm": monitor.last_force_norm
        if monitor is not None
        else None,
        "force_threshold": (
            cfg.force_threshold_override
            if cfg.force_threshold_override is not None
            else max(cfg.atol, cfg.rtol * monitor.initial_force_norm)
            if monitor is not None and monitor.initial_force_norm is not None
            else None
        ),
        "failed_trial": monitor.failed_trial if monitor is not None else None,
        "operation_timings": (
            timed_problem.timing_receipt() if timed_problem is not None else None
        ),
        "protocol_sha256": (
            sha256(cfg.output_dir / "protocol.json")
            if (cfg.output_dir / "protocol.json").exists()
            else None
        ),
        "complete_source_soft_bone_contact": True,
        "bone_bone_contact": False,
        "bulk_baseline_stress": 0,
        "activation": "none",
        "inverse_solve": False,
        "adjoint_solve": False,
        "fallback": None,
        "inversions_allowed": cfg.allow_inverted_cells,
        "common_poisson": cfg.common_poisson,
    }
    if (
        physics is not None
        and monitor is not None
        and monitor.last_accepted_u is not None
    ):
        # Restore the last accepted state if a failed line-search left the live model
        # at a rejected trial. Only the accepted displacement may enter evidence.
        model = physics.runtime.forward.model
        model.update(physics.runtime.forward.state, monitor.last_accepted_u)
        assert timed_problem is not None
        terminal_gradient = timed_problem.grad(physics.runtime.forward.state)
        terminal_force_norm = float(torch.linalg.vector_norm(terminal_gradient))
        monitor.last_force_norm = terminal_force_norm
        final["final_free_force_norm"] = terminal_force_norm
        if (
            monitor.last_accepted_step % cfg.checkpoint_interval_steps
            or not monitor.checkpoints
        ):
            checkpoint = monitor.checkpoint(
                timed_problem,
                physics.runtime.forward.state,
                None,
                label=f"terminal-step-{monitor.last_accepted_step:05d}",
            )
        else:
            checkpoint = monitor.checkpoints[-1]
        fem_u = monitor.last_accepted_u[: physics.full_skull.geometry.fem_node_count]
        metrics = physics.metrics(fem_u)
        contact = contact_receipt(physics)
        components = force_component_receipt(physics)
        component_match = abs(
            components["total_free_gradient_norm"] - terminal_force_norm
        ) / max(terminal_force_norm, 1e-300)
        physical_validity = {
            "finite_displacement": bool(torch.isfinite(fem_u).all()),
            "no_inverted_tetrahedra": metrics["inverted_tetrahedra"] == 0,
            "positive_minimum_detF": metrics["detF_min"] > 0,
            "contact_numerically_valid": contact["contact_numerically_valid"],
            "force_component_relative_norm_match": component_match,
        }
        physically_valid = bool(
            physical_validity["finite_displacement"]
            and physical_validity["no_inverted_tetrahedra"]
            and physical_validity["positive_minimum_detF"]
            and physical_validity["contact_numerically_valid"]
            and component_match <= 1e-12
        )
        numerically_valid = bool(
            physical_validity["finite_displacement"]
            and physical_validity["contact_numerically_valid"]
            and component_match <= 1e-12
        )
        terminal_force_valid = bool(
            monitor.initial_force_norm is not None
            and terminal_force_norm <= final["force_threshold"]
        )
        admissible = numerically_valid if cfg.allow_inverted_cells else physically_valid
        if completed and (not admissible or not terminal_force_valid):
            completed = False
            status = "terminal_physical_or_force_gate_failed"
            final["success"] = False
            final["status"] = status
        final.update(
            {
                "checkpoint": checkpoint,
                "metrics": metrics,
                "contact": contact,
                "force_components": components,
                "terminal_force_gate_met": terminal_force_valid,
                "physical_validity": physical_validity,
                "physically_valid": physically_valid,
                "numerically_valid": numerically_valid,
            }
        )
        final["operation_timings"] = timed_problem.timing_receipt()
    write_json(cfg.output_dir / "summary.json", final)
    cherries.log_output(cfg.output_dir)
    if not completed:
        raise RuntimeError(f"simple forward did not converge: {status}")


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
