# ruff: noqa: PLR0915
"""CPU-only two-tetrahedron reachability study for tensor active stress.

The fixture has two mixed-material active tetrahedra joined at one triangular
face.  It is deliberately too small to support anatomical claims.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pydantic_settings as ps
import pyvista as pv
import torch
from experiment_profile import ProfileCometNoCommit
from scipy.optimize import least_squares, minimize

from liblaf import cherries

LOG = logging.getLogger(__name__)
DTYPE = torch.float64
# Both cells meet at face [1, 2, 3]; its edge [1, 2] is the x-axis unit edge.
TETS = np.asarray(((0, 1, 2, 3), (1, 2, 3, 4)), dtype=np.int64)
X0 = np.asarray(
    ((0, 0, -1), (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)),
    dtype=np.float64,
)
FRACTION = np.asarray((0.3, 0.8), dtype=np.float64)
# Remove global rigid modes without constraining the shared x edge.
FIXED = np.asarray((3, 4, 5, 7, 8, 11), dtype=np.int64)
FREE = np.setdiff1d(np.arange(X0.size), FIXED)
FORWARD_GRADIENT_INF_TOL = 1e-8
EDGE = np.asarray((1, 2), dtype=np.int64)


class Config(cherries.BaseConfig):
    """Numerical settings for the bounded, CPU-only fixture."""

    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output_dir: Path = cherries.output("12-small-model-study", mkdir=True)
    muscle_E_mpa: float = 0.03
    fat_E_mpa: float = 0.003
    poisson_ratio: float = 0.49
    outer_maxiter: int = 120
    inner_maxiter: int = 400


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def stable_parameters(young: float, nu: float) -> tuple[float, float]:
    return young / (2 * (1 + nu)), young * nu / ((1 + nu) * (1 - 2 * nu))


def unpack_symmetric(raw: np.ndarray) -> np.ndarray:
    return np.asarray(
        ((raw[0], raw[3], raw[5]), (raw[3], raw[1], raw[4]), (raw[5], raw[4], raw[2])),
        dtype=np.float64,
    )


def pack_symmetric(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(
        (
            matrix[0, 0],
            matrix[1, 1],
            matrix[2, 2],
            matrix[0, 1],
            matrix[1, 2],
            matrix[0, 2],
        )
    )


def project_psd(raw: np.ndarray, cap: float) -> tuple[np.ndarray, np.ndarray, bool]:
    values, vectors = np.linalg.eigh(unpack_symmetric(raw))
    clipped = np.clip(values, 0.0, cap)
    return (vectors * clipped) @ vectors.T, clipped, bool(np.any(values != clipped))


@dataclass(frozen=True)
class Fixture:
    x: np.ndarray
    tets: np.ndarray
    dm_inv: np.ndarray
    volumes: np.ndarray

    @classmethod
    def build(cls) -> Fixture:
        dm = np.asarray([(X0[t[1:]] - X0[t[0]]).T for t in TETS])
        return cls(X0, TETS, np.linalg.inv(dm), np.abs(np.linalg.det(dm)) / 6)


def rigid_mode_constraint_rank() -> int:
    """Return the rank by which fixed DOFs constrain 3 translations and 3 rotations."""
    translations = [np.tile(axis, len(X0)) for axis in np.eye(3)]
    rotations = [
        np.cross(np.broadcast_to(axis, X0.shape), X0).ravel() for axis in np.eye(3)
    ]
    constraint = np.column_stack((*translations, *rotations))[FIXED]
    return int(np.linalg.matrix_rank(constraint))


def deformation_gradients(x: torch.Tensor, fixture: Fixture) -> torch.Tensor:
    values = []
    for tet, dm_inv in zip(fixture.tets, fixture.dm_inv, strict=True):
        ds = torch.stack(tuple(x[i] - x[tet[0]] for i in tet[1:]), dim=1)
        values.append(ds @ torch.as_tensor(dm_inv, dtype=DTYPE))
    return torch.stack(values)


def passive_energy(F: torch.Tensor, young: float, nu: float) -> torch.Tensor:
    mu, lam = stable_parameters(young, nu)
    j = torch.linalg.det(F)
    return 0.5 * mu * ((F * F).sum() - 3) - mu * (j - 1) + 0.5 * lam * (j - 1) ** 2


def total_energy(
    x: torch.Tensor,
    fixture: Fixture,
    cfg: Config,
    q_cells: np.ndarray | None,
    external: np.ndarray | None = None,
) -> torch.Tensor:
    fs = deformation_gradients(x, fixture)
    energy = torch.zeros((), dtype=DTYPE)
    identity = torch.eye(3, dtype=DTYPE)
    for cell, F in enumerate(fs):
        fraction = float(FRACTION[cell])
        mixture = fraction * passive_energy(F, cfg.muscle_E_mpa, cfg.poisson_ratio)
        mixture += (1 - fraction) * passive_energy(F, cfg.fat_E_mpa, cfg.poisson_ratio)
        if q_cells is not None:
            q = torch.as_tensor(q_cells[cell], dtype=DTYPE)
            mixture += fraction * 0.5 * torch.sum(q * (F.T @ F - identity))
        energy += fixture.volumes[cell] * mixture
    if external is not None:
        energy -= torch.sum(x * torch.as_tensor(external, dtype=DTYPE))
    return energy


def solve_equilibrium(
    fixture: Fixture,
    cfg: Config,
    q_cells: np.ndarray | None = None,
    *,
    external: np.ndarray | None = None,
    start: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    base = X0.copy() if start is None else start.copy()

    def objective(free: np.ndarray) -> tuple[float, np.ndarray]:
        full = base.ravel().copy()
        full[FREE] = free
        x = torch.tensor(full.reshape(X0.shape), dtype=DTYPE, requires_grad=True)
        energy = total_energy(x, fixture, cfg, q_cells, external)
        (grad,) = torch.autograd.grad(energy, x)
        return float(energy.detach()), grad.detach().numpy().ravel()[FREE]

    result = minimize(
        objective,
        base.ravel()[FREE],
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": cfg.inner_maxiter, "ftol": 1e-16, "gtol": 1e-10},
    )
    full = base.ravel().copy()
    full[FREE] = result.x
    x = full.reshape(X0.shape)
    fs = deformation_gradients(torch.as_tensor(x, dtype=DTYPE), fixture).numpy()
    return x, {
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nit),
        "gradient_inf": float(np.max(np.abs(result.jac))),
        "energy": float(result.fun),
        "detF": np.linalg.det(fs).tolist(),
        "accepted": bool(result.success)
        and float(np.max(np.abs(result.jac))) <= FORWARD_GRADIENT_INF_TOL,
        "acceptance_gradient_inf_tolerance": FORWARD_GRADIENT_INF_TOL,
    }


def shape_rms(x: np.ndarray, target: np.ndarray) -> float:
    # Fixed coordinates define the frame, so do not hide mismatch with a rigid fit.
    return float(np.sqrt(np.mean((x - target) ** 2)))


def target_from_gradients(
    fixture: Fixture, requested: np.ndarray
) -> tuple[np.ndarray, float]:
    def residual(free: np.ndarray) -> np.ndarray:
        full = X0.ravel().copy()
        full[FREE] = free
        f = deformation_gradients(
            torch.as_tensor(full.reshape(X0.shape), dtype=DTYPE), fixture
        ).numpy()
        return (f - requested).ravel()

    fitted = least_squares(
        residual, X0.ravel()[FREE], xtol=1e-14, ftol=1e-14, gtol=1e-14
    )
    x = X0.copy().ravel()
    x[FREE] = fitted.x
    return x.reshape(X0.shape), float(np.sqrt(np.mean(fitted.fun**2)))


def stress(
    F: np.ndarray,
    cfg: Config,
    cell: int,
    q: np.ndarray | None,
    *,
    fa: np.ndarray | None = None,
    muscle_scale: float = 1.0,
) -> np.ndarray:
    def piola(gradient: np.ndarray, young: float) -> np.ndarray:
        mu, lam = stable_parameters(young, cfg.poisson_ratio)
        j = np.linalg.det(gradient)
        return mu * gradient + (-mu + lam * (j - 1)) * (j * np.linalg.inv(gradient).T)

    muscle = (
        piola(F, cfg.muscle_E_mpa)
        if fa is None
        else piola(F @ np.linalg.inv(fa), muscle_scale * cfg.muscle_E_mpa)
        @ np.linalg.inv(fa).T
    )
    passive = FRACTION[cell] * muscle + (1 - FRACTION[cell]) * piola(F, cfg.fat_E_mpa)
    return passive if q is None else passive + FRACTION[cell] * F @ q


def save_mesh(path: Path, x: np.ndarray, state: dict[str, Any]) -> None:
    grid = pv.UnstructuredGrid(
        np.c_[np.full(2, 4), TETS].ravel(), np.full(2, pv.CellType.TETRA), x
    )
    grid.point_data["RestPosition"] = X0
    grid.point_data["Displacement"] = x - X0
    grid.cell_data["MuscleFraction"] = FRACTION
    grid.cell_data["detF"] = np.asarray(state["detF"])
    grid.save(path)


def forward_failure(name: str, solver: dict[str, Any]) -> RuntimeError:
    return RuntimeError(f"forward residual did not converge for {name}: {solver}")


def balanced_q(target_gradients: np.ndarray, cfg: Config, cap: float) -> np.ndarray:
    """Return the PSD/capped local force-balance candidate for supplied F."""
    values = []
    for cell, F in enumerate(target_gradients):
        passive = stress(F, cfg, cell, None)
        candidate = -np.linalg.solve(F, passive) / FRACTION[cell]
        values.append(
            project_psd(pack_symmetric(0.5 * (candidate + candidate.T)), cap)[0]
        )
    return np.asarray(values)


def run_balanced_q_case(
    name: str,
    target: np.ndarray,
    target_gradients: np.ndarray,
    fixture: Fixture,
    cfg: Config,
    cap: float,
) -> dict[str, Any]:
    """Solve one exact local-balance PSD/cap candidate without outer fitting."""
    q = balanced_q(target_gradients, cfg, cap)
    x, solver = solve_equilibrium(fixture, cfg, q)
    if not solver["accepted"]:
        raise forward_failure(name, solver)
    fs = deformation_gradients(torch.as_tensor(x, dtype=DTYPE), fixture).numpy()
    eig = [np.linalg.eigvalsh(item).tolist() for item in q]
    return {
        "id": name,
        "method": "local_force_balance_PSD_capped_candidate",
        "positions": x.tolist(),
        "target_shape_rms": shape_rms(x, target),
        "solver": solver,
        "outer": {
            "status": "analytic_candidate",
            "evaluations": 0,
            "claim_limit": "Per-cell local force balance followed by PSD/cap projection; no global reachability or fit-optimum claim.",
        },
        "detF": np.linalg.det(fs).tolist(),
        "F": fs.tolist(),
        "Q_MPa": q.tolist(),
        "Q_eigenvalues_MPa": eig,
        "Q_cap_MPa": cap,
        "Q_saturated": [bool(np.any(np.isclose(item, cap, atol=1e-8))) for item in eig],
        "piola_frobenius_MPa": [
            float(np.linalg.norm(stress(F, cfg, cell, q[cell])))
            for cell, F in enumerate(fs)
        ],
    }


def run_q_case(
    name: str,
    target: np.ndarray,
    target_gradients: np.ndarray,
    fixture: Fixture,
    cfg: Config,
    cap: float,
    q_ref: float,
) -> dict[str, Any]:
    """Fit in dimensionless Q/Qref coordinates from a force-balanced seed."""
    seed_q = balanced_q(target_gradients, cfg, cap)
    z0 = np.concatenate([pack_symmetric(q) for q in seed_q]) / q_ref
    span = cap / q_ref
    simplex = np.vstack((z0, z0 + np.eye(12) * (0.2 * span)))

    def objective(z: np.ndarray) -> float:
        raw = z * q_ref
        q = np.asarray([project_psd(raw[:6], cap)[0], project_psd(raw[6:], cap)[0]])
        x, solver = solve_equilibrium(fixture, cfg, q)
        return (
            shape_rms(x, target) ** 2
            if solver["accepted"]
            else 1e6 + solver["gradient_inf"]
        )

    outer = minimize(
        objective,
        z0,
        method="Nelder-Mead",
        options={
            "maxiter": cfg.outer_maxiter,
            "xatol": 1e-7,
            "fatol": 1e-12,
            "adaptive": True,
            "initial_simplex": simplex,
        },
    )
    raw = outer.x * q_ref
    q0, eig0, projected0 = project_psd(raw[:6], cap)
    q1, eig1, projected1 = project_psd(raw[6:], cap)
    q = np.asarray((q0, q1))
    x, solver = solve_equilibrium(fixture, cfg, q)
    if not solver["accepted"]:
        raise forward_failure(name, solver)
    fs = deformation_gradients(torch.as_tensor(x, dtype=DTYPE), fixture).numpy()
    return {
        "id": name,
        "method": f"PSD_Q_cap_{cap:.6f}_MPa",
        "positions": x.tolist(),
        "target_shape_rms": shape_rms(x, target),
        "solver": solver,
        "outer": {
            "status": "converged" if outer.success else "fixed_budget_candidate",
            "success": bool(outer.success),
            "message": str(outer.message),
            "evaluations": int(outer.nfev),
            "coordinates": "dimensionless Q/Qref",
            "claim_limit": "This is a bounded-search candidate, not a reachability optimum when status is fixed_budget_candidate.",
        },
        "detF": np.linalg.det(fs).tolist(),
        "F": fs.tolist(),
        "Q_MPa": q.tolist(),
        "Q_eigenvalues_MPa": [eig0.tolist(), eig1.tolist()],
        "Q_projection_applied": [projected0, projected1],
        "Q_cap_MPa": cap,
        "Q_saturated": [
            bool(np.any(np.isclose(eig0, cap, atol=1e-8))),
            bool(np.any(np.isclose(eig1, cap, atol=1e-8))),
        ],
        "balanced_seed_Q_MPa": seed_q.tolist(),
        "balanced_seed_interpretation": "Per-cell local P=0 candidate from supplied target F, projected to PSD/cap; it initializes search and is not an optimum claim.",
        "piola_frobenius_MPa": [
            float(np.linalg.norm(stress(F, cfg, cell, q[cell])))
            for cell, F in enumerate(fs)
        ],
    }


def run_active_strain(
    name: str,
    target: np.ndarray,
    fixture: Fixture,
    cfg: Config,
    fa_cells: np.ndarray,
    *,
    muscle_scale: float = 1.0,
) -> dict[str, Any]:
    """Apply active strain only to each cell's muscle portion; fat sees F."""

    def energy_active(x: torch.Tensor) -> torch.Tensor:
        fs = deformation_gradients(x, fixture)
        energy = torch.zeros((), dtype=DTYPE)
        for cell, F in enumerate(fs):
            fraction = float(FRACTION[cell])
            inv_fa = torch.as_tensor(np.linalg.inv(fa_cells[cell]), dtype=DTYPE)
            energy += fixture.volumes[cell] * (
                fraction
                * passive_energy(
                    F @ inv_fa, muscle_scale * cfg.muscle_E_mpa, cfg.poisson_ratio
                )
                + (1 - fraction) * passive_energy(F, cfg.fat_E_mpa, cfg.poisson_ratio)
            )
        return energy

    base = X0.copy()

    def objective(free: np.ndarray) -> tuple[float, np.ndarray]:
        full = base.ravel().copy()
        full[FREE] = free
        x = torch.tensor(full.reshape(X0.shape), dtype=DTYPE, requires_grad=True)
        value = energy_active(x)
        (grad,) = torch.autograd.grad(value, x)
        return float(value.detach()), grad.detach().numpy().ravel()[FREE]

    result = minimize(
        objective,
        base.ravel()[FREE],
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": cfg.inner_maxiter, "ftol": 1e-16, "gtol": 1e-10},
    )
    full = base.ravel().copy()
    full[FREE] = result.x
    x = full.reshape(X0.shape)
    fs = deformation_gradients(torch.as_tensor(x, dtype=DTYPE), fixture).numpy()
    solver = {
        "success": bool(result.success),
        "message": str(result.message),
        "iterations": int(result.nit),
        "gradient_inf": float(np.max(np.abs(result.jac))),
        "energy": float(result.fun),
        "detF": np.linalg.det(fs).tolist(),
        "accepted": bool(result.success)
        and float(np.max(np.abs(result.jac))) <= FORWARD_GRADIENT_INF_TOL,
        "acceptance_gradient_inf_tolerance": FORWARD_GRADIENT_INF_TOL,
    }
    if not solver["accepted"]:
        raise forward_failure(name, solver)
    return {
        "id": name,
        "method": "active_strain_muscle_only",
        "positions": x.tolist(),
        "target_shape_rms": shape_rms(x, target),
        "solver": solver,
        "detF": np.linalg.det(fs).tolist(),
        "F": fs.tolist(),
        "Fa": fa_cells.tolist(),
        "muscle_E_scale": muscle_scale,
        "Q_MPa": None,
        "piola_frobenius_MPa": [
            float(
                np.linalg.norm(
                    stress(
                        F, cfg, cell, None, fa=fa_cells[cell], muscle_scale=muscle_scale
                    )
                )
            )
            for cell, F in enumerate(fs)
        ],
    }


def write_progress(out: Path, phase: str) -> None:
    """Persist a phase marker before and after every forward solve."""
    (out / "progress.json").write_text(json.dumps({"phase": phase}) + "\n")


def write_compatible_progress(
    out: Path, states: list[dict[str, Any]], q_ref: float
) -> None:
    """Publish accepted compatible cases during the long bounded study."""
    compatible = [state for state in states if state["target"] == "uniform_diagonal_Fa"]
    payload = {
        "schema_version": 1,
        "status": "partial_accepted_cases",
        "scope": "Compatible uniform-Fa cases written immediately after accepted forward equilibria; the full study remains in progress.",
        "Qref_MPa": q_ref,
        "cases": compatible,
    }
    (out / "compatible-progress.json").write_text(json.dumps(payload, indent=2) + "\n")


def plot(
    states: list[dict[str, Any]],
    targets: dict[str, tuple[np.ndarray, np.ndarray]],
    path: Path,
) -> None:
    fig = plt.figure(figsize=(12, 9))
    edges = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    for row, (target_name, (target, _)) in enumerate(targets.items()):
        axis = fig.add_subplot(2, 1, row + 1, projection="3d")
        axis.scatter(*target.T, marker="x", color="black", label="kinematic target")
        for state in (s for s in states if s["target"] == target_name):
            x = np.asarray(state["positions"])
            for tet in TETS:
                for a, b in edges:
                    axis.plot(*x[tet[[a, b]]].T, alpha=0.65)
            axis.scatter(*x.T, s=10, label=state["id"])
        axis.set_title(target_name)
        axis.set_box_aspect((1, 1, 1))
        axis.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main(cfg: Config) -> None:
    # The tiny fixture is slower under the host's large OpenMP pool than serially.
    torch.set_num_threads(1)
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        message = f"choose an empty output directory: {out}"
        raise FileExistsError(message)
    fixture = Fixture.build()
    rigid_rank = rigid_mode_constraint_rank()
    assert rigid_rank == 6, f"fixed DOFs constrain only {rigid_rank} rigid modes"
    source = Path(__file__)
    for path in (source, source.with_name("experiment_profile.py")):
        shutil.copy2(path, out / path.name)
    config = cfg.model_dump(mode="json")
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    write_progress(out, "fixture_and_config_written")
    uniform_fa = np.diag((0.82, 1.04, 1.0))
    compatible_fa = np.asarray((uniform_fa, uniform_fa))
    compatible = (uniform_fa @ X0.T).T
    incompatible_f = np.asarray((np.diag((0.75, 1.0, 1.0)), np.diag((0.90, 1.0, 1.0))))
    incompatible, gradient_rms = target_from_gradients(fixture, incompatible_f)
    requested_lengths = [
        float(np.linalg.norm(F @ (X0[2] - X0[1]))) for F in incompatible_f
    ]
    mismatch_bound = abs(requested_lengths[1] - requested_lengths[0]) / 2
    targets = {
        "uniform_diagonal_Fa": (compatible, compatible_fa),
        "incompatible_shared_edge_length": (incompatible, incompatible_f),
    }
    mu = cfg.muscle_E_mpa / (2 * (1 + cfg.poisson_ratio))
    load = np.zeros_like(X0)
    load[4] = (0.001, -0.0004, 0.0002)
    write_progress(out, "before_passive_external_load")
    passive, passive_solver = solve_equilibrium(fixture, cfg, external=load)
    write_progress(out, "after_passive_external_load")
    write_progress(out, "before_zero_Q_external_load")
    zero_q, zero_q_solver = solve_equilibrium(
        fixture, cfg, np.zeros((2, 3, 3)), external=load
    )
    write_progress(out, "after_zero_Q_external_load")
    assert passive_solver["accepted"], "passive external-load forward solve failed"
    assert zero_q_solver["accepted"], "zero-Q external-load forward solve failed"
    zero_error = float(np.max(np.abs(passive - zero_q)))
    assert zero_error == 0.0, "both-cell Q=0 did not restore the passive solution"
    states: list[dict[str, Any]] = []
    q_ref = 3 * mu
    for target_name, (target, target_gradients) in targets.items():
        active_fa = target_gradients
        entries = [
            ("Q_balance_Qref", "balance_qref"),
            ("Q_balance_10Qref", "balance_10qref"),
            ("active_strain", None),
            ("active_strain_10x_muscle_E", -1.0),
            ("Q_cap_Qref", q_ref),
            ("Q_cap_10Qref", 10 * q_ref),
        ]
        for label, cap in entries:
            cherries.set_step(len(states))
            write_progress(out, f"before_{target_name}_{label}")
            state = (
                run_balanced_q_case(
                    label, target, target_gradients, fixture, cfg, q_ref
                )
                if cap == "balance_qref"
                else run_balanced_q_case(
                    label, target, target_gradients, fixture, cfg, 10 * q_ref
                )
                if cap == "balance_10qref"
                else run_active_strain(
                    label,
                    target,
                    fixture,
                    cfg,
                    active_fa,
                    muscle_scale=10.0 if cap == -1.0 else 1.0,
                )
                if cap is None or cap == -1.0
                else run_q_case(
                    label, target, target_gradients, fixture, cfg, cap, q_ref
                )
            )
            state["target"] = target_name
            stem = f"{target_name}-{label}"
            mesh = out / f"{stem}.vtu"
            save_mesh(mesh, np.asarray(state["positions"]), state)
            state["mesh"] = record(mesh)
            states.append(state)
            write_progress(out, f"after_{target_name}_{label}")
            if target_name == "uniform_diagonal_Fa":
                write_compatible_progress(out, states, q_ref)
            cherries.log_metrics(
                {
                    f"{target_name}/{label}/shape_rms": state["target_shape_rms"],
                    f"{target_name}/{label}/detF_min": min(state["detF"]),
                }
            )
    np.savez_compressed(
        out / "states.npz",
        rest=X0,
        compatible_target=compatible,
        incompatible_target=incompatible,
        incompatible_requested_F=incompatible_f,
        fractions=FRACTION,
    )
    plot(states, targets, out / "comparison.png")
    summary = {
        "schema_version": 2,
        "status": "completed",
        "scope": "CPU-only two-tetrahedron shared-face study; no fibers, skin, or full-face inference.",
        "fixture": {
            "points": X0.tolist(),
            "tets": TETS.tolist(),
            "shared_face": [1, 2, 3],
            "shared_edge": EDGE.tolist(),
            "fixed_dofs": FIXED.tolist(),
            "rigid_mode_constraint_rank": rigid_rank,
            "both_cells_active": True,
            "muscle_fractions": FRACTION.tolist(),
            "passive_remainder": "fat",
        },
        "constitutive": {
            "passive": "fraction*stable(muscle E)+(1-fraction)*stable(fat E) in each cell",
            "active_Q": "one independent symmetric PSD Q per cell; additive 0.5 Q:(F^T F-I)",
            "active_strain": "cellwise supplied Fa applied only to each cell's muscle energy; fat uses F",
            "Fa": compatible_fa.tolist(),
            "muscle_mu_MPa": mu,
            "caps_MPa": {"Qref_equals_3mu": 3 * mu, "10Qref": 30 * mu},
        },
        "zero_Q_check": {
            "external_load": load.tolist(),
            "passive_solver": passive_solver,
            "zero_Q_solver": zero_q_solver,
            "both_cells_Q_zero_max_position_abs_error": zero_error,
        },
        "incompatible_shared_edge": {
            "requested_per_cell_F": incompatible_f.tolist(),
            "requested_shared_edge_lengths": requested_lengths,
            "analytical_minimax_length_mismatch_bound": mismatch_bound,
            "least_squares_gradient_rms": gradient_rms,
            "meaning": "The two desired lengths are for the same physical shared edge. Any one realized length differs from at least one request by this bound; this is a fixture-specific kinematic incompatibility statement.",
        },
        "states": states,
        "cap_interpretation": "Qref=3mu is the small-study lower cap (0.030201 MPa); the frozen face setting is 10Qref=0.302013 MPa, a finite exploratory bound. Selection uses fitting and bounded-Q receipts, not geometry or inversion gates. Fixed-budget Q candidates make no reachability-optimum claim; 10x active strain is a higher-muscle-stiffness authority sensitivity, not a matched stress-budget comparison.",
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    provenance = {
        "sources": {
            "study": record(out / source.name),
            "profile": record(out / "experiment_profile.py"),
        },
        "inputs": {
            "fixture_sha256": hashlib.sha256(
                json.dumps(
                    {
                        "x": X0.tolist(),
                        "tets": TETS.tolist(),
                        "fractions": FRACTION.tolist(),
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest(),
            "states": record(out / "states.npz"),
        },
        "runtime": "NumPy/Torch float64/SciPy CPU",
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    for path in out.iterdir():
        if path.is_file():
            cherries.log_output(path)
    LOG.info("Wrote %s", out)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileCometNoCommit)
