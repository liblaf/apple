import json
from pathlib import Path
from typing import Any, cast

import jax.numpy as jnp
import numpy as np
import pyvista as pv
import scipy.sparse.linalg as spla
import warp as wp
from environs import env
from liblaf.peach.optim import Optimizer
from liblaf.peach.optim.pncg._pncg import _make_preconditioner

from liblaf import cherries, melon
from liblaf.apple.consts import (
    ACTIVATION,
    DIRICHLET_MASK,
    DIRICHLET_VALUE,
    GLOBAL_POINT_ID,
    LAMBDA,
    MU,
    MUSCLE_FRACTION,
    SMAS_FRACTION,
)
from liblaf.apple.model import Forward, Full, Model, ModelBuilder
from liblaf.apple.optim import PNCG
from liblaf.apple.warp import WarpStableNeoHookean, WarpStableNeoHookeanMuscle

SUFFIX: str = "-smas46-muscle46"


class Config(cherries.BaseConfig):
    arch_height: float = env.float("ARCH_HEIGHT", 1.0)
    eigsh_k: int = env.int("EIGSH_K", 2)
    eigsh_maxiter: int = env.int("EIGSH_MAXITER", 500)
    eigsh_tol: float = env.float("EIGSH_TOL", 1e-1)
    input: Path = cherries.input(f"10-input{SUFFIX}.vtu")
    run_forward: bool = env.bool("RUN_FORWARD", True)


def load_mesh(cfg: Config) -> pv.UnstructuredGrid:
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.input)
    return mesh


def compute_arch_profile(
    mesh: pv.UnstructuredGrid,
    arch_height: float,
    points: np.ndarray | None = None,
) -> np.ndarray:
    if points is None:
        points = mesh.points
    bottom_mask: np.ndarray = mesh.points[:, 1] < 1e-3
    bottom_points: np.ndarray = mesh.points[bottom_mask]

    x_min, x_max = bottom_points[:, 0].min(), bottom_points[:, 0].max()
    z_min, z_max = bottom_points[:, 2].min(), bottom_points[:, 2].max()
    x_center = 0.5 * (x_min + x_max)
    z_center = 0.5 * (z_min + z_max)
    x_hat: np.ndarray = 2.0 * (points[:, 0] - x_center) / (x_max - x_min)
    z_hat: np.ndarray = 2.0 * (points[:, 2] - z_center) / (z_max - z_min)
    arch: np.ndarray = arch_height * (1.0 - x_hat**2) * (1.0 - z_hat**2)
    return np.clip(arch, 0.0, None)


def apply_bottom_arch_dirichlet(
    mesh: pv.UnstructuredGrid, arch_height: float
) -> pv.UnstructuredGrid:
    bottom_mask: np.ndarray = mesh.points[:, 1] < 1e-3
    arch: np.ndarray = compute_arch_profile(mesh, arch_height, mesh.points[bottom_mask])

    mesh.point_data[DIRICHLET_MASK][bottom_mask] = True
    mesh.point_data[DIRICHLET_VALUE][bottom_mask] = 0.0
    mesh.point_data[DIRICHLET_VALUE][bottom_mask, 1] = arch
    return mesh


def build_phace_v3(mesh: pv.UnstructuredGrid, arch_height: float) -> Model:
    builder = ModelBuilder()
    mesh = builder.add_points(mesh)
    mesh = apply_bottom_arch_dirichlet(mesh, arch_height)

    muscle_frac: np.ndarray = mesh.cell_data[MUSCLE_FRACTION]
    smas_frac: np.ndarray = mesh.cell_data[SMAS_FRACTION]
    aponeurosis_frac: np.ndarray = smas_frac - muscle_frac
    fat_frac: np.ndarray = 1.0 - smas_frac

    mesh.cell_data[ACTIVATION] = np.zeros((mesh.n_cells, 6))
    mesh.cell_data[ACTIVATION][smas_frac > 1e-3] = np.asarray(
        [2.0 - 1.0, 0.25 - 1.0, 2.0 - 1.0, 0.0, 0.0, 0.0]
    )
    mesh.cell_data[ACTIVATION][muscle_frac > 1e-3] = np.asarray(
        [4.0 - 1.0, 0.125 - 1.0, 2.0 - 1.0, 0.0, 0.0, 0.0]
    )
    builder.add_dirichlet(mesh)

    mesh.cell_data["Fraction"] = fat_frac
    mesh.cell_data[MU] = np.full((mesh.n_cells,), 1.0)
    mesh.cell_data[LAMBDA] = np.full((mesh.n_cells,), 49.0)
    builder.add_energy(WarpStableNeoHookean.from_pyvista(mesh))

    mesh.cell_data["Fraction"] = aponeurosis_frac
    mesh.cell_data[MU] = np.full((mesh.n_cells,), 1.0e2)
    mesh.cell_data[LAMBDA] = np.full((mesh.n_cells,), 49.0e2)
    builder.add_energy(WarpStableNeoHookeanMuscle.from_pyvista(mesh))

    mesh.cell_data["Fraction"] = muscle_frac
    mesh.cell_data[MU] = np.full((mesh.n_cells,), 1.0e2)
    mesh.cell_data[LAMBDA] = np.full((mesh.n_cells,), 49.0e2)
    builder.add_energy(
        WarpStableNeoHookeanMuscle.from_pyvista(
            mesh, requires_grad=("activation",), name="muscle"
        )
    )

    return builder.finalize()


def initialize_parabolic_guess(
    model: Model, mesh: pv.UnstructuredGrid, arch_height: float
) -> None:
    u_full: np.ndarray = np.zeros((model.n_points, model.dim), dtype=mesh.points.dtype)
    u_full[mesh.point_data[GLOBAL_POINT_ID], 1] = compute_arch_profile(
        mesh, arch_height
    )
    model.u_free = model.dirichlet.get_free(jnp.asarray(u_full))


def _eigsh_values(
    operator: spla.LinearOperator,
    *,
    k: int,
    which: str,
    tol: float,
    maxiter: int,
) -> dict[str, Any]:
    converged = True
    try:
        values = spla.eigsh(
            operator,
            k=k,
            which=which,
            tol=tol,
            maxiter=maxiter,
            return_eigenvectors=False,
        )
    except spla.ArpackNoConvergence as exc:
        converged = False
        values = exc.eigenvalues
    values = np.asarray(values, dtype=np.float64)
    values.sort()
    return {"converged": converged, "values": values.tolist()}


def estimate_free_hessian_spectrum(
    model: Model,
    *,
    state: Any,
    k: int,
    tol: float,
    maxiter: int,
) -> dict[str, Any]:
    n_free = int(model.n_free)
    hess_diag_full = np.asarray(model.hess_diag(state), dtype=np.float64)
    hess_diag_free = np.asarray(
        model.dirichlet.get_free(jnp.asarray(hess_diag_full)), dtype=np.float64
    )
    preconditioner = np.asarray(
        _make_preconditioner(jnp.asarray(hess_diag_free)), dtype=np.float64
    )
    preconditioner_sqrt = np.sqrt(preconditioner)

    summary: dict[str, Any] = {
        "n_free": n_free,
        "hess_diag_min": float(np.min(hess_diag_free)),
        "hess_diag_max": float(np.max(hess_diag_free)),
        "hess_diag_mean": float(np.mean(hess_diag_free)),
        "hess_diag_abs_min": float(np.min(np.abs(hess_diag_free))),
        "hess_diag_nonpositive": int(np.count_nonzero(hess_diag_free <= 0.0)),
        "preconditioner_min": float(np.min(preconditioner)),
        "preconditioner_max": float(np.max(preconditioner)),
    }
    if n_free < 3:
        summary["raw_smallest_algebraic"] = {"converged": True, "values": []}
        summary["raw_largest_algebraic"] = {"converged": True, "values": []}
        summary["preconditioned_smallest_algebraic"] = {
            "converged": True,
            "values": [],
        }
        summary["preconditioned_largest_algebraic"] = {
            "converged": True,
            "values": [],
        }
        return summary

    k_eff = max(1, min(k, n_free - 2))

    def hess_matvec(v: np.ndarray) -> np.ndarray:
        v_free = jnp.asarray(v, dtype=model.u_full.dtype)
        v_full: Full = model.dirichlet.to_full(v_free, dirichlet=0.0)
        output_full = model.hess_prod(state, v_full)
        output_free = model.dirichlet.get_free(output_full)
        return np.asarray(output_free, dtype=np.float64)

    def preconditioned_matvec(v: np.ndarray) -> np.ndarray:
        scaled = preconditioner_sqrt * v
        return preconditioner_sqrt * hess_matvec(scaled)

    hess_operator = spla.LinearOperator(
        shape=(n_free, n_free),
        matvec=hess_matvec,
        rmatvec=hess_matvec,
        dtype=np.float64,
    )
    preconditioned_operator = spla.LinearOperator(
        shape=(n_free, n_free),
        matvec=preconditioned_matvec,
        rmatvec=preconditioned_matvec,
        dtype=np.float64,
    )
    summary["raw_smallest_algebraic"] = _eigsh_values(
        hess_operator, k=k_eff, which="SA", tol=tol, maxiter=maxiter
    )
    summary["raw_largest_algebraic"] = _eigsh_values(
        hess_operator, k=k_eff, which="LA", tol=tol, maxiter=maxiter
    )
    summary["preconditioned_smallest_algebraic"] = _eigsh_values(
        preconditioned_operator, k=k_eff, which="SA", tol=tol, maxiter=maxiter
    )
    summary["preconditioned_largest_algebraic"] = _eigsh_values(
        preconditioned_operator, k=k_eff, which="LA", tol=tol, maxiter=maxiter
    )
    return summary


def write_outputs(
    *,
    mesh: pv.UnstructuredGrid,
    forward: Forward,
    solution: Optimizer.Solution | None,
    initial_spectrum: dict[str, Any],
    final_spectrum: dict[str, Any],
) -> None:
    report = {
        "forward": None
        if solution is None
        else {
            "result": str(solution.result),
            "success": bool(solution.success),
            "n_steps": int(np.asarray(solution.state.n_steps)),
            "best_decrease": float(np.asarray(solution.state.best_decrease)),
            "relative_decrease": float(np.asarray(solution.stats.relative_decrease)),
            "grad_norm": float(np.linalg.norm(np.asarray(solution.state.grad))),
        },
        "initial_spectrum": initial_spectrum,
        "final_spectrum": final_spectrum,
    }
    cherries.output(
        "21-forward-bottom-dirichlet-stable-neo-hookean-hessian.json"
    ).write_text(json.dumps(report, indent=2))

    mesh = mesh.copy()
    mesh.point_data["Solution"] = np.asarray(
        forward.u_full[mesh.point_data[GLOBAL_POINT_ID]]
    )
    melon.save(
        cherries.output("21-forward-bottom-dirichlet-stable-neo-hookean-hessian.vtu"),
        mesh,
    )


def main(cfg: Config) -> None:
    wp.init()
    mesh: pv.UnstructuredGrid = load_mesh(cfg)
    model: Model = build_phace_v3(mesh, cfg.arch_height)
    initialize_parabolic_guess(model, mesh, cfg.arch_height)

    forward = Forward(model)
    optimizer = cast("PNCG", forward.optimizer)
    optimizer.rtol = jnp.asarray(1e-3)
    optimizer.rtol_primary = jnp.asarray(1e-5)

    initial_spectrum = estimate_free_hessian_spectrum(
        model,
        state=forward.state,
        k=cfg.eigsh_k,
        tol=cfg.eigsh_tol,
        maxiter=cfg.eigsh_maxiter,
    )

    solution: Optimizer.Solution | None = None
    if cfg.run_forward:
        solution = forward.step()

    final_spectrum = estimate_free_hessian_spectrum(
        model,
        state=forward.state,
        k=cfg.eigsh_k,
        tol=cfg.eigsh_tol,
        maxiter=cfg.eigsh_maxiter,
    )

    ic({"initial_spectrum": initial_spectrum, "final_spectrum": final_spectrum})
    if solution is not None:
        ic(solution)
    write_outputs(
        mesh=mesh,
        forward=forward,
        solution=solution,
        initial_spectrum=initial_spectrum,
        final_spectrum=final_spectrum,
    )


if __name__ == "__main__":
    cherries.main(main, profile="debug")
