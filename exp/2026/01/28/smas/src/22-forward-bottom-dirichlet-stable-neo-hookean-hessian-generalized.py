import json
from pathlib import Path
from typing import Any

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
from liblaf.apple.warp import WarpStableNeoHookean, WarpStableNeoHookeanMuscle

SUFFIX: str = "-smas46-muscle46"


class Config(cherries.BaseConfig):
    arch_height: float = env.float("ARCH_HEIGHT", 1.0)
    eigsh_k: int = env.int("EIGSH_K", 2)
    eigsh_maxiter: int = env.int("EIGSH_MAXITER", 500)
    eigsh_seed: int = env.int("EIGSH_SEED", 0)
    eigsh_tol: float = env.float("EIGSH_TOL", 1e-1)
    input: Path = cherries.input(f"10-input{SUFFIX}.vtu")
    minres_maxiter: int = env.int("MINRES_MAXITER", 40)
    minres_rtol: float = env.float("MINRES_RTOL", 1e-2)
    run_forward: bool = env.bool("RUN_FORWARD", True)
    shift_margin: float = env.float("SHIFT_MARGIN", 1e-3)


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


def _array_to_sorted_list(values: np.ndarray | None) -> list[float]:
    array = np.asarray([] if values is None else values, dtype=np.float64)
    array.sort()
    return array.tolist()


def _diag_operator(diag: np.ndarray) -> spla.LinearOperator:
    diag = np.asarray(diag, dtype=np.float64)
    size = int(diag.size)

    def matvec(v: np.ndarray) -> np.ndarray:
        return diag * np.asarray(v, dtype=np.float64)

    return spla.LinearOperator(
        shape=(size, size),
        matvec=matvec,
        rmatvec=matvec,
        dtype=np.float64,
    )


def _negate_operator(operator: spla.LinearOperator) -> spla.LinearOperator:
    def matvec(v: np.ndarray) -> np.ndarray:
        return -np.asarray(operator.matvec(v), dtype=np.float64)

    return spla.LinearOperator(
        shape=operator.shape,
        matvec=matvec,
        rmatvec=matvec,
        dtype=np.float64,
    )


class _ShiftInvertMinres:
    def __init__(
        self,
        operator: spla.LinearOperator,
        *,
        shift: float,
        mass_operator: spla.LinearOperator | None,
        preconditioner: spla.LinearOperator | None,
        rtol: float,
        maxiter: int,
    ) -> None:
        self.maxiter = maxiter
        self.n_calls = 0
        self.n_failures = 0
        self.operator = operator
        self.preconditioner = preconditioner
        self._info_history: list[int] = []
        self._mass_operator = mass_operator
        self._rtol = rtol
        self._shift = shift
        self.shifted_operator = spla.LinearOperator(
            shape=operator.shape,
            matvec=self._shifted_matvec,
            rmatvec=self._shifted_matvec,
            dtype=np.float64,
        )
        self.operator_inverse = spla.LinearOperator(
            shape=operator.shape,
            matvec=self.solve,
            rmatvec=self.solve,
            dtype=np.float64,
        )

    def _shifted_matvec(self, v: np.ndarray) -> np.ndarray:
        output = np.asarray(self.operator.matvec(v), dtype=np.float64)
        if self._mass_operator is None:
            return output - self._shift * np.asarray(v, dtype=np.float64)
        return output - self._shift * np.asarray(
            self._mass_operator.matvec(v), dtype=np.float64
        )

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        rhs = np.asarray(rhs, dtype=np.float64)
        solution, info = spla.minres(
            self.shifted_operator,
            rhs,
            rtol=self._rtol,
            maxiter=self.maxiter,
            M=self.preconditioner,
        )
        self.n_calls += 1
        self._info_history.append(int(info))
        if info != 0:
            self.n_failures += 1
        return np.asarray(solution, dtype=np.float64)

    def summary(self) -> dict[str, Any]:
        return {
            "info_history": self._info_history,
            "maxiter": self.maxiter,
            "n_calls": self.n_calls,
            "n_failures": self.n_failures,
            "rtol": self._rtol,
            "shift": self._shift,
        }


def _eigsh_values(
    operator: spla.LinearOperator,
    *,
    k: int,
    maxiter: int,
    mass_operator: spla.LinearOperator | None = None,
    mass_inverse_operator: spla.LinearOperator | None = None,
    opinv: spla.LinearOperator | None = None,
    rng_seed: int,
    sigma: float | None = None,
    tol: float,
    which: str,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "k": k,
        "maxiter": maxiter,
        "return_eigenvectors": False,
        "rng": np.random.default_rng(rng_seed),
        "tol": tol,
        "which": which,
    }
    if mass_operator is not None:
        kwargs["M"] = mass_operator
    if sigma is None and mass_inverse_operator is not None:
        kwargs["Minv"] = mass_inverse_operator
    if sigma is not None:
        kwargs["sigma"] = sigma
    if opinv is not None:
        kwargs["OPinv"] = opinv

    converged = True
    try:
        values = spla.eigsh(operator, **kwargs)
    except spla.ArpackNoConvergence as exc:
        converged = False
        values = exc.eigenvalues

    return {
        "converged": converged,
        "values": _array_to_sorted_list(values),
    }


def _coarse_smallest_values(
    operator: spla.LinearOperator,
    *,
    k: int,
    maxiter: int,
    mass_operator: spla.LinearOperator | None,
    mass_inverse_operator: spla.LinearOperator | None,
    rng_seed: int,
    tol: float,
) -> dict[str, Any]:
    negated = _eigsh_values(
        _negate_operator(operator),
        k=k,
        mass_operator=mass_operator,
        mass_inverse_operator=mass_inverse_operator,
        maxiter=maxiter,
        rng_seed=rng_seed,
        tol=tol,
        which="LA",
    )
    values = -np.asarray(negated["values"], dtype=np.float64)
    values.sort()
    return {
        "converged": bool(negated["converged"]),
        "values": values.tolist(),
    }


def _estimate_smallest_algebraic(
    operator: spla.LinearOperator,
    *,
    base_diag: np.ndarray,
    k: int,
    mass_diag: np.ndarray,
    mass_operator: spla.LinearOperator | None,
    mass_inverse_operator: spla.LinearOperator | None,
    maxiter: int,
    minres_maxiter: int,
    minres_rtol: float,
    rng_seed: int,
    shift_margin: float,
    tol: float,
) -> dict[str, Any]:
    coarse = _coarse_smallest_values(
        operator,
        k=k,
        mass_operator=mass_operator,
        mass_inverse_operator=mass_inverse_operator,
        maxiter=maxiter,
        rng_seed=rng_seed,
        tol=tol,
    )
    coarse_values = np.asarray(coarse["values"], dtype=np.float64)
    quotient_diag = np.divide(
        base_diag,
        mass_diag,
        out=np.zeros_like(base_diag, dtype=np.float64),
        where=np.abs(mass_diag) > 0.0,
    )
    quotient_abs_mean = float(np.mean(np.abs(quotient_diag)))

    sigma_candidates: list[float] = []
    if coarse_values.size > 0:
        sigma_candidates.append(
            float(
                coarse_values[0]
                - max(
                    shift_margin * max(1.0, abs(coarse_values[0])),
                    np.finfo(np.float64).eps,
                )
            )
        )
    fallback_scale = max(1.0, quotient_abs_mean)
    sigma_candidates.extend(
        [
            0.0,
            -shift_margin * fallback_scale,
            -fallback_scale,
        ]
    )
    sigma_candidates = list(dict.fromkeys(float(sigma) for sigma in sigma_candidates))

    shift_attempts: list[dict[str, Any]] = []
    selected_shifted_values = np.asarray([], dtype=np.float64)
    selected_shifted_converged = False
    for attempt_index, sigma in enumerate(sigma_candidates):
        shifted_diag = base_diag - sigma * mass_diag
        shifted_preconditioner = np.asarray(
            _make_preconditioner(jnp.asarray(shifted_diag)),
            dtype=np.float64,
        )
        solver = _ShiftInvertMinres(
            operator,
            shift=sigma,
            mass_operator=mass_operator,
            preconditioner=_diag_operator(shifted_preconditioner),
            rtol=minres_rtol,
            maxiter=minres_maxiter,
        )
        shifted = _eigsh_values(
            operator,
            k=k,
            mass_operator=mass_operator,
            maxiter=maxiter,
            opinv=solver.operator_inverse,
            rng_seed=rng_seed + 1 + attempt_index,
            sigma=sigma,
            tol=tol,
            which="LM",
        )
        shifted_values = np.asarray(shifted["values"], dtype=np.float64)
        shifted_values.sort()
        shift_attempts.append(
            {
                "converged": bool(shifted["converged"]),
                "minres": solver.summary(),
                "sigma": sigma,
                "values": shifted_values.tolist(),
            }
        )
        if shifted_values.size == 0:
            continue
        if selected_shifted_values.size == 0:
            selected_shifted_values = shifted_values
            selected_shifted_converged = bool(shifted["converged"])
            continue
        combined = np.concatenate([selected_shifted_values, shifted_values])
        combined.sort()
        selected_shifted_values = combined[:k]
        selected_shifted_converged = selected_shifted_converged or bool(
            shifted["converged"]
        )

    if selected_shifted_values.size > 0:
        selected_values = selected_shifted_values.tolist()
        method = "shift_invert_minres"
        converged = selected_shifted_converged
    else:
        selected_values = coarse_values.tolist()
        method = "coarse_only"
        converged = bool(coarse["converged"] and coarse_values.size > 0)

    return {
        "coarse": coarse,
        "converged": converged,
        "method": method,
        "shift_invert_attempts": shift_attempts,
        "values": selected_values,
    }


def estimate_free_hessian_spectrum(
    model: Model,
    *,
    cfg: Config,
    state: Any,
) -> dict[str, Any]:
    n_free = int(model.n_free)
    hess_diag_full = np.asarray(model.hess_diag(state), dtype=np.float64)
    hess_diag_free = np.asarray(
        model.dirichlet.get_free(jnp.asarray(hess_diag_full)),
        dtype=np.float64,
    )
    preconditioner = np.asarray(
        _make_preconditioner(jnp.asarray(hess_diag_free)),
        dtype=np.float64,
    )
    preconditioner_inverse = np.reciprocal(preconditioner)

    summary: dict[str, Any] = {
        "n_free": n_free,
        "hess_diag_abs_min": float(np.min(np.abs(hess_diag_free))),
        "hess_diag_max": float(np.max(hess_diag_free)),
        "hess_diag_mean": float(np.mean(hess_diag_free)),
        "hess_diag_min": float(np.min(hess_diag_free)),
        "hess_diag_nonpositive": int(np.count_nonzero(hess_diag_free <= 0.0)),
        "preconditioner_inverse_max": float(np.max(preconditioner_inverse)),
        "preconditioner_inverse_min": float(np.min(preconditioner_inverse)),
        "preconditioner_max": float(np.max(preconditioner)),
        "preconditioner_min": float(np.min(preconditioner)),
    }
    if n_free < 3:
        empty = {"converged": True, "values": []}
        summary["lambda_max"] = empty
        summary["lambda_min"] = empty
        summary["preconditioned_lambda_max"] = empty
        summary["preconditioned_lambda_min"] = empty
        return summary

    k_eff = max(1, min(cfg.eigsh_k, n_free - 2))

    def hess_matvec(v: np.ndarray) -> np.ndarray:
        v_free = jnp.asarray(v, dtype=model.u_full.dtype)
        v_full: Full = model.dirichlet.to_full(v_free, dirichlet=0.0)
        output_full = model.hess_prod(state, v_full)
        output_free = model.dirichlet.get_free(output_full)
        return np.asarray(output_free, dtype=np.float64)

    hess_operator = spla.LinearOperator(
        shape=(n_free, n_free),
        matvec=hess_matvec,
        rmatvec=hess_matvec,
        dtype=np.float64,
    )
    mass_operator = _diag_operator(preconditioner_inverse)
    mass_inverse_operator = _diag_operator(preconditioner)
    identity_diag = np.ones((n_free,), dtype=np.float64)

    summary["lambda_max"] = _eigsh_values(
        hess_operator,
        k=k_eff,
        maxiter=cfg.eigsh_maxiter,
        rng_seed=cfg.eigsh_seed + 10,
        tol=cfg.eigsh_tol,
        which="LA",
    )
    summary["lambda_min"] = _estimate_smallest_algebraic(
        hess_operator,
        base_diag=hess_diag_free,
        k=k_eff,
        mass_diag=identity_diag,
        mass_operator=None,
        mass_inverse_operator=None,
        maxiter=cfg.eigsh_maxiter,
        minres_maxiter=cfg.minres_maxiter,
        minres_rtol=cfg.minres_rtol,
        rng_seed=cfg.eigsh_seed + 20,
        shift_margin=cfg.shift_margin,
        tol=cfg.eigsh_tol,
    )
    # The generalized problem H x = lambda P^{-1} x matches the spectrum of
    # P^{1/2} H P^{1/2} for the diagonal P used by PNCG.
    summary["preconditioned_lambda_max"] = _eigsh_values(
        hess_operator,
        k=k_eff,
        mass_operator=mass_operator,
        mass_inverse_operator=mass_inverse_operator,
        maxiter=cfg.eigsh_maxiter,
        rng_seed=cfg.eigsh_seed + 30,
        tol=cfg.eigsh_tol,
        which="LA",
    )
    summary["preconditioned_lambda_min"] = _estimate_smallest_algebraic(
        hess_operator,
        base_diag=hess_diag_free,
        k=k_eff,
        mass_diag=preconditioner_inverse,
        mass_operator=mass_operator,
        mass_inverse_operator=mass_inverse_operator,
        maxiter=cfg.eigsh_maxiter,
        minres_maxiter=cfg.minres_maxiter,
        minres_rtol=cfg.minres_rtol,
        rng_seed=cfg.eigsh_seed + 40,
        shift_margin=cfg.shift_margin,
        tol=cfg.eigsh_tol,
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
        "22-forward-bottom-dirichlet-stable-neo-hookean-hessian-generalized.json"
    ).write_text(json.dumps(report, indent=2))

    mesh = mesh.copy()
    mesh.point_data["Solution"] = np.asarray(
        forward.u_full[mesh.point_data[GLOBAL_POINT_ID]]
    )
    melon.save(
        cherries.output(
            "22-forward-bottom-dirichlet-stable-neo-hookean-hessian-generalized.vtu"
        ),
        mesh,
    )


def _compact_spectrum_summary(spectrum: dict[str, Any]) -> dict[str, Any]:
    return {
        "lambda_max": spectrum["lambda_max"]["values"],
        "lambda_min": spectrum["lambda_min"]["values"],
        "preconditioned_lambda_max": spectrum["preconditioned_lambda_max"]["values"],
        "preconditioned_lambda_min": spectrum["preconditioned_lambda_min"]["values"],
    }


def main(cfg: Config) -> None:
    wp.init()
    mesh: pv.UnstructuredGrid = load_mesh(cfg)
    model: Model = build_phace_v3(mesh, cfg.arch_height)
    initialize_parabolic_guess(model, mesh, cfg.arch_height)

    forward = Forward(model)

    initial_spectrum = estimate_free_hessian_spectrum(
        model, cfg=cfg, state=forward.state
    )

    solution: Optimizer.Solution | None = None
    if cfg.run_forward:
        solution = forward.step()

    final_spectrum = estimate_free_hessian_spectrum(model, cfg=cfg, state=forward.state)

    ic(
        {
            "initial_spectrum": _compact_spectrum_summary(initial_spectrum),
            "final_spectrum": _compact_spectrum_summary(final_spectrum),
        }
    )
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
