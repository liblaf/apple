import contextlib
import io
import json
import logging
import math
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
import pyvista as pv
import torch
import warp as wp

from liblaf import cherries, melon

PREP_STEM = "10-inverse-face"
OUTPUT_STEM = "20-inverse-face"
TARGET_SURFACE_MASK = "TargetSurfaceMask"
BACKGROUND_FRACTION = "BackgroundFraction"
ACTIVE_FRACTION = "ActiveFraction"
SMAS_STIFFNESS_FRACTION = "SmasStiffnessFraction"


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    input: Path = cherries.input(f"{PREP_STEM}-input.vtu")
    target: Path = cherries.input(f"{PREP_STEM}-target.vtu")
    output_input: Path = cherries.output(f"{OUTPUT_STEM}-input.vtu")
    output_target: Path = cherries.output(f"{OUTPUT_STEM}-target.vtu")
    output: Path = cherries.output(f"{OUTPUT_STEM}.vtu")
    output_series: Path = cherries.output(f"{OUTPUT_STEM}.vtu.series")
    output_summary: Path = cherries.output(f"{OUTPUT_STEM}-summary.json")
    checkpoint: Path = cherries.output(f"{OUTPUT_STEM}-checkpoint.npz")
    initial_activation_inv: Path | None = None

    E: float = 1.0
    nu: float = 0.49
    smas_stiffness_ratio: float = 1.0e2
    active_fraction_tol: float = 1.0e-3
    target_point_mask: str = "IsFace"

    forward_rtol: float = 5.0e-4
    forward_atol: float = 0.0
    forward_max_steps: int = 10000

    inverse_lr: float = 0.03
    adam_beta1: float = 0.3
    adam_beta2: float = 0.9
    adam_eps: float = 1.0e-8
    inverse_max_steps: int = 1000
    inverse_min_steps: int = 80
    stagnation_patience: int = 250
    stagnation_rel_tol: float = 1.0e-5
    stagnation_abs_tol: float = 1.0e-8
    loss_tol: float = 1.0e-7
    max_point_error_cm: float = 0.2
    failure_patience: int = 3
    adjoint_maxiter: int = 10000
    adjoint_rtol: float = 5.0e-4
    adjoint_atol: float = 0.0

    activation_inv_diag_min: float = -8.0
    activation_inv_diag_max: float = 8.0
    activation_inv_shear_abs_max: float = 3.0


def configure_runtime() -> None:
    if not torch.cuda.is_available():
        msg = "This experiment uses Warp kernels through Torch and needs CUDA."
        raise RuntimeError(msg)
    logging.getLogger("liblaf.apple.forward._forward").setLevel(logging.WARNING)
    logging.getLogger("liblaf.apple.inverse._diff_forward").setLevel(logging.WARNING)
    warnings.filterwarnings(
        "ignore",
        message=r"The \.grad attribute of a Tensor that is not a leaf Tensor.*",
        category=UserWarning,
    )
    torch.set_default_dtype(torch.float64)
    torch.set_default_device("cuda")
    wp.config.mode = "release"
    wp.init()


def forward_quiet(differentiable_forward: Any, materials: Any) -> torch.Tensor:
    with contextlib.redirect_stdout(io.StringIO()):
        return differentiable_forward.forward(materials)


def to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def to_float(value: Any, default: float = math.nan) -> float:
    if value is None:
        return default
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


def relative_value(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return math.nan
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else math.inf
    return numerator / denominator


def require_path(path: Path) -> None:
    if path.exists():
        return
    msg = f"missing input: {path}. Run {PREP_STEM}.py first."
    raise FileNotFoundError(msg)


def load_problem(cfg: Config) -> tuple[pv.UnstructuredGrid, pv.UnstructuredGrid]:
    require_path(cfg.input)
    require_path(cfg.target)
    mesh = pv.read(cfg.input)
    target = pv.read(cfg.target)
    if not isinstance(mesh, pv.UnstructuredGrid):
        mesh = mesh.cast_to_unstructured_grid()
    if not isinstance(target, pv.UnstructuredGrid):
        target = target.cast_to_unstructured_grid()
    if mesh.n_points != target.n_points or mesh.n_cells != target.n_cells:
        msg = (
            "input and target topology sizes differ: "
            f"points {mesh.n_points} != {target.n_points}, "
            f"cells {mesh.n_cells} != {target.n_cells}"
        )
        raise ValueError(msg)
    if not np.allclose(mesh.points, target.points):
        msg = "input and target rest points differ"
        raise ValueError(msg)
    if "Displacement" not in target.point_data:
        msg = f"{cfg.target} has no point_data['Displacement']"
        raise KeyError(msg)
    return mesh, target


def lame_parameters(E: float, nu: float) -> tuple[float, float]:
    lambda_ = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    return float(lambda_), float(mu)


def set_material(
    mesh: pv.UnstructuredGrid,
    *,
    E: float,
    nu: float,
    fraction: np.ndarray,
) -> None:
    from liblaf.apple.common import FRACTION, LAMBDA, MU, NU
    from liblaf.apple.common import E as YOUNG_MODULUS

    lambda_, mu = lame_parameters(E, nu)
    mesh.cell_data[YOUNG_MODULUS.vtk] = np.full(mesh.n_cells, E, dtype=np.float64)
    mesh.cell_data[NU.vtk] = np.full(mesh.n_cells, nu, dtype=np.float64)
    mesh.cell_data[LAMBDA.vtk] = np.full(mesh.n_cells, lambda_, dtype=np.float64)
    mesh.cell_data[MU.vtk] = np.full(mesh.n_cells, mu, dtype=np.float64)
    mesh.cell_data[FRACTION.vtk] = np.asarray(fraction, dtype=np.float64)


def active_cell_ids(mesh: pv.UnstructuredGrid, cfg: Config) -> np.ndarray:
    if "ActivationMask" in mesh.cell_data:
        active = np.asarray(mesh.cell_data["ActivationMask"], dtype=bool)
    else:
        active = (
            np.asarray(mesh.cell_data["MuscleFraction"], dtype=np.float64)
            > cfg.active_fraction_tol
        )
    ids = np.flatnonzero(active).astype(np.int64)
    if ids.size == 0:
        msg = "no active muscle tetrahedra selected"
        raise ValueError(msg)
    return ids


def target_point_ids(target: pv.UnstructuredGrid, cfg: Config) -> np.ndarray:
    if cfg.target_point_mask in target.point_data:
        mask = np.asarray(target.point_data[cfg.target_point_mask], dtype=bool)
    elif TARGET_SURFACE_MASK in target.point_data:
        mask = np.asarray(target.point_data[TARGET_SURFACE_MASK], dtype=bool)
    else:
        msg = (
            f"target has neither point_data[{cfg.target_point_mask!r}] nor "
            f"point_data[{TARGET_SURFACE_MASK!r}]"
        )
        raise KeyError(msg)
    ids = np.flatnonzero(mask)
    if ids.size == 0:
        msg = "target point mask selected no points"
        raise ValueError(msg)
    return ids.astype(np.int64)


def clamp_activation_inv_(activation_inv: torch.Tensor, cfg: Config) -> None:
    activation_inv[:, :3].clamp_(
        cfg.activation_inv_diag_min, cfg.activation_inv_diag_max
    )
    activation_inv[:, 3:].clamp_(
        -cfg.activation_inv_shear_abs_max, cfg.activation_inv_shear_abs_max
    )


def full_activation_inv_from_active(
    active_activation_inv: torch.Tensor,
    active_ids_t: torch.Tensor,
    n_cells: int,
) -> torch.Tensor:
    full_activation_inv = torch.zeros(
        (n_cells, 6),
        dtype=active_activation_inv.dtype,
        device=active_activation_inv.device,
    )
    return full_activation_inv.index_copy(0, active_ids_t, active_activation_inv)


def load_initial_active_activation_inv(
    path: Path | None, active_ids: np.ndarray
) -> np.ndarray | None:
    if path is None:
        return None
    require_path(path)
    if path.suffix == ".npz":
        with np.load(path) as data:
            if "active_activation_inv" in data:
                activation_inv = np.asarray(
                    data["active_activation_inv"], dtype=np.float64
                )
            else:
                activation_inv = np.asarray(data["activation_inv"], dtype=np.float64)[
                    active_ids
                ]
    else:
        from liblaf.apple.common import ACTIVATION_INV

        mesh = pv.read(path)
        if "RecoveredActivationInv" in mesh.cell_data:
            activation_inv = np.asarray(
                mesh.cell_data["RecoveredActivationInv"], dtype=np.float64
            )[active_ids]
        else:
            activation_inv = np.asarray(
                mesh.cell_data[ACTIVATION_INV.vtk], dtype=np.float64
            )[active_ids]
    expected_shape = (active_ids.size, 6)
    if activation_inv.shape != expected_shape:
        msg = (
            f"initial activation has shape {activation_inv.shape}; "
            f"expected {expected_shape}"
        )
        raise ValueError(msg)
    return activation_inv


def save_checkpoint(
    path: Path,
    active_activation_inv: torch.Tensor,
    *,
    step: int,
    loss: float,
    max_error: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    with tmp.open("wb") as file:
        np.savez(
            file,
            active_activation_inv=to_numpy(active_activation_inv),
            step=np.asarray(step, dtype=np.int64),
            loss=np.asarray(loss, dtype=np.float64),
            max_error=np.asarray(max_error, dtype=np.float64),
        )
    tmp.replace(path)


def build_forward(mesh: pv.UnstructuredGrid, cfg: Config):
    from liblaf.apple.forward import Forward, ModelBuilder
    from liblaf.apple.warp.fem import StableNeoHookean, StableNeoHookeanActive

    builder = ModelBuilder()
    builder.add_vertices(mesh)
    builder.add_fixed(mesh)

    set_material(mesh, E=cfg.E, nu=cfg.nu, fraction=mesh.cell_data[BACKGROUND_FRACTION])
    builder.add_potential(StableNeoHookean.from_pyvista(mesh, name="background"))

    set_material(mesh, E=cfg.E, nu=cfg.nu, fraction=mesh.cell_data[ACTIVE_FRACTION])
    builder.add_potential(StableNeoHookeanActive.from_pyvista(mesh, name="muscle"))

    set_material(
        mesh,
        E=cfg.smas_stiffness_ratio * cfg.E,
        nu=cfg.nu,
        fraction=mesh.cell_data[SMAS_STIFFNESS_FRACTION],
    )
    builder.add_potential(StableNeoHookean.from_pyvista(mesh, name="smas"))

    forward = Forward(builder.finalize())
    forward.optimizer = forward.default_optimizer(
        max_steps=cfg.forward_max_steps,
        atol=cfg.forward_atol,
        rtol=cfg.forward_rtol,
    )
    return forward


def material_tree(
    base_materials: dict[str, dict[str, torch.Tensor]],
    active_activation_inv: torch.Tensor,
    active_ids_t: torch.Tensor,
    n_cells: int,
) -> dict[str, dict[str, torch.Tensor]]:
    materials = {name: dict(values) for name, values in base_materials.items()}
    materials["muscle"]["activation_inv"] = full_activation_inv_from_active(
        active_activation_inv, active_ids_t, n_cells
    )
    return materials


def forward_solution_metrics(solution: Any) -> dict[str, Any]:
    if solution is None:
        return {
            "forward/result": "missing",
            "forward/success": False,
            "forward/steps": math.nan,
            "forward/grad_norm": math.nan,
            "forward/relative_grad_norm": math.nan,
            "forward/grad_norm_first": math.nan,
            "forward/line_search_ok": False,
            "forward/line_search_steps": math.nan,
            "forward/stagnation_count": math.nan,
        }
    convergence_state = solution.state.convergence_state
    line_search_state = solution.state.line_search_state
    grad_norm = to_float(convergence_state.grad_norm)
    grad_norm_first = to_float(convergence_state.grad_norm_first)
    return {
        "forward/result": str(solution.result),
        "forward/success": bool(solution.success),
        "forward/steps": int(convergence_state.step),
        "forward/grad_norm": grad_norm,
        "forward/relative_grad_norm": relative_value(grad_norm, grad_norm_first),
        "forward/grad_norm_first": grad_norm_first,
        "forward/line_search_ok": bool(line_search_state.ok),
        "forward/line_search_steps": int(line_search_state.step),
        "forward/stagnation_count": int(convergence_state.stagnation_count),
    }


def adjoint_solution_metrics(solution: Any) -> dict[str, Any]:
    if solution is None:
        return {
            "adjoint/result": "missing",
            "adjoint/success": False,
            "adjoint/solver_count": 0,
            "adjoint/best_solver": -1,
            "adjoint/absolute_residual": math.nan,
            "adjoint/relative_residual": math.nan,
        }
    state = solution.state
    best_index = int(state.best_index.detach().cpu())
    absolute_residuals = to_numpy(state.absolute_residuals)
    relative_residuals = to_numpy(state.relative_residuals)
    metrics: dict[str, Any] = {
        "adjoint/result": str(solution.result),
        "adjoint/success": bool(solution.success),
        "adjoint/solver_count": len(state.solutions),
        "adjoint/best_solver": best_index,
        "adjoint/absolute_residual": float(absolute_residuals[best_index]),
        "adjoint/relative_residual": float(relative_residuals[best_index]),
    }
    for i, solver_solution in enumerate(state.solutions):
        prefix = f"adjoint/solver_{i}"
        metrics[f"{prefix}/result"] = str(solver_solution.result)
        metrics[f"{prefix}/success"] = bool(solver_solution.success)
        metrics[f"{prefix}/steps"] = (
            -1
            if solver_solution.state.step is None
            else int(solver_solution.state.step)
        )
        metrics[f"{prefix}/info"] = int(solver_solution.state.info)
        metrics[f"{prefix}/absolute_residual"] = float(absolute_residuals[i])
        metrics[f"{prefix}/relative_residual"] = float(relative_residuals[i])
    return metrics


def point_error_stats(residual: torch.Tensor) -> dict[str, torch.Tensor]:
    point_error = torch.linalg.vector_norm(residual, dim=1)
    return {
        "mean": point_error.mean(),
        "rms": torch.linalg.vector_norm(residual) / math.sqrt(residual.shape[0]),
        "max": point_error.max(),
    }


def is_significant_decrease(
    value: float, best: float, *, rel_tol: float, abs_tol: float
) -> bool:
    if math.isinf(best):
        return True
    threshold = min(best * (1.0 - rel_tol), best - abs_tol)
    return value < threshold


def inverse_tensors(
    mesh: pv.UnstructuredGrid,
    target_displacement: np.ndarray,
    point_ids: np.ndarray,
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
    from liblaf.apple.common import GLOBAL_POINT_ID

    global_ids = np.asarray(mesh.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64)
    point_global_ids = global_ids[point_ids]
    target = torch.as_tensor(
        target_displacement,
        dtype=torch.get_default_dtype(),
        device=torch.get_default_device(),
    )
    point_ids_t = torch.as_tensor(
        point_ids,
        dtype=torch.long,
        device=torch.get_default_device(),
    )
    point_global_ids_t = torch.as_tensor(
        point_global_ids,
        dtype=torch.long,
        device=torch.get_default_device(),
    )
    return global_ids, target, point_ids_t, point_global_ids_t


def add_metric_fields(
    mesh: pv.UnstructuredGrid, metrics: dict[str, float | int | bool | str]
) -> None:
    for name, value in metrics.items():
        if isinstance(value, str):
            continue
        mesh.field_data[name] = np.asarray([value])


def add_masks(
    mesh: pv.UnstructuredGrid, target_ids: np.ndarray, active_ids: np.ndarray
) -> None:
    target_mask = np.zeros(mesh.n_points, dtype=np.int8)
    target_mask[target_ids] = 1
    active_mask = np.zeros(mesh.n_cells, dtype=np.int8)
    active_mask[active_ids] = 1
    mesh.point_data[TARGET_SURFACE_MASK] = target_mask
    mesh.cell_data["InverseActiveMask"] = active_mask


def make_target_mesh(
    target: pv.UnstructuredGrid, target_ids: np.ndarray, active_ids: np.ndarray
) -> pv.UnstructuredGrid:
    result = target.copy(deep=True)
    add_masks(result, target_ids, active_ids)
    displacement = np.asarray(result.point_data["Displacement"], dtype=np.float64)
    result.point_data["TargetDisplacement"] = displacement
    result.point_data["TargetPoint"] = result.points + displacement
    return result


def make_result_mesh(
    mesh: pv.UnstructuredGrid,
    target_displacement: np.ndarray,
    displacement: np.ndarray,
    recovered_activation_inv: np.ndarray,
    target_ids: np.ndarray,
    active_ids: np.ndarray,
    metrics: dict[str, float | int | bool | str],
) -> pv.UnstructuredGrid:
    from liblaf.apple.common import ACTIVATION_INV

    result = mesh.copy(deep=True)
    add_masks(result, target_ids, active_ids)
    error = displacement - target_displacement
    result.point_data["Displacement"] = displacement
    result.point_data["TargetDisplacement"] = target_displacement
    result.point_data["DisplacementError"] = error
    result.point_data["DisplacementErrorNorm"] = np.linalg.norm(error, axis=1)
    result.point_data["DeformedPoint"] = result.points + displacement
    result.point_data["TargetPoint"] = result.points + target_displacement
    result.cell_data[ACTIVATION_INV.vtk] = recovered_activation_inv
    result.cell_data["RecoveredActivationInv"] = recovered_activation_inv
    result.cell_data["RecoveredActivationInvNorm"] = np.linalg.norm(
        recovered_activation_inv, axis=1
    )
    add_metric_fields(result, metrics)
    return result


def solve_inverse(  # noqa: C901, PLR0912, PLR0915
    mesh: pv.UnstructuredGrid,
    target_displacement: np.ndarray,
    target_ids: np.ndarray,
    active_ids: np.ndarray,
    cfg: Config,
    series_writer: Any,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[dict[str, Any]],
    str,
    int,
    int,
    dict[str, float],
]:
    from liblaf.peach.linalg import FallbackSolver
    from liblaf.peach.linalg.cupy import CupyCG, CupyMinRes

    from liblaf.apple.inverse import DifferentiableForward

    class RecordingDifferentiableForward(DifferentiableForward):
        __slots__ = ("last_adjoint_solution", "last_forward_solution")

        def step(self) -> Any:
            solution = super().step()
            self.last_forward_solution = solution
            return solution

        def adjoint_solve(self, u_grad: torch.Tensor) -> Any:
            solution = super().adjoint_solve(u_grad)
            self.last_adjoint_solution = solution
            return solution

    forward = build_forward(mesh, cfg)
    differentiable_forward = RecordingDifferentiableForward(forward)
    differentiable_forward.adjoint_solver = FallbackSolver(
        solvers=[
            CupyCG(
                maxiter=cfg.adjoint_maxiter,
                rtol=cfg.adjoint_rtol,
                atol=cfg.adjoint_atol,
            ),
            CupyMinRes(maxiter=cfg.adjoint_maxiter, tol=cfg.adjoint_rtol),
        ]
    )
    base_materials = forward.model.get_materials()
    global_ids, target, point_ids_t, point_global_ids_t = inverse_tensors(
        mesh, target_displacement, target_ids
    )
    active_ids_t = torch.as_tensor(
        active_ids,
        dtype=torch.long,
        device=torch.get_default_device(),
    )
    active_activation_inv = torch.nn.Parameter(
        torch.zeros(
            (active_ids.size, 6),
            dtype=torch.get_default_dtype(),
            device=torch.get_default_device(),
        )
    )
    initial_activation_inv = load_initial_active_activation_inv(
        cfg.initial_activation_inv, active_ids
    )
    if initial_activation_inv is not None:
        with torch.no_grad():
            active_activation_inv.copy_(
                torch.as_tensor(
                    initial_activation_inv,
                    dtype=active_activation_inv.dtype,
                    device=active_activation_inv.device,
                )
            )
            clamp_activation_inv_(active_activation_inv, cfg)
    optimizer = torch.optim.Adam(
        [active_activation_inv],
        lr=cfg.inverse_lr,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_eps,
    )

    trace: list[dict[str, Any]] = []
    stop_reason = "step_safety_limit"
    optimizer_steps = 0
    best_step = 0
    best_displacement: np.ndarray | None = None
    best_activation_inv: np.ndarray | None = None
    best_active_activation_inv: torch.Tensor | None = None
    best_loss = math.inf
    best_max_error = math.inf
    lowest_loss = math.inf
    lowest_max_error = math.inf
    lowest_loss_step = 0
    lowest_max_error_step = 0
    no_improve_steps = 0
    timing = {
        "time/inverse_s": 0.0,
        "time/forward_s": 0.0,
        "time/backward_s": 0.0,
        "time/optimizer_s": 0.0,
        "time/series_s": 0.0,
    }
    inverse_start = time.perf_counter()
    for step in range(cfg.inverse_max_steps + 1):
        step_start = time.perf_counter()
        optimizer.zero_grad()
        materials = material_tree(
            base_materials, active_activation_inv, active_ids_t, mesh.n_cells
        )

        forward_start = time.perf_counter()
        output = forward_quiet(differentiable_forward, materials)
        forward_elapsed = time.perf_counter() - forward_start
        timing["time/forward_s"] += forward_elapsed
        forward_metrics = forward_solution_metrics(
            getattr(differentiable_forward, "last_forward_solution", None)
        )

        backward_start = time.perf_counter()
        residual = output[point_global_ids_t] - target[point_ids_t]
        loss = residual.square().mean()
        loss.backward()
        backward_elapsed = time.perf_counter() - backward_start
        timing["time/backward_s"] += backward_elapsed
        adjoint_metrics = adjoint_solution_metrics(
            getattr(differentiable_forward, "last_adjoint_solution", None)
        )
        adjoint_residual_converged = (
            adjoint_metrics["adjoint/relative_residual"] <= cfg.adjoint_rtol
        )
        adjoint_metrics["adjoint/residual_converged"] = bool(adjoint_residual_converged)

        grad = active_activation_inv.grad
        if grad is None:
            msg = "differentiable forward did not produce activation gradients"
            raise RuntimeError(msg)
        if not torch.isfinite(grad).all():
            nonfinite = int((~torch.isfinite(grad)).sum().detach().cpu())
            msg = f"non-finite inverse gradient at step {step}: {nonfinite} entries"
            raise FloatingPointError(msg)

        error_stats = point_error_stats(residual.detach())
        loss_value = float(loss.detach().cpu())
        mean_error = float(error_stats["mean"].cpu())
        rms_error = float(error_stats["rms"].cpu())
        max_error = float(error_stats["max"].cpu())
        grad_norm = float(torch.linalg.vector_norm(grad).cpu())
        grad_abs_max = float(grad.abs().max().cpu())
        active_values = active_activation_inv.detach()
        activation_inv_rms = float(
            torch.linalg.vector_norm(active_values).cpu()
            / math.sqrt(active_values.numel())
        )
        activation_inv_min = float(active_values.min().cpu())
        activation_inv_max = float(active_values.max().cpu())
        displacement = to_numpy(output)[global_ids]

        improved = is_significant_decrease(
            loss_value,
            best_loss,
            rel_tol=cfg.stagnation_rel_tol,
            abs_tol=cfg.stagnation_abs_tol,
        )
        if loss_value < lowest_loss:
            lowest_loss = loss_value
            lowest_loss_step = step
        if max_error < lowest_max_error:
            lowest_max_error = max_error
            lowest_max_error_step = step
        if max_error < best_max_error or (
            math.isclose(max_error, best_max_error) and loss_value < best_loss
        ):
            best_step = step
            best_loss = loss_value
            best_max_error = max_error
            best_active_activation_inv = active_values.clone()
            save_checkpoint(
                cfg.checkpoint,
                best_active_activation_inv,
                step=step,
                loss=loss_value,
                max_error=max_error,
            )
            best_activation_inv = to_numpy(
                full_activation_inv_from_active(
                    active_values, active_ids_t, mesh.n_cells
                )
            )
            best_displacement = displacement
        if improved or step == 0:
            no_improve_steps = 0
        else:
            no_improve_steps += 1

        stopped = False
        if step >= cfg.inverse_min_steps and max_error <= cfg.max_point_error_cm:
            stop_reason = "max_point_error_tol"
            stopped = True
        elif loss_value <= cfg.loss_tol:
            stop_reason = "loss_tol"
            stopped = True
        elif (
            step >= cfg.inverse_min_steps
            and no_improve_steps >= cfg.stagnation_patience
        ):
            stop_reason = "stagnation"
            stopped = True

        trace_record = {
            "step": float(step),
            "loss/total": loss_value,
            "target/error_mean": mean_error,
            "target/error_rms": rms_error,
            "target/error_max": max_error,
            "activation_inv/rms": activation_inv_rms,
            "activation_inv/min": activation_inv_min,
            "activation_inv/max": activation_inv_max,
            "grad/norm": grad_norm,
            "grad/abs_max": grad_abs_max,
            "optimizer/steps": float(optimizer_steps),
            "optimizer/lr": float(optimizer.param_groups[0]["lr"]),
            "best/step": float(best_step),
            "best/loss": best_loss,
            "best/target_error_max": best_max_error,
            "lowest/loss": lowest_loss,
            "lowest/loss_step": float(lowest_loss_step),
            "lowest/target_error_max": lowest_max_error,
            "lowest/target_error_max_step": float(lowest_max_error_step),
            "stagnation/no_improve_steps": float(no_improve_steps),
            "stopped": float(stopped),
            "time/forward_s": forward_elapsed,
            "time/backward_s": backward_elapsed,
            **forward_metrics,
            **adjoint_metrics,
        }
        trace.append(trace_record)

        series_start = time.perf_counter()
        evaluated_activation_inv = to_numpy(
            full_activation_inv_from_active(active_values, active_ids_t, mesh.n_cells)
        )
        step_mesh = make_result_mesh(
            mesh,
            target_displacement,
            displacement,
            evaluated_activation_inv,
            target_ids,
            active_ids,
            {
                "inverse/step": step,
                **numeric_metrics(trace_record, exclude={"step"}),
            },
        )
        series_writer.append(step_mesh, time=float(step))
        series_elapsed = time.perf_counter() - series_start
        timing["time/series_s"] += series_elapsed
        trace_record["time/series_s"] = series_elapsed

        step_elapsed = time.perf_counter() - step_start
        trace_record["time/step_s"] = step_elapsed
        cherries.set_step(step)
        cherries.log_metrics(numeric_metrics(trace_record, exclude={"step"}))
        print(
            "inverse step:",
            f"{step:03d}",
            f"loss={loss_value:.3e}",
            f"mean={mean_error:.3e}cm",
            f"rms={rms_error:.3e}cm",
            f"max={max_error:.3e}cm",
            f"tol={cfg.max_point_error_cm:.3e}cm",
            f"best_max={best_max_error:.3e}cm",
            f"grad={grad_norm:.3e}",
            f"fwd={forward_metrics['forward/result']}/"
            f"{forward_metrics['forward/steps']}",
            f"fwd_grad={forward_metrics['forward/grad_norm']:.1e}",
            f"fwd_rel={forward_metrics['forward/relative_grad_norm']:.1e}",
            f"adj={adjoint_metrics['adjoint/result']}/"
            f"{adjoint_metrics['adjoint/best_solver']}",
            f"adj_abs={adjoint_metrics['adjoint/absolute_residual']:.1e}",
            f"adj_rel={adjoint_metrics['adjoint/relative_residual']:.1e}",
            f"lr={float(optimizer.param_groups[0]['lr']):.3e}",
            f"no_improve={no_improve_steps}",
            f"elapsed={step_elapsed:.2f}s",
            flush=True,
        )
        if stopped or optimizer_steps >= cfg.inverse_max_steps:
            break
        optimizer_start = time.perf_counter()
        optimizer.step()
        optimizer_steps += 1
        timing["time/optimizer_s"] += time.perf_counter() - optimizer_start
        with torch.no_grad():
            clamp_activation_inv_(active_activation_inv, cfg)

    if best_displacement is None or best_activation_inv is None:
        msg = "inverse solve did not evaluate any forward states"
        raise RuntimeError(msg)
    timing["time/inverse_s"] = time.perf_counter() - inverse_start
    return (
        best_displacement,
        best_activation_inv,
        trace,
        stop_reason,
        optimizer_steps,
        best_step,
        timing,
    )


def summarize(
    mesh: pv.UnstructuredGrid,
    target_displacement: np.ndarray,
    displacement: np.ndarray,
    recovered_activation_inv: np.ndarray,
    target_ids: np.ndarray,
    active_ids: np.ndarray,
    trace: list[dict[str, Any]],
    stop_reason: str,
    optimizer_steps: int,
    best_step: int,
    timing: dict[str, float],
    total_elapsed_s: float,
    cfg: Config,
) -> dict[str, Any]:
    error = displacement - target_displacement
    error_norm = np.linalg.norm(error, axis=1)
    target_error = error[target_ids]
    target_error_norm = np.linalg.norm(target_error, axis=1)
    target_norm = np.linalg.norm(target_displacement[target_ids], axis=1)
    active_activation_inv = recovered_activation_inv[active_ids]
    final_loss = float(np.mean(np.square(target_error)))
    target_rms = float(np.linalg.norm(target_error) / math.sqrt(target_ids.size))
    all_rms = float(np.linalg.norm(error) / math.sqrt(error.shape[0]))
    forward_failures = sum(
        1 for record in trace if not bool(record.get("forward/success", False))
    )
    adjoint_failures = sum(
        1 for record in trace if not bool(record.get("adjoint/success", False))
    )
    adjoint_relative_residuals = [
        float(record["adjoint/relative_residual"])
        for record in trace
        if np.isfinite(float(record.get("adjoint/relative_residual", math.nan)))
    ]
    adjoint_absolute_residuals = [
        float(record["adjoint/absolute_residual"])
        for record in trace
        if np.isfinite(float(record.get("adjoint/absolute_residual", math.nan)))
    ]
    forward_steps = [
        float(record["forward/steps"])
        for record in trace
        if np.isfinite(float(record.get("forward/steps", math.nan)))
    ]
    forward_grad_norms = [
        float(record["forward/grad_norm"])
        for record in trace
        if np.isfinite(float(record.get("forward/grad_norm", math.nan)))
    ]
    forward_relative_grad_norms = [
        float(record["forward/relative_grad_norm"])
        for record in trace
        if np.isfinite(float(record.get("forward/relative_grad_norm", math.nan)))
    ]
    metrics: dict[str, Any] = {
        "mesh/n_points": int(mesh.n_points),
        "mesh/n_cells": int(mesh.n_cells),
        "target/n_points": int(target_ids.size),
        "activation/n_active_tets": int(active_ids.size),
        "activation/n_params": int(active_ids.size * 6),
        "optimized_parameterization": "per active muscle tetrahedron ActivationInv, 6 DoF",
        "optimizer/steps": int(optimizer_steps),
        "best/step": int(best_step),
        "series/frames": int(sum(1 for record in trace if "time/series_s" in record)),
        "stop_reason": stop_reason,
        "time/total_s": float(total_elapsed_s),
        **{name: float(value) for name, value in timing.items()},
        "target/displacement_mean": float(target_norm.mean()),
        "target/displacement_rms": float(
            np.linalg.norm(target_displacement[target_ids]) / math.sqrt(target_ids.size)
        ),
        "target/displacement_max": float(target_norm.max()),
        "loss/final": final_loss,
        "best/loss": float(trace[best_step]["loss/total"]),
        "lowest/loss": float(min(record["loss/total"] for record in trace)),
        "lowest/target_error_max": float(
            min(record["target/error_max"] for record in trace)
        ),
        "forward/all_success": forward_failures == 0,
        "forward/failures": int(forward_failures),
        "forward/max_steps_used": float(max(forward_steps, default=math.nan)),
        "forward/max_grad_norm": float(max(forward_grad_norms, default=math.nan)),
        "forward/max_relative_grad_norm": float(
            max(forward_relative_grad_norms, default=math.nan)
        ),
        "adjoint/all_success": adjoint_failures == 0,
        "adjoint/failures": int(adjoint_failures),
        "adjoint/max_absolute_residual": float(
            max(adjoint_absolute_residuals, default=math.nan)
        ),
        "adjoint/max_relative_residual": float(
            max(adjoint_relative_residuals, default=math.nan)
        ),
        "target/error_mean": float(target_error_norm.mean()),
        "target/error_rms": target_rms,
        "target/error_max": float(target_error_norm.max()),
        "all/error_rms": all_rms,
        "all/error_max": float(error_norm.max()),
        "active_activation_inv/mean": active_activation_inv.mean(axis=0).tolist(),
        "active_activation_inv/min": active_activation_inv.min(axis=0).tolist(),
        "active_activation_inv/max": active_activation_inv.max(axis=0).tolist(),
        "active_activation_inv/rms": float(
            np.linalg.norm(active_activation_inv)
            / math.sqrt(active_activation_inv.size)
        ),
        "trace": trace,
    }
    metrics["repeated/forward_failure"] = bool(forward_failures > cfg.failure_patience)
    metrics["repeated/adjoint_failure"] = bool(adjoint_failures > cfg.failure_patience)
    metrics["passed"] = bool(
        metrics["target/error_max"] <= cfg.max_point_error_cm
        and not metrics["repeated/forward_failure"]
        and not metrics["repeated/adjoint_failure"]
        and np.isfinite(metrics["target/error_max"])
    )
    return metrics


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def numeric_metrics(
    summary: dict[str, Any], *, exclude: frozenset[str] = frozenset()
) -> dict[str, int | float]:
    return {
        name: value
        for name, value in summary.items()
        if name not in exclude
        and isinstance(value, int | float)
        and not isinstance(value, bool)
    }


def main(cfg: Config) -> None:
    total_start = time.perf_counter()
    configure_runtime()

    mesh, target = load_problem(cfg)
    target_ids = target_point_ids(target, cfg)
    active_ids = active_cell_ids(mesh, cfg)
    target_displacement = np.asarray(
        target.point_data["Displacement"], dtype=np.float64
    )

    add_masks(mesh, target_ids, active_ids)
    melon.save(cfg.output_input, mesh)
    melon.save(cfg.output_target, make_target_mesh(target, target_ids, active_ids))

    inverse_mesh = mesh.copy(deep=True)
    with melon.SeriesWriter(cfg.output_series, clear=True) as series_writer:
        (
            displacement,
            recovered_activation_inv,
            trace,
            stop_reason,
            optimizer_steps,
            best_step,
            timing,
        ) = solve_inverse(
            inverse_mesh,
            target_displacement,
            target_ids,
            active_ids,
            cfg,
            series_writer,
        )
    total_elapsed_s = time.perf_counter() - total_start
    summary = summarize(
        inverse_mesh,
        target_displacement,
        displacement,
        recovered_activation_inv,
        target_ids,
        active_ids,
        trace,
        stop_reason,
        optimizer_steps,
        best_step,
        timing,
        total_elapsed_s,
        cfg,
    )
    result = make_result_mesh(
        inverse_mesh,
        target_displacement,
        displacement,
        recovered_activation_inv,
        target_ids,
        active_ids,
        numeric_metrics(summary),
    )
    melon.save(cfg.output, result)
    save_json(cfg.output_summary, summary)
    cherries.log_metrics(numeric_metrics(summary, exclude={"trace"}))

    print(
        "inverse result:",
        f"stop={summary['stop_reason']}",
        f"best_step={summary['best/step']}",
        f"target_mean_error={summary['target/error_mean']:.3e}cm",
        f"target_rms_error={summary['target/error_rms']:.3e}cm",
        f"target_max_error={summary['target/error_max']:.3e}cm",
        f"loss={summary['loss/final']:.3e}",
        f"steps={summary['optimizer/steps']}",
    )
    print(f"saved: {cfg.output}")
    print(f"saved: {cfg.output_series}")
    print(f"saved: {cfg.output_summary}")
    if not summary["passed"]:
        msg = (
            "inverse solve did not meet the required max point error: "
            f"{summary['target/error_max']:.6g} cm > "
            f"{cfg.max_point_error_cm:.6g} cm"
        )
        raise RuntimeError(msg)


if __name__ == "__main__":
    cherries.main(main)
