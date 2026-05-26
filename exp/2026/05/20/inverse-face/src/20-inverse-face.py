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

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
PREP_STEM = "10-inverse-face"
OUTPUT_STEM = "20-inverse-face"
TARGET_SURFACE_MASK = "TargetSurfaceMask"
BACKGROUND_FRACTION = "BackgroundFraction"
ACTIVE_FRACTION = "ActiveFraction"
SMAS_STIFFNESS_FRACTION = "SmasStiffnessFraction"


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    input: Path = cherries.output(f"{PREP_STEM}-input.vtu")
    target: Path = cherries.output(f"{PREP_STEM}-target.vtu")
    output_input: Path = cherries.output(f"{OUTPUT_STEM}-input.vtu")
    output_target: Path = cherries.output(f"{OUTPUT_STEM}-target.vtu")
    output: Path = cherries.output(f"{OUTPUT_STEM}.vtu")
    output_series: Path = cherries.output(f"{OUTPUT_STEM}.vtu.series")
    output_summary: Path = cherries.output(f"{OUTPUT_STEM}-summary.json")
    report: Path = EXPERIMENT_DIR / "docs" / f"{OUTPUT_STEM}.md"

    E: float = 1.0
    nu: float = 0.49
    smas_stiffness_ratio: float = 1.0e2
    active_fraction_tol: float = 1.0e-3
    inverse_active_fraction_floor: float = 0.0
    inverse_active_fraction_mode: str = "all"
    target_point_mask: str = "IsFace"

    forward_rtol: float = 1.0e-2
    forward_atol: float = 1.0e-4
    forward_max_steps: int = 800
    require_forward_convergence: bool = True
    require_adjoint_convergence: bool = True

    inverse_lr: float = 0.02
    adam_beta1: float = 0.0
    adam_beta2: float = 0.9
    adam_eps: float = 1.0e-8
    inverse_max_steps: int = 120
    inverse_min_steps: int = 20
    best_metric: str = "target_max_error"
    stagnation_metric: str = "loss"
    stagnation_patience: int = 24
    stagnation_rel_tol: float = 1.0e-4
    stagnation_abs_tol: float = 1.0e-7
    lr_reduction_patience: int = 12
    lr_reduction_factor: float = 0.5
    max_lr_reductions: int = 6
    min_inverse_lr: float = 1.0e-4
    loss_tol: float = 1.0e-7
    max_point_error_cm: float = 0.2
    activation_l2_weight: float = 1.0e-9
    max_error_weight: float = 5.0
    over_tolerance_weight: float = 1.0
    p_norm_weight: float = 0.05
    p_norm: float = 8.0
    initial_activation_scale: float = 0.0
    initial_activation_surface_only: bool = False
    adjoint_maxiter: int = 60
    adjoint_rtol: float = 1.0e-2
    adjoint_atol: float = 0.0

    activation_inv_diag_min: float = -8.0
    activation_inv_diag_max: float = 8.0
    activation_inv_shear_abs_max: float = 3.0
    series_stride: int = 5


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


def apply_inverse_active_fraction_floor(
    mesh: pv.UnstructuredGrid, cfg: Config, target_ids: np.ndarray
) -> None:
    if cfg.inverse_active_fraction_floor <= 0.0:
        return
    active_fraction = np.asarray(mesh.cell_data[ACTIVE_FRACTION], dtype=np.float64)
    mode = cfg.inverse_active_fraction_mode.casefold()
    if mode == "all":
        cell_mask = np.ones(mesh.n_cells, dtype=bool)
    elif mode in {"target-adjacent", "face-adjacent"}:
        cells = np.asarray(mesh.cells, dtype=np.int64).reshape(mesh.n_cells, 5)[:, 1:]
        point_mask = np.zeros(mesh.n_points, dtype=bool)
        point_mask[target_ids] = True
        cell_mask = point_mask[cells].any(axis=1)
    else:
        msg = f"unknown inverse_active_fraction_mode: {cfg.inverse_active_fraction_mode!r}"
        raise ValueError(msg)
    active_fraction[cell_mask] = np.maximum(
        active_fraction[cell_mask], cfg.inverse_active_fraction_floor
    )
    mesh.cell_data[ACTIVE_FRACTION] = active_fraction
    mesh.cell_data["ActivationMask"] = (
        active_fraction > cfg.active_fraction_tol
    ).astype(np.int8)


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


def activation_inv_to_activation_numpy(activation_inv: np.ndarray) -> np.ndarray:
    activation_inv = np.asarray(activation_inv, dtype=np.float64)
    matrices = np.zeros((*activation_inv.shape[:-1], 3, 3), dtype=np.float64)
    matrices[..., 0, 0] = 1.0 + activation_inv[..., 0]
    matrices[..., 1, 1] = 1.0 + activation_inv[..., 1]
    matrices[..., 2, 2] = 1.0 + activation_inv[..., 2]
    matrices[..., 0, 1] = activation_inv[..., 3]
    matrices[..., 1, 0] = activation_inv[..., 3]
    matrices[..., 0, 2] = activation_inv[..., 4]
    matrices[..., 2, 0] = activation_inv[..., 4]
    matrices[..., 1, 2] = activation_inv[..., 5]
    matrices[..., 2, 1] = activation_inv[..., 5]
    activation = np.linalg.pinv(matrices, rcond=1.0e-8)
    packed = np.stack(
        (
            activation[..., 0, 0] - 1.0,
            activation[..., 1, 1] - 1.0,
            activation[..., 2, 2] - 1.0,
            activation[..., 0, 1],
            activation[..., 0, 2],
            activation[..., 1, 2],
        ),
        axis=-1,
    )
    return np.nan_to_num(packed, nan=0.0, posinf=1.0e6, neginf=-1.0e6)


def clamp_activation_inv_(activation_inv: torch.Tensor, cfg: Config) -> None:
    activation_inv[:, :3].clamp_(
        cfg.activation_inv_diag_min, cfg.activation_inv_diag_max
    )
    activation_inv[:, 3:].clamp_(
        -cfg.activation_inv_shear_abs_max, cfg.activation_inv_shear_abs_max
    )


def clamp_activation_inv_numpy(activation_inv: np.ndarray, cfg: Config) -> np.ndarray:
    result = np.asarray(activation_inv, dtype=np.float64).copy()
    result[:, :3] = np.clip(
        result[:, :3], cfg.activation_inv_diag_min, cfg.activation_inv_diag_max
    )
    result[:, 3:] = np.clip(
        result[:, 3:],
        -cfg.activation_inv_shear_abs_max,
        cfg.activation_inv_shear_abs_max,
    )
    return result


def initial_active_activation_inv(
    mesh: pv.UnstructuredGrid,
    target_displacement: np.ndarray,
    target_ids: np.ndarray,
    active_ids: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    cells = np.asarray(mesh.cells, dtype=np.int64).reshape(mesh.n_cells, 5)[:, 1:]
    active_cells = cells[active_ids]
    rest = np.asarray(mesh.points, dtype=np.float64)
    displacement = np.asarray(target_displacement, dtype=np.float64)
    if cfg.initial_activation_surface_only:
        surface_displacement = np.zeros_like(displacement)
        surface_displacement[target_ids] = displacement[target_ids]
        displacement = surface_displacement
    target = rest + displacement

    X0 = rest[active_cells[:, 0]]
    x0 = target[active_cells[:, 0]]
    Dm = np.stack(
        (
            rest[active_cells[:, 1]] - X0,
            rest[active_cells[:, 2]] - X0,
            rest[active_cells[:, 3]] - X0,
        ),
        axis=-1,
    )
    Ds = np.stack(
        (
            target[active_cells[:, 1]] - x0,
            target[active_cells[:, 2]] - x0,
            target[active_cells[:, 3]] - x0,
        ),
        axis=-1,
    )
    deformation = Ds @ np.linalg.inv(Dm)
    right_cauchy_green = np.swapaxes(deformation, -1, -2) @ deformation
    eigvals, eigvecs = np.linalg.eigh(right_cauchy_green)
    eigvals = np.clip(eigvals, 1.0e-8, None)
    activation_inv_matrix = (
        eigvecs
        @ (np.reciprocal(np.sqrt(eigvals))[..., None] * np.swapaxes(eigvecs, -1, -2))
    )
    activation_inv = np.stack(
        (
            activation_inv_matrix[:, 0, 0] - 1.0,
            activation_inv_matrix[:, 1, 1] - 1.0,
            activation_inv_matrix[:, 2, 2] - 1.0,
            activation_inv_matrix[:, 0, 1],
            activation_inv_matrix[:, 0, 2],
            activation_inv_matrix[:, 1, 2],
        ),
        axis=-1,
    )
    activation_inv = cfg.initial_activation_scale * activation_inv
    activation_inv = np.nan_to_num(activation_inv, nan=0.0, posinf=0.0, neginf=0.0)
    return clamp_activation_inv_numpy(activation_inv, cfg)


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
            "forward_result": "missing",
            "forward_success": False,
            "forward_steps": math.nan,
            "forward_grad_norm": math.nan,
            "forward_grad_norm_first": math.nan,
            "forward_line_search_ok": False,
            "forward_line_search_steps": math.nan,
            "forward_stagnation_count": math.nan,
        }
    convergence_state = solution.state.convergence_state
    line_search_state = solution.state.line_search_state
    return {
        "forward_result": str(solution.result),
        "forward_success": bool(solution.success),
        "forward_steps": int(convergence_state.step),
        "forward_grad_norm": to_float(convergence_state.grad_norm),
        "forward_grad_norm_first": to_float(convergence_state.grad_norm_first),
        "forward_line_search_ok": bool(line_search_state.ok),
        "forward_line_search_steps": int(line_search_state.step),
        "forward_stagnation_count": int(convergence_state.stagnation_count),
    }


def adjoint_solution_metrics(solution: Any) -> dict[str, Any]:
    if solution is None:
        return {
            "adjoint_result": "missing",
            "adjoint_success": False,
            "adjoint_solver_count": 0,
            "adjoint_best_solver": -1,
            "adjoint_absolute_residual": math.nan,
            "adjoint_relative_residual": math.nan,
        }
    state = solution.state
    best_index = int(state.best_index.detach().cpu())
    absolute_residuals = to_numpy(state.absolute_residuals)
    relative_residuals = to_numpy(state.relative_residuals)
    metrics: dict[str, Any] = {
        "adjoint_result": str(solution.result),
        "adjoint_success": bool(solution.success),
        "adjoint_solver_count": len(state.solutions),
        "adjoint_best_solver": best_index,
        "adjoint_absolute_residual": float(absolute_residuals[best_index]),
        "adjoint_relative_residual": float(relative_residuals[best_index]),
    }
    for i, solver_solution in enumerate(state.solutions):
        metrics[f"adjoint_solver_{i}_result"] = str(solver_solution.result)
        metrics[f"adjoint_solver_{i}_success"] = bool(solver_solution.success)
        metrics[f"adjoint_solver_{i}_steps"] = (
            -1
            if solver_solution.state.step is None
            else int(solver_solution.state.step)
        )
        metrics[f"adjoint_solver_{i}_info"] = int(solver_solution.state.info)
        metrics[f"adjoint_solver_{i}_relative_residual"] = float(relative_residuals[i])
    return metrics


def point_error_stats(residual: torch.Tensor) -> dict[str, torch.Tensor]:
    point_error = torch.linalg.vector_norm(residual, dim=1)
    return {
        "mean": point_error.mean(),
        "rms": torch.linalg.vector_norm(residual) / math.sqrt(residual.shape[0]),
        "max": point_error.max(),
    }


def choose_metric(metrics: dict[str, float], name: str) -> float:
    normalized = name.casefold().replace("-", "_")
    aliases = {
        "objective": "loss",
        "objective_loss": "loss",
        "mse": "data_loss",
        "mean_square": "data_loss",
        "mean_square_error": "data_loss",
        "mean": "target_mean_error",
        "rms": "target_rms_error",
        "max": "target_max_error",
        "max_error": "target_max_error",
    }
    key = aliases.get(normalized, normalized)
    if key not in metrics:
        choices = ", ".join(sorted(metrics))
        msg = f"unknown metric {name!r}; choose one of: {choices}"
        raise ValueError(msg)
    return metrics[key]


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
    recovered_activation: np.ndarray,
    recovered_activation_inv: np.ndarray,
    target_ids: np.ndarray,
    active_ids: np.ndarray,
    metrics: dict[str, float | int | bool | str],
) -> pv.UnstructuredGrid:
    from liblaf.apple.common import ACTIVATION, ACTIVATION_INV

    result = mesh.copy(deep=True)
    add_masks(result, target_ids, active_ids)
    error = displacement - target_displacement
    result.point_data["Displacement"] = displacement
    result.point_data["TargetDisplacement"] = target_displacement
    result.point_data["DisplacementError"] = error
    result.point_data["DisplacementErrorNorm"] = np.linalg.norm(error, axis=1)
    result.point_data["DeformedPoint"] = result.points + displacement
    result.point_data["TargetPoint"] = result.points + target_displacement
    result.cell_data[ACTIVATION.vtk] = recovered_activation
    result.cell_data[ACTIVATION_INV.vtk] = recovered_activation_inv
    result.cell_data["RecoveredActivation"] = recovered_activation
    result.cell_data["RecoveredActivationInv"] = recovered_activation_inv
    result.cell_data["RecoveredActivationNorm"] = np.linalg.norm(
        recovered_activation, axis=1
    )
    result.cell_data["RecoveredActivationInvNorm"] = np.linalg.norm(
        recovered_activation_inv, axis=1
    )
    add_metric_fields(result, metrics)
    return result


def should_write_series(
    step: int, *, stopped: bool, improved: bool, cfg: Config
) -> bool:
    _ = improved
    stride = max(1, cfg.series_stride)
    return step == 0 or stopped or step % stride == 0


def solve_inverse(  # noqa: C901, PLR0912, PLR0915
    mesh: pv.UnstructuredGrid,
    target_displacement: np.ndarray,
    target_ids: np.ndarray,
    active_ids: np.ndarray,
    initial_activation_inv: np.ndarray,
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
        torch.as_tensor(
            initial_activation_inv,
            dtype=torch.get_default_dtype(),
            device=torch.get_default_device(),
        )
    )
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
    best_state_u: torch.Tensor | None = None
    best_metric_value = math.inf
    best_loss = math.inf
    best_data_loss = math.inf
    best_max_error = math.inf
    best_stagnation_metric = math.inf
    lowest_loss = math.inf
    lowest_data_loss = math.inf
    lowest_max_error = math.inf
    lowest_loss_step = 0
    lowest_data_loss_step = 0
    lowest_max_error_step = 0
    no_improve_steps = 0
    lr_reductions = 0
    timing = {
        "inverse_elapsed_s": 0.0,
        "forward_elapsed_s": 0.0,
        "backward_elapsed_s": 0.0,
        "optimizer_elapsed_s": 0.0,
        "series_elapsed_s": 0.0,
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
        timing["forward_elapsed_s"] += forward_elapsed
        forward_metrics = forward_solution_metrics(
            getattr(differentiable_forward, "last_forward_solution", None)
        )
        if cfg.require_forward_convergence and not forward_metrics["forward_success"]:
            msg = (
                f"forward solve did not converge at inverse step {step}: "
                f"{forward_metrics['forward_result']}"
            )
            raise RuntimeError(msg)

        backward_start = time.perf_counter()
        residual = output[point_global_ids_t] - target[point_ids_t]
        point_error = torch.linalg.vector_norm(residual, dim=1)
        data_loss = residual.square().mean()
        max_error_loss = torch.relu(
            point_error.max() - cfg.max_point_error_cm
        ).square()
        over_tolerance_loss = (
            torch.relu(point_error - cfg.max_point_error_cm).square().mean()
        )
        p_norm_loss = point_error.pow(cfg.p_norm).mean().pow(2.0 / cfg.p_norm)
        reg_loss = cfg.activation_l2_weight * active_activation_inv.square().mean()
        loss = (
            data_loss
            + cfg.max_error_weight * max_error_loss
            + cfg.over_tolerance_weight * over_tolerance_loss
            + cfg.p_norm_weight * p_norm_loss
            + reg_loss
        )
        loss.backward()
        backward_elapsed = time.perf_counter() - backward_start
        timing["backward_elapsed_s"] += backward_elapsed
        adjoint_metrics = adjoint_solution_metrics(
            getattr(differentiable_forward, "last_adjoint_solution", None)
        )
        if cfg.require_adjoint_convergence and not adjoint_metrics["adjoint_success"]:
            msg = (
                f"adjoint solve did not converge at inverse step {step}: "
                f"{adjoint_metrics['adjoint_result']} "
                f"(relative residual {adjoint_metrics['adjoint_relative_residual']:.3e})"
            )
            raise RuntimeError(msg)

        grad = active_activation_inv.grad
        if grad is None:
            msg = "differentiable forward did not produce activation gradients"
            raise RuntimeError(msg)
        if not torch.isfinite(grad).all():
            nonfinite = int((~torch.isfinite(grad)).sum().detach().cpu())
            msg = f"non-finite inverse gradient at step {step}: {nonfinite} entries"
            raise FloatingPointError(msg)

        error_stats = point_error_stats(residual.detach())
        data_loss_value = float(data_loss.detach().cpu())
        loss_value = float(loss.detach().cpu())
        reg_loss_value = float(reg_loss.detach().cpu())
        max_error_loss_value = float(max_error_loss.detach().cpu())
        over_tolerance_loss_value = float(over_tolerance_loss.detach().cpu())
        p_norm_loss_value = float(p_norm_loss.detach().cpu())
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

        metric_values = {
            "loss": loss_value,
            "data_loss": data_loss_value,
            "target_mean_error": mean_error,
            "target_rms_error": rms_error,
            "target_max_error": max_error,
            "max_error_loss": max_error_loss_value,
            "over_tolerance_loss": over_tolerance_loss_value,
            "p_norm_loss": p_norm_loss_value,
        }
        current_best_metric = choose_metric(metric_values, cfg.best_metric)
        current_stagnation_metric = choose_metric(metric_values, cfg.stagnation_metric)
        improved = is_significant_decrease(
            current_stagnation_metric,
            best_stagnation_metric,
            rel_tol=cfg.stagnation_rel_tol,
            abs_tol=cfg.stagnation_abs_tol,
        )
        best_improved = current_best_metric < best_metric_value
        if loss_value < lowest_loss:
            lowest_loss = loss_value
            lowest_loss_step = step
        if data_loss_value < lowest_data_loss:
            lowest_data_loss = data_loss_value
            lowest_data_loss_step = step
        if max_error < lowest_max_error:
            lowest_max_error = max_error
            lowest_max_error_step = step
        best_stagnation_metric = min(best_stagnation_metric, current_stagnation_metric)
        if best_improved:
            best_step = step
            best_metric_value = current_best_metric
            best_loss = loss_value
            best_data_loss = data_loss_value
            best_max_error = max_error
            best_active_activation_inv = active_values.clone()
            best_state_u = output.detach().clone()
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
            and (
                lr_reductions >= cfg.max_lr_reductions
                or optimizer.param_groups[0]["lr"] <= cfg.min_inverse_lr
            )
        ):
            stop_reason = "stagnation"
            stopped = True

        did_optimizer_step = False
        lr_reduced = False
        if (
            not stopped
            and step >= cfg.inverse_min_steps
            and no_improve_steps >= cfg.lr_reduction_patience
            and lr_reductions < cfg.max_lr_reductions
            and optimizer.param_groups[0]["lr"] > cfg.min_inverse_lr
            and best_active_activation_inv is not None
        ):
            with torch.no_grad():
                active_activation_inv.copy_(best_active_activation_inv)
            if best_state_u is not None:
                forward.state = forward.model.State(u=best_state_u.clone())
            forward.optimizer = forward.default_optimizer(
                max_steps=cfg.forward_max_steps,
                atol=cfg.forward_atol,
                rtol=cfg.forward_rtol,
            )
            new_lr = max(
                cfg.min_inverse_lr,
                float(optimizer.param_groups[0]["lr"]) * cfg.lr_reduction_factor,
            )
            optimizer = torch.optim.Adam(
                [active_activation_inv],
                lr=new_lr,
                betas=(cfg.adam_beta1, cfg.adam_beta2),
                eps=cfg.adam_eps,
            )
            lr_reductions += 1
            no_improve_steps = 0
            lr_reduced = True
        if not stopped and not lr_reduced and optimizer_steps < cfg.inverse_max_steps:
            optimizer_start = time.perf_counter()
            optimizer.step()
            optimizer_steps += 1
            did_optimizer_step = True
            timing["optimizer_elapsed_s"] += time.perf_counter() - optimizer_start
            with torch.no_grad():
                clamp_activation_inv_(active_activation_inv, cfg)

        trace_record = {
            "step": float(step),
            "loss": loss_value,
            "data_loss": data_loss_value,
            "max_error_loss": max_error_loss_value,
            "over_tolerance_loss": over_tolerance_loss_value,
            "p_norm_loss": p_norm_loss_value,
            "regularization_loss": reg_loss_value,
            "target_mean_error": mean_error,
            "target_rms_error": rms_error,
            "target_max_error": max_error,
            "max_point_error_cm": cfg.max_point_error_cm,
            "activation_inv_rms": activation_inv_rms,
            "activation_inv_min": activation_inv_min,
            "activation_inv_max": activation_inv_max,
            "grad_norm": grad_norm,
            "grad_abs_max": grad_abs_max,
            "optimizer_steps": float(optimizer_steps),
            "inverse_lr": float(optimizer.param_groups[0]["lr"]),
            "lr_reductions": float(lr_reductions),
            "lr_reduced": float(lr_reduced),
            "best_step": float(best_step),
            "best_metric": best_metric_value,
            "best_loss": best_loss,
            "best_data_loss": best_data_loss,
            "best_target_max_error": best_max_error,
            "best_stagnation_metric": best_stagnation_metric,
            "lowest_loss": lowest_loss,
            "lowest_loss_step": float(lowest_loss_step),
            "lowest_data_loss": lowest_data_loss,
            "lowest_data_loss_step": float(lowest_data_loss_step),
            "lowest_target_max_error": lowest_max_error,
            "lowest_target_max_error_step": float(lowest_max_error_step),
            "no_improve_steps": float(no_improve_steps),
            "stopped": float(stopped),
            "forward_elapsed_s": forward_elapsed,
            "backward_elapsed_s": backward_elapsed,
            **forward_metrics,
            **adjoint_metrics,
        }
        trace.append(trace_record)

        if should_write_series(step, stopped=stopped, improved=improved, cfg=cfg):
            series_start = time.perf_counter()
            evaluated_activation_inv = to_numpy(
                full_activation_inv_from_active(
                    active_values, active_ids_t, mesh.n_cells
                )
            )
            evaluated_activation = activation_inv_to_activation_numpy(
                evaluated_activation_inv
            )
            step_mesh = make_result_mesh(
                mesh,
                target_displacement,
                displacement,
                evaluated_activation,
                evaluated_activation_inv,
                target_ids,
                active_ids,
                {
                    "inverse_step": step,
                    "optimizer_steps": optimizer_steps,
                    "loss": loss_value,
                    "data_loss": data_loss_value,
                    "target_mean_error": mean_error,
                    "target_rms_error": rms_error,
                    "target_max_error": max_error,
                    "max_point_error_cm": cfg.max_point_error_cm,
                    "activation_inv_rms": activation_inv_rms,
                    "activation_inv_min": activation_inv_min,
                    "activation_inv_max": activation_inv_max,
                    "grad_norm": grad_norm,
                    "grad_abs_max": grad_abs_max,
                    "best_step": best_step,
                    "best_loss": best_loss,
                    "best_data_loss": best_data_loss,
                    "best_target_max_error": best_max_error,
                    "lowest_loss": lowest_loss,
                    "lowest_data_loss": lowest_data_loss,
                    "lowest_target_max_error": lowest_max_error,
                    "forward_success": forward_metrics["forward_success"],
                    "forward_steps": forward_metrics["forward_steps"],
                    "forward_grad_norm": forward_metrics["forward_grad_norm"],
                    "adjoint_success": adjoint_metrics["adjoint_success"],
                    "adjoint_relative_residual": adjoint_metrics[
                        "adjoint_relative_residual"
                    ],
                    "stopped": stopped,
                },
            )
            series_writer.append(step_mesh, time=float(step))
            series_elapsed = time.perf_counter() - series_start
            timing["series_elapsed_s"] += series_elapsed
            trace_record["series_elapsed_s"] = series_elapsed

        step_elapsed = time.perf_counter() - step_start
        trace_record["step_elapsed_s"] = step_elapsed
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
            f"fwd={forward_metrics['forward_result']}/"
            f"{forward_metrics['forward_steps']}",
            f"adj={adjoint_metrics['adjoint_result']}/"
            f"{adjoint_metrics['adjoint_relative_residual']:.1e}",
            f"lr={float(optimizer.param_groups[0]['lr']):.3e}",
            f"lr_cuts={lr_reductions}",
            f"no_improve={no_improve_steps}",
            f"elapsed={step_elapsed:.2f}s",
            flush=True,
        )
        if stopped or not (did_optimizer_step or lr_reduced):
            break

    if best_displacement is None or best_activation_inv is None:
        msg = "inverse solve did not evaluate any forward states"
        raise RuntimeError(msg)
    timing["inverse_elapsed_s"] = time.perf_counter() - inverse_start
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
    recovered_activation: np.ndarray,
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
    active_activation = recovered_activation[active_ids]
    active_activation_inv = recovered_activation_inv[active_ids]
    final_loss = float(np.mean(np.square(target_error)))
    target_rms = float(np.linalg.norm(target_error) / math.sqrt(target_ids.size))
    all_rms = float(np.linalg.norm(error) / math.sqrt(error.shape[0]))
    forward_failures = sum(
        1 for record in trace if not bool(record.get("forward_success", False))
    )
    adjoint_failures = sum(
        1 for record in trace if not bool(record.get("adjoint_success", False))
    )
    adjoint_relative_residuals = [
        float(record["adjoint_relative_residual"])
        for record in trace
        if np.isfinite(float(record.get("adjoint_relative_residual", math.nan)))
    ]
    forward_steps = [
        float(record["forward_steps"])
        for record in trace
        if np.isfinite(float(record.get("forward_steps", math.nan)))
    ]
    metrics: dict[str, Any] = {
        "input": str(cfg.input),
        "target": str(cfg.target),
        "output": str(cfg.output),
        "output_series": str(cfg.output_series),
        "output_summary": str(cfg.output_summary),
        "report": str(cfg.report),
        "n_points": int(mesh.n_points),
        "n_cells": int(mesh.n_cells),
        "n_target_points": int(target_ids.size),
        "n_active_tets": int(active_ids.size),
        "n_activation_params": int(active_ids.size * 6),
        "optimized_parameterization": "per active muscle tetrahedron ActivationInv, 6 DoF",
        "E": float(cfg.E),
        "nu": float(cfg.nu),
        "smas_stiffness_ratio": float(cfg.smas_stiffness_ratio),
        "inverse_active_fraction_floor": float(cfg.inverse_active_fraction_floor),
        "inverse_active_fraction_mode": cfg.inverse_active_fraction_mode,
        "inverse_lr": float(cfg.inverse_lr),
        "adam_beta1": float(cfg.adam_beta1),
        "adam_beta2": float(cfg.adam_beta2),
        "adam_eps": float(cfg.adam_eps),
        "forward_rtol": float(cfg.forward_rtol),
        "forward_atol": float(cfg.forward_atol),
        "forward_max_steps": int(cfg.forward_max_steps),
        "require_forward_convergence": bool(cfg.require_forward_convergence),
        "require_adjoint_convergence": bool(cfg.require_adjoint_convergence),
        "inverse_max_steps": int(cfg.inverse_max_steps),
        "inverse_min_steps": int(cfg.inverse_min_steps),
        "best_metric": cfg.best_metric,
        "stagnation_metric": cfg.stagnation_metric,
        "optimizer_steps": int(optimizer_steps),
        "best_step": int(best_step),
        "series_stride": int(cfg.series_stride),
        "series_frames": int(
            sum(1 for record in trace if "series_elapsed_s" in record)
        ),
        "stop_reason": stop_reason,
        "max_point_error_cm": float(cfg.max_point_error_cm),
        "loss_tol": float(cfg.loss_tol),
        "stagnation_patience": int(cfg.stagnation_patience),
        "stagnation_rel_tol": float(cfg.stagnation_rel_tol),
        "lr_reduction_patience": int(cfg.lr_reduction_patience),
        "lr_reduction_factor": float(cfg.lr_reduction_factor),
        "max_lr_reductions": int(cfg.max_lr_reductions),
        "min_inverse_lr": float(cfg.min_inverse_lr),
        "activation_l2_weight": float(cfg.activation_l2_weight),
        "max_error_weight": float(cfg.max_error_weight),
        "over_tolerance_weight": float(cfg.over_tolerance_weight),
        "p_norm_weight": float(cfg.p_norm_weight),
        "p_norm": float(cfg.p_norm),
        "initial_activation_scale": float(cfg.initial_activation_scale),
        "initial_activation_surface_only": bool(cfg.initial_activation_surface_only),
        "adjoint_maxiter": int(cfg.adjoint_maxiter),
        "adjoint_rtol": float(cfg.adjoint_rtol),
        "adjoint_atol": float(cfg.adjoint_atol),
        "total_elapsed_s": float(total_elapsed_s),
        **{name: float(value) for name, value in timing.items()},
        "target_displacement_mean": float(target_norm.mean()),
        "target_displacement_rms": float(
            np.linalg.norm(target_displacement[target_ids]) / math.sqrt(target_ids.size)
        ),
        "target_displacement_max": float(target_norm.max()),
        "final_loss": final_loss,
        "best_loss": float(trace[best_step]["loss"]),
        "best_data_loss": float(trace[best_step]["data_loss"]),
        "lowest_loss": float(min(record["loss"] for record in trace)),
        "lowest_data_loss": float(min(record["data_loss"] for record in trace)),
        "lowest_target_max_error": float(
            min(record["target_max_error"] for record in trace)
        ),
        "forward_all_success": forward_failures == 0,
        "forward_failures": int(forward_failures),
        "forward_max_steps_used": float(max(forward_steps, default=math.nan)),
        "adjoint_all_success": adjoint_failures == 0,
        "adjoint_failures": int(adjoint_failures),
        "adjoint_max_relative_residual": float(
            max(adjoint_relative_residuals, default=math.nan)
        ),
        "target_mean_error": float(target_error_norm.mean()),
        "target_rms_error": target_rms,
        "target_max_error": float(target_error_norm.max()),
        "all_rms_error": all_rms,
        "all_max_error": float(error_norm.max()),
        "active_activation_mean": active_activation.mean(axis=0).tolist(),
        "active_activation_min": active_activation.min(axis=0).tolist(),
        "active_activation_max": active_activation.max(axis=0).tolist(),
        "active_activation_rms": float(
            np.linalg.norm(active_activation) / math.sqrt(active_activation.size)
        ),
        "active_activation_inv_mean": active_activation_inv.mean(axis=0).tolist(),
        "active_activation_inv_min": active_activation_inv.min(axis=0).tolist(),
        "active_activation_inv_max": active_activation_inv.max(axis=0).tolist(),
        "active_activation_inv_rms": float(
            np.linalg.norm(active_activation_inv)
            / math.sqrt(active_activation_inv.size)
        ),
        "trace": trace,
    }
    metrics["passed"] = bool(
        metrics["target_max_error"] <= cfg.max_point_error_cm
        and metrics["forward_all_success"]
        and metrics["adjoint_all_success"]
        and np.isfinite(metrics["target_max_error"])
    )
    return metrics


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def fmt(value: Any, precision: int = 6) -> str:
    if isinstance(value, float):
        return f"{value:.{precision}g}"
    return str(value)


def save_markdown_report(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 20 Inverse Face",
        "",
        "## Result",
        "",
        f"- stop reason: `{summary['stop_reason']}`",
        f"- passed: `{summary['passed']}`",
        f"- best step: `{summary['best_step']}`",
        f"- target mean error: `{fmt(summary['target_mean_error'])} cm`",
        f"- target RMS error: `{fmt(summary['target_rms_error'])} cm`",
        f"- target max error: `{fmt(summary['target_max_error'])} cm`",
        f"- required max error: `< {fmt(summary['max_point_error_cm'])} cm`",
        f"- final loss: `{fmt(summary['final_loss'])}`",
        f"- best objective loss: `{fmt(summary['best_loss'])}`",
        f"- lowest objective loss: `{fmt(summary['lowest_loss'])}`",
        f"- optimizer steps: `{summary['optimizer_steps']}`",
        f"- series frames: `{summary['series_frames']}`",
        f"- forward converged: `{summary['forward_all_success']}` "
        f"({summary['forward_failures']} failures)",
        f"- adjoint converged: `{summary['adjoint_all_success']}` "
        f"({summary['adjoint_failures']} failures, "
        f"max relative residual `{fmt(summary['adjoint_max_relative_residual'])}`)",
        "",
        "## Problem",
        "",
        f"- input: `{summary['input']}`",
        f"- target: `{summary['target']}`",
        f"- output: `{summary['output']}`",
        f"- optimization series: `{summary['output_series']}`",
        f"- points: `{summary['n_points']}`",
        f"- tetrahedra: `{summary['n_cells']}`",
        f"- target `IsFace` points: `{summary['n_target_points']}`",
        f"- active muscle tetrahedra: `{summary['n_active_tets']}`",
        f"- activation parameters: `{summary['n_activation_params']}`",
        f"- target displacement max: `{fmt(summary['target_displacement_max'])} cm`",
        "",
        "## Model",
        "",
        f"- material: stable neo-Hookean, `nu = {summary['nu']}`",
        f"- SMAS stiffness ratio: `{summary['smas_stiffness_ratio']}`",
        "- collisions: `off`",
        f"- optimized field: `{summary['optimized_parameterization']}`",
        f"- best metric: `{summary['best_metric']}`",
        f"- stagnation metric: `{summary['stagnation_metric']}`",
        f"- Adam: `lr={summary['inverse_lr']}`, "
        f"`betas=({summary['adam_beta1']}, {summary['adam_beta2']})`",
        f"- forward tolerance: `rtol={summary['forward_rtol']}`, "
        f"`atol={summary['forward_atol']}`",
        f"- adjoint tolerance: `rtol={summary['adjoint_rtol']}`, "
        f"`atol={summary['adjoint_atol']}`",
        "",
        "## Trace",
        "",
        "| step | loss | mean error (cm) | max error (cm) | best max (cm) | grad | fwd | adj rel |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: |",
    ]
    lines.extend(
        (
            "| "
            f"{int(record['step'])} | "
            f"{fmt(record['loss'])} | "
            f"{fmt(record['target_mean_error'])} | "
            f"{fmt(record['target_max_error'])} | "
            f"{fmt(record['best_target_max_error'])} | "
            f"{fmt(record['grad_norm'])} | "
            f"{record.get('forward_result', 'n/a')}/{fmt(record.get('forward_steps', math.nan))} | "
            f"{fmt(record.get('adjoint_relative_residual', math.nan))} |"
        )
        for record in summary["trace"]
    )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def save_readme(path: Path, report: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    relative_report = report.name
    path.write_text(f"# Inverse Face\n\n- [{relative_report}]({relative_report})\n")


def numeric_metrics(summary: dict[str, Any]) -> dict[str, int | float | bool]:
    return {
        name: value
        for name, value in summary.items()
        if isinstance(value, int | float | bool)
    }


def main(cfg: Config) -> None:
    total_start = time.perf_counter()
    configure_runtime()

    mesh, target = load_problem(cfg)
    target_ids = target_point_ids(target, cfg)
    apply_inverse_active_fraction_floor(mesh, cfg, target_ids)
    apply_inverse_active_fraction_floor(target, cfg, target_ids)
    active_ids = active_cell_ids(mesh, cfg)
    target_displacement = np.asarray(
        target.point_data["Displacement"], dtype=np.float64
    )
    initial_activation_inv = initial_active_activation_inv(
        mesh, target_displacement, target_ids, active_ids, cfg
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
            initial_activation_inv,
            cfg,
            series_writer,
        )
    recovered_activation = activation_inv_to_activation_numpy(recovered_activation_inv)
    total_elapsed_s = time.perf_counter() - total_start
    summary = summarize(
        inverse_mesh,
        target_displacement,
        displacement,
        recovered_activation,
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
        recovered_activation,
        recovered_activation_inv,
        target_ids,
        active_ids,
        numeric_metrics(summary),
    )
    melon.save(cfg.output, result)
    save_json(cfg.output_summary, summary)
    save_markdown_report(cfg.report, summary)
    save_readme(EXPERIMENT_DIR / "docs" / "README.md", cfg.report)
    cherries.log_metrics(numeric_metrics(summary))

    print(
        "inverse result:",
        f"stop={summary['stop_reason']}",
        f"best_step={summary['best_step']}",
        f"target_mean_error={summary['target_mean_error']:.3e}cm",
        f"target_rms_error={summary['target_rms_error']:.3e}cm",
        f"target_max_error={summary['target_max_error']:.3e}cm",
        f"loss={summary['final_loss']:.3e}",
        f"steps={summary['optimizer_steps']}",
    )
    print(f"saved: {cfg.output}")
    print(f"saved: {cfg.output_series}")
    print(f"saved: {cfg.output_summary}")
    print(f"saved: {cfg.report}")
    if not summary["passed"]:
        msg = (
            "inverse solve did not meet the required max point error: "
            f"{summary['target_max_error']:.6g} cm > "
            f"{cfg.max_point_error_cm:.6g} cm"
        )
        raise RuntimeError(msg)


if __name__ == "__main__":
    cherries.main(main)
