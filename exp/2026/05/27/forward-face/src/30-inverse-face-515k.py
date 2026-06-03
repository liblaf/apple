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

logger = logging.getLogger(__name__)

INPUT_STEM = "10-forward-face-515k-nosmas"
TARGET_STEM = "20-forward-face-515k-nosmas"
OUTPUT_STEM = "30-inverse-face-515k-nosmas"
BACKGROUND_FRACTION = "BackgroundFraction"
ACTIVE_FRACTION = "ActiveFraction"
SMAS_STIFFNESS_FRACTION = "SmasStiffnessFraction"
TARGET_SURFACE_MASK = "TargetSurfaceMask"


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    input: Path = cherries.input(f"{INPUT_STEM}-input.vtu")
    target: Path = cherries.input(f"{TARGET_STEM}.vtu")
    output_input: Path = cherries.output(f"{OUTPUT_STEM}-input.vtu")
    output_target: Path = cherries.output(f"{OUTPUT_STEM}-target.vtu")
    output: Path = cherries.output(f"{OUTPUT_STEM}.vtu")
    output_snapshot: Path = cherries.output(f"{OUTPUT_STEM}.png")
    output_summary: Path = cherries.output(f"{OUTPUT_STEM}-summary.json")
    checkpoint: Path = cherries.output(f"{OUTPUT_STEM}-checkpoint.npz")

    E: float = 1.0
    nu: float = 0.49
    smas_stiffness_ratio: float = 1.0
    use_smas: bool = False
    target_point_mask: str = "IsFace"

    forward_rtol: float = 5.0e-4
    forward_atol: float = 0.0
    forward_max_steps: int = 10000

    inverse_lr: float = 0.08
    adam_beta1: float = 0.3
    adam_beta2: float = 0.9
    adam_eps: float = 1.0e-8
    inverse_max_steps: int = 180
    inverse_min_steps: int = 20
    loss_tol: float = 1.0e-9
    max_point_error_cm: float = 5.0e-3
    stop_on_max_point_error: bool = True
    adjoint_maxiter: int = 10000
    adjoint_rtol: float = 5.0e-4
    adjoint_atol: float = 0.0
    activation_l2_weight: float = 0.0
    require_success: bool = True


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


def require_path(path: Path) -> None:
    if path.exists():
        return
    msg = f"missing input: {path}"
    raise FileNotFoundError(msg)


def require_array(obj: pv.DataSet, association: str, name: str) -> np.ndarray:
    data = obj.cell_data if association == "cell" else obj.point_data
    if name not in data:
        msg = f"{association}_data[{name!r}] is missing"
        raise KeyError(msg)
    return np.asarray(data[name])


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


def build_forward(mesh: pv.UnstructuredGrid, cfg: Config):
    from liblaf.apple.forward import Forward, ModelBuilder
    from liblaf.apple.warp.fem import StableNeoHookean, StableNeoHookeanActive

    builder = ModelBuilder()
    builder.add_vertices(mesh)
    builder.add_fixed(mesh)

    set_material(mesh, E=cfg.E, nu=cfg.nu, fraction=mesh.cell_data[BACKGROUND_FRACTION])
    builder.add_potential(StableNeoHookean.from_pyvista(mesh, name="background"))

    set_material(
        mesh,
        E=cfg.smas_stiffness_ratio * cfg.E,
        nu=cfg.nu,
        fraction=mesh.cell_data[ACTIVE_FRACTION],
    )
    builder.add_potential(StableNeoHookeanActive.from_pyvista(mesh, name="muscle"))

    if cfg.use_smas:
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


def active_cell_ids(mesh: pv.UnstructuredGrid) -> np.ndarray:
    active = require_array(mesh, "cell", "ActivationMask").astype(bool)
    ids = np.flatnonzero(active).astype(np.int64)
    if ids.size == 0:
        msg = "ActivationMask selected no active tetrahedra"
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
    ids = np.flatnonzero(mask).astype(np.int64)
    if ids.size == 0:
        msg = "target point mask selected no points"
        raise ValueError(msg)
    return ids


def output_point_ids(mesh: pv.UnstructuredGrid) -> np.ndarray:
    from liblaf.apple.common import GLOBAL_POINT_ID

    if GLOBAL_POINT_ID.vtk in mesh.point_data:
        return np.asarray(mesh.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64)
    return np.arange(mesh.n_points, dtype=np.int64)


def tensor_from_symmetric(values: torch.Tensor) -> torch.Tensor:
    xx, yy, zz, xy, xz, yz = values.unbind()
    return torch.stack(
        (
            torch.stack((xx, xy, xz)),
            torch.stack((xy, yy, yz)),
            torch.stack((xz, yz, zz)),
        )
    )


def matrix_from_symmetric(values: np.ndarray) -> np.ndarray:
    xx, yy, zz, xy, xz, yz = values
    return np.asarray(
        ((xx, xy, xz), (xy, yy, yz), (xz, yz, zz)),
        dtype=np.float64,
    )


def pack_symmetric_t(matrices: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (
            matrices[..., 0, 0],
            matrices[..., 1, 1],
            matrices[..., 2, 2],
            matrices[..., 0, 1],
            matrices[..., 0, 2],
            matrices[..., 1, 2],
        ),
        dim=-1,
    )


def pack_symmetric_np(matrices: np.ndarray) -> np.ndarray:
    packed = np.empty((*matrices.shape[:-2], 6), dtype=np.float64)
    packed[..., 0] = matrices[..., 0, 0]
    packed[..., 1] = matrices[..., 1, 1]
    packed[..., 2] = matrices[..., 2, 2]
    packed[..., 3] = matrices[..., 0, 1]
    packed[..., 4] = matrices[..., 0, 2]
    packed[..., 5] = matrices[..., 1, 2]
    return packed


def local_deltas_from_activation_inv(
    local_activation_inv_delta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    eye = np.eye(3, dtype=np.float64)
    local_activation_inv = eye + matrix_from_symmetric(local_activation_inv_delta)
    local_activation = np.linalg.inv(local_activation_inv)
    local_activation_delta = pack_symmetric_np(local_activation[None, ...] - eye)[0]
    return local_activation_delta, np.asarray(local_activation_inv_delta, dtype=np.float64)


def activation_inv_from_local_delta(
    local_activation_inv_delta: torch.Tensor,
    active_orientation: torch.Tensor,
    active_ids_t: torch.Tensor,
    n_cells: int,
) -> torch.Tensor:
    eye = torch.eye(3, dtype=local_activation_inv_delta.dtype)
    local_activation_inv = eye + tensor_from_symmetric(local_activation_inv_delta)
    active_activation_inv = (
        active_orientation.transpose(1, 2)
        @ local_activation_inv.expand(active_orientation.shape[0], 3, 3)
        @ active_orientation
    )
    active_activation_inv_delta = pack_symmetric_t(active_activation_inv - eye)
    full = torch.zeros(
        (n_cells, 6),
        dtype=local_activation_inv_delta.dtype,
        device=local_activation_inv_delta.device,
    )
    return full.index_copy(0, active_ids_t, active_activation_inv_delta)


def full_activation_fields_from_local(
    mesh: pv.UnstructuredGrid,
    active_ids: np.ndarray,
    local_activation_inv_delta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    orientation = require_array(mesh, "cell", "MuscleOrientation").astype(np.float64)
    orientation = orientation.reshape(mesh.n_cells, 3, 3)
    eye = np.eye(3, dtype=np.float64)
    local_activation_inv = eye + matrix_from_symmetric(local_activation_inv_delta)
    local_activation = np.linalg.inv(local_activation_inv)
    R = orientation[active_ids]
    active_inv = np.einsum("aji,jk,akl->ail", R, local_activation_inv, R)
    active_activation = np.einsum("aji,jk,akl->ail", R, local_activation, R)

    activation_inv = np.zeros((mesh.n_cells, 6), dtype=np.float64)
    activation = np.zeros((mesh.n_cells, 6), dtype=np.float64)
    activation_inv[active_ids] = pack_symmetric_np(active_inv - eye)
    activation[active_ids] = pack_symmetric_np(active_activation - eye)
    local_activation_delta = pack_symmetric_np(local_activation[None, ...] - eye)[0]
    return activation, activation_inv, local_activation_delta


def material_tree(
    base_materials: dict[str, dict[str, torch.Tensor]],
    activation_inv: torch.Tensor,
) -> dict[str, dict[str, torch.Tensor]]:
    materials = {name: dict(values) for name, values in base_materials.items()}
    materials["muscle"]["activation_inv"] = activation_inv
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


def target_local_activation_inv_stats(
    mesh: pv.UnstructuredGrid,
    target: pv.UnstructuredGrid,
    active_ids: np.ndarray,
) -> dict[str, Any]:
    from liblaf.apple.common import ACTIVATION_INV

    if ACTIVATION_INV.vtk not in target.cell_data:
        return {}
    orientation = require_array(mesh, "cell", "MuscleOrientation").astype(np.float64)
    orientation = orientation.reshape(mesh.n_cells, 3, 3)
    target_activation_inv_delta = np.asarray(
        target.cell_data[ACTIVATION_INV.vtk], dtype=np.float64
    )[active_ids]
    eye = np.eye(3, dtype=np.float64)
    world_delta = target_activation_inv_delta
    world = np.zeros((active_ids.size, 3, 3), dtype=np.float64)
    world[:, 0, 0] = 1.0 + world_delta[:, 0]
    world[:, 1, 1] = 1.0 + world_delta[:, 1]
    world[:, 2, 2] = 1.0 + world_delta[:, 2]
    world[:, 0, 1] = world[:, 1, 0] = world_delta[:, 3]
    world[:, 0, 2] = world[:, 2, 0] = world_delta[:, 4]
    world[:, 1, 2] = world[:, 2, 1] = world_delta[:, 5]
    R = orientation[active_ids]
    local = np.einsum("aij,ajk,akl->ail", R, world, np.swapaxes(R, 1, 2))
    local_delta = pack_symmetric_np(local - eye)
    return {
        "target_local_activation_inv_delta/mean": local_delta.mean(axis=0).tolist(),
        "target_local_activation_inv_delta/min": local_delta.min(axis=0).tolist(),
        "target_local_activation_inv_delta/max": local_delta.max(axis=0).tolist(),
        "target_activation_inv/rms": float(
            np.linalg.norm(target_activation_inv_delta)
            / math.sqrt(target_activation_inv_delta.size)
        ),
        "target_activation_inv/max_norm": float(
            np.linalg.norm(target_activation_inv_delta, axis=1).max()
        ),
    }


def save_checkpoint(
    path: Path,
    *,
    local_activation_inv_delta: np.ndarray,
    activation_inv: np.ndarray,
    displacement: np.ndarray,
    step: int,
    loss: float,
    max_error: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    with tmp.open("wb") as file:
        np.savez(
            file,
            local_activation_inv_delta=np.asarray(
                local_activation_inv_delta, dtype=np.float64
            ),
            activation_inv=np.asarray(activation_inv, dtype=np.float64),
            displacement=np.asarray(displacement, dtype=np.float64),
            step=np.asarray(step, dtype=np.int64),
            loss=np.asarray(loss, dtype=np.float64),
            max_error=np.asarray(max_error, dtype=np.float64),
        )
    tmp.replace(path)


def numeric_metrics(
    data: dict[str, Any], *, exclude: frozenset[str] = frozenset()
) -> dict[str, float | int | bool]:
    return {
        name: value
        for name, value in data.items()
        if name not in exclude and isinstance(value, bool | int | float)
    }


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
    target: pv.UnstructuredGrid,
    displacement: np.ndarray,
    activation: np.ndarray,
    activation_inv: np.ndarray,
    local_activation_delta: np.ndarray,
    local_activation_inv_delta: np.ndarray,
    target_ids: np.ndarray,
    active_ids: np.ndarray,
    metrics: dict[str, float | int | bool | str],
) -> pv.UnstructuredGrid:
    from liblaf.apple.common import ACTIVATION, ACTIVATION_INV

    result = mesh.copy(deep=True)
    add_masks(result, target_ids, active_ids)
    target_displacement = np.asarray(target.point_data["Displacement"], dtype=np.float64)
    error = displacement - target_displacement
    result.point_data["Displacement"] = displacement
    result.point_data["DisplacementNorm"] = np.linalg.norm(displacement, axis=1)
    result.point_data["TargetDisplacement"] = target_displacement
    result.point_data["TargetDisplacementNorm"] = np.linalg.norm(
        target_displacement, axis=1
    )
    result.point_data["DisplacementError"] = error
    result.point_data["DisplacementErrorNorm"] = np.linalg.norm(error, axis=1)
    result.point_data["DeformedPoint"] = result.points + displacement
    result.point_data["TargetPoint"] = result.points + target_displacement
    result.cell_data[ACTIVATION.vtk] = activation
    result.cell_data[ACTIVATION_INV.vtk] = activation_inv
    result.cell_data["RecoveredActivation"] = activation
    result.cell_data["RecoveredActivationInv"] = activation_inv
    result.cell_data["RecoveredActivationInvNorm"] = np.linalg.norm(
        activation_inv, axis=1
    )
    if ACTIVATION_INV.vtk in target.cell_data:
        target_activation_inv = np.asarray(
            target.cell_data[ACTIVATION_INV.vtk], dtype=np.float64
        )
        result.cell_data["TargetActivationInv"] = target_activation_inv
        result.cell_data["ActivationInvError"] = activation_inv - target_activation_inv
        result.cell_data["ActivationInvErrorNorm"] = np.linalg.norm(
            activation_inv - target_activation_inv, axis=1
        )
    result.field_data["LocalActivationDelta"] = np.asarray(local_activation_delta)
    result.field_data["LocalActivationInvDelta"] = np.asarray(
        local_activation_inv_delta
    )
    add_metric_fields(result, metrics)
    return result


def save_snapshot(path: Path, result: pv.UnstructuredGrid) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    view = result.copy(deep=True)
    view.points = np.asarray(view.point_data["DeformedPoint"], dtype=np.float64)

    plotter = pv.Plotter(off_screen=True, shape=(1, 2), window_size=(1800, 900))
    plotter.subplot(0, 0)
    plotter.add_mesh(
        view.extract_surface(),
        scalars="DisplacementErrorNorm",
        cmap="viridis",
        show_edges=False,
    )
    plotter.add_text("surface error (cm)", font_size=12)
    plotter.view_xy()
    plotter.camera.zoom(1.25)

    plotter.subplot(0, 1)
    clipped = view.clip(normal="z", origin=view.center)
    plotter.add_mesh(
        clipped,
        scalars="RecoveredActivationInvNorm",
        cmap="magma",
        show_edges=False,
    )
    plotter.add_text("activation_inv norm", font_size=12)
    plotter.view_xy()
    plotter.camera.zoom(1.25)

    plotter.screenshot(path)
    plotter.close()


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def solve_inverse(  # noqa: C901, PLR0915
    mesh: pv.UnstructuredGrid,
    target: pv.UnstructuredGrid,
    target_ids: np.ndarray,
    active_ids: np.ndarray,
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
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
    output_ids = output_point_ids(mesh)
    output_ids_t = torch.as_tensor(
        output_ids[target_ids],
        dtype=torch.long,
        device=torch.get_default_device(),
    )
    target_ids_t = torch.as_tensor(
        target_ids,
        dtype=torch.long,
        device=torch.get_default_device(),
    )
    target_displacement = np.asarray(
        target.point_data["Displacement"], dtype=np.float64
    )
    target_t = torch.as_tensor(
        target_displacement,
        dtype=torch.get_default_dtype(),
        device=torch.get_default_device(),
    )
    active_ids_t = torch.as_tensor(
        active_ids,
        dtype=torch.long,
        device=torch.get_default_device(),
    )
    orientation = require_array(mesh, "cell", "MuscleOrientation").astype(np.float64)
    orientation = orientation.reshape(mesh.n_cells, 3, 3)[active_ids]
    active_orientation = torch.as_tensor(
        orientation,
        dtype=torch.get_default_dtype(),
        device=torch.get_default_device(),
    )
    local_activation_inv_delta = torch.nn.Parameter(
        torch.zeros(6, dtype=torch.get_default_dtype(), device=torch.get_default_device())
    )
    optimizer = torch.optim.Adam(
        [local_activation_inv_delta],
        lr=cfg.inverse_lr,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_eps,
    )

    trace: list[dict[str, Any]] = []
    best_step = 0
    best_loss = math.inf
    best_max_error = math.inf
    best_displacement: np.ndarray | None = None
    best_activation_inv: np.ndarray | None = None
    best_local_activation_inv_delta: np.ndarray | None = None
    stop_reason = "step_limit"
    start = time.perf_counter()
    optimizer_steps = 0
    for step in range(cfg.inverse_max_steps + 1):
        step_start = time.perf_counter()
        optimizer.zero_grad()
        full_activation_inv = activation_inv_from_local_delta(
            local_activation_inv_delta,
            active_orientation,
            active_ids_t,
            mesh.n_cells,
        )
        materials = material_tree(base_materials, full_activation_inv)

        forward_start = time.perf_counter()
        output = forward_quiet(differentiable_forward, materials)
        forward_elapsed = time.perf_counter() - forward_start
        forward_metrics = forward_solution_metrics(
            getattr(differentiable_forward, "last_forward_solution", None)
        )

        residual = output[output_ids_t] - target_t[target_ids_t]
        data_loss = residual.square().mean()
        activation_l2 = local_activation_inv_delta.square().mean()
        loss = data_loss + cfg.activation_l2_weight * activation_l2

        backward_start = time.perf_counter()
        loss.backward()
        backward_elapsed = time.perf_counter() - backward_start
        adjoint_metrics = adjoint_solution_metrics(
            getattr(differentiable_forward, "last_adjoint_solution", None)
        )
        grad = local_activation_inv_delta.grad
        if grad is None:
            msg = "differentiable forward did not produce activation gradients"
            raise RuntimeError(msg)
        if not torch.isfinite(grad).all():
            nonfinite = int((~torch.isfinite(grad)).sum().detach().cpu())
            msg = f"non-finite inverse gradient at step {step}: {nonfinite} entries"
            raise FloatingPointError(msg)

        error_stats = point_error_stats(residual.detach())
        loss_value = float(loss.detach().cpu())
        data_loss_value = float(data_loss.detach().cpu())
        activation_l2_value = float(activation_l2.detach().cpu())
        mean_error = float(error_stats["mean"].detach().cpu())
        rms_error = float(error_stats["rms"].detach().cpu())
        max_error = float(error_stats["max"].detach().cpu())
        grad_norm = float(torch.linalg.vector_norm(grad).detach().cpu())
        grad_abs_max = float(grad.abs().max().detach().cpu())
        current_local_activation_inv_delta = to_numpy(local_activation_inv_delta)
        current_local_activation_delta, current_local_activation_inv_delta = (
            local_deltas_from_activation_inv(current_local_activation_inv_delta)
        )
        current_activation_inv = to_numpy(full_activation_inv)
        displacement = to_numpy(output)[output_ids]
        improved = max_error < best_max_error or (
            math.isclose(max_error, best_max_error) and loss_value < best_loss
        )
        if improved:
            best_step = step
            best_loss = loss_value
            best_max_error = max_error
            best_displacement = displacement
            best_activation_inv = current_activation_inv
            best_local_activation_inv_delta = current_local_activation_inv_delta
            save_checkpoint(
                cfg.checkpoint,
                local_activation_inv_delta=current_local_activation_inv_delta,
                activation_inv=current_activation_inv,
                displacement=displacement,
                step=step,
                loss=loss_value,
                max_error=max_error,
            )

        stopped = False
        if (
            cfg.stop_on_max_point_error
            and step >= cfg.inverse_min_steps
            and best_max_error <= cfg.max_point_error_cm
        ):
            stop_reason = "max_point_error_tol"
            stopped = True
        elif loss_value <= cfg.loss_tol:
            stop_reason = "loss_tol"
            stopped = True

        record = {
            "step": float(step),
            "loss/total": loss_value,
            "loss/data": data_loss_value,
            "loss/activation_l2": activation_l2_value,
            "target/error_mean": mean_error,
            "target/error_rms": rms_error,
            "target/error_max": max_error,
            "best/step": float(best_step),
            "best/loss": best_loss,
            "best/target_error_max": best_max_error,
            "activation/local_xx": float(current_local_activation_delta[0]),
            "activation/local_yy": float(current_local_activation_delta[1]),
            "activation/local_zz": float(current_local_activation_delta[2]),
            "activation_inv/local_xx": float(current_local_activation_inv_delta[0]),
            "activation_inv/local_yy": float(current_local_activation_inv_delta[1]),
            "activation_inv/local_zz": float(current_local_activation_inv_delta[2]),
            "activation_inv/local_xy": float(current_local_activation_inv_delta[3]),
            "activation_inv/local_xz": float(current_local_activation_inv_delta[4]),
            "activation_inv/local_yz": float(current_local_activation_inv_delta[5]),
            "activation_inv/local_norm": float(
                np.linalg.norm(current_local_activation_inv_delta)
            ),
            "grad/norm": grad_norm,
            "grad/abs_max": grad_abs_max,
            "optimizer/steps": float(optimizer_steps),
            "optimizer/lr": float(optimizer.param_groups[0]["lr"]),
            "stopped": float(stopped),
            "time/forward_s": forward_elapsed,
            "time/backward_s": backward_elapsed,
            "time/step_s": time.perf_counter() - step_start,
            **forward_metrics,
            **adjoint_metrics,
        }
        trace.append(record)
        cherries.set_step(step)
        cherries.log_metrics(numeric_metrics(record, exclude=frozenset({"step"})))
        print(
            "inverse step:",
            f"{step:03d}",
            f"loss={loss_value:.3e}",
            f"rms={rms_error:.3e}cm",
            f"max={max_error:.3e}cm",
            f"best_max={best_max_error:.3e}cm",
            f"act=({current_local_activation_delta[0]:.3f},"
            f"{current_local_activation_delta[1]:.3f},"
            f"{current_local_activation_delta[2]:.3f},"
            f"{current_local_activation_delta[3]:.3f},"
            f"{current_local_activation_delta[4]:.3f},"
            f"{current_local_activation_delta[5]:.3f})",
            f"act_inv=({current_local_activation_inv_delta[0]:.3f},"
            f"{current_local_activation_inv_delta[1]:.3f},"
            f"{current_local_activation_inv_delta[2]:.3f},"
            f"{current_local_activation_inv_delta[3]:.3f},"
            f"{current_local_activation_inv_delta[4]:.3f},"
            f"{current_local_activation_inv_delta[5]:.3f})",
            f"grad={grad_norm:.3e}",
            f"fwd={forward_metrics['forward/result']}/"
            f"{forward_metrics['forward/steps']}",
            f"adj={adjoint_metrics['adjoint/result']}/"
            f"{adjoint_metrics['adjoint/best_solver']}",
            flush=True,
        )
        if stopped or optimizer_steps >= cfg.inverse_max_steps:
            break
        optimizer.step()
        optimizer_steps += 1

    if (
        best_displacement is None
        or best_activation_inv is None
        or best_local_activation_inv_delta is None
    ):
        msg = "inverse solve did not evaluate any forward states"
        raise RuntimeError(msg)
    final = {
        "stop_reason": stop_reason,
        "optimizer/steps": int(optimizer_steps),
        "best/step": int(best_step),
        "best/loss": float(best_loss),
        "best/target_error_max": float(best_max_error),
        "time/inverse_s": float(time.perf_counter() - start),
    }
    return (
        best_displacement,
        best_activation_inv,
        best_local_activation_inv_delta,
        trace,
        final,
    )


def summarize(
    mesh: pv.UnstructuredGrid,
    target: pv.UnstructuredGrid,
    displacement: np.ndarray,
    activation_inv: np.ndarray,
    local_activation_delta: np.ndarray,
    local_activation_inv_delta: np.ndarray,
    target_ids: np.ndarray,
    active_ids: np.ndarray,
    trace: list[dict[str, Any]],
    final: dict[str, Any],
    total_elapsed_s: float,
    cfg: Config,
) -> dict[str, Any]:
    from liblaf.apple.common import ACTIVATION_INV

    target_displacement = np.asarray(target.point_data["Displacement"], dtype=np.float64)
    error = displacement - target_displacement
    target_error = error[target_ids]
    target_error_norm = np.linalg.norm(target_error, axis=1)
    target_norm = np.linalg.norm(target_displacement[target_ids], axis=1)
    active_activation_inv = activation_inv[active_ids]
    metrics: dict[str, Any] = {
        "mesh/n_points": int(mesh.n_points),
        "mesh/n_cells": int(mesh.n_cells),
        "target/source": str(cfg.target),
        "target/n_points": int(target_ids.size),
        "target/displacement_mean": float(target_norm.mean()),
        "target/displacement_rms": float(
            np.linalg.norm(target_displacement[target_ids]) / math.sqrt(target_ids.size)
        ),
        "target/displacement_max": float(target_norm.max()),
        "target/error_mean": float(target_error_norm.mean()),
        "target/error_rms": float(
            np.linalg.norm(target_error) / math.sqrt(target_ids.size)
        ),
        "target/error_max": float(target_error_norm.max()),
        "all/error_rms": float(np.linalg.norm(error) / math.sqrt(error.shape[0])),
        "all/error_max": float(np.linalg.norm(error, axis=1).max()),
        "tolerance/max_point_error_cm": float(cfg.max_point_error_cm),
        "activation/parameterization": "single full local ActivationInv delta, 6 DoF",
        "activation/n_active_tets": int(active_ids.size),
        "activation/n_params": 6,
        "active_activation_inv/rms": float(
            np.linalg.norm(active_activation_inv)
            / math.sqrt(active_activation_inv.size)
        ),
        "active_activation_inv/max_norm": float(
            np.linalg.norm(active_activation_inv, axis=1).max()
        ),
        "local_activation_delta": local_activation_delta.tolist(),
        "local_activation_inv_delta": local_activation_inv_delta.tolist(),
        "local_activation_delta/xx": float(local_activation_delta[0]),
        "local_activation_delta/yy": float(local_activation_delta[1]),
        "local_activation_delta/zz": float(local_activation_delta[2]),
        "local_activation_delta/xy": float(local_activation_delta[3]),
        "local_activation_delta/xz": float(local_activation_delta[4]),
        "local_activation_delta/yz": float(local_activation_delta[5]),
        "local_activation_inv_delta/xx": float(local_activation_inv_delta[0]),
        "local_activation_inv_delta/yy": float(local_activation_inv_delta[1]),
        "local_activation_inv_delta/zz": float(local_activation_inv_delta[2]),
        "local_activation_inv_delta/xy": float(local_activation_inv_delta[3]),
        "local_activation_inv_delta/xz": float(local_activation_inv_delta[4]),
        "local_activation_inv_delta/yz": float(local_activation_inv_delta[5]),
        "E": float(cfg.E),
        "nu": float(cfg.nu),
        "smas/enabled": bool(cfg.use_smas),
        "time/total_s": float(total_elapsed_s),
        "trace": trace,
        **final,
        **target_local_activation_inv_stats(mesh, target, active_ids),
    }
    if ACTIVATION_INV.vtk in target.cell_data:
        target_activation_inv = np.asarray(
            target.cell_data[ACTIVATION_INV.vtk], dtype=np.float64
        )
        active_error = activation_inv[active_ids] - target_activation_inv[active_ids]
        metrics["activation_inv/error_rms"] = float(
            np.linalg.norm(active_error) / math.sqrt(active_error.size)
        )
        metrics["activation_inv/error_max_norm"] = float(
            np.linalg.norm(active_error, axis=1).max()
        )
    metrics["passed"] = bool(
        metrics["target/error_max"] <= cfg.max_point_error_cm
        and np.isfinite(metrics["target/error_max"])
    )
    return metrics


def main(cfg: Config) -> None:
    total_start = time.perf_counter()
    configure_runtime()
    mesh, target = load_problem(cfg)
    target_ids = target_point_ids(target, cfg)
    active_ids = active_cell_ids(mesh)
    add_masks(mesh, target_ids, active_ids)
    melon.save(cfg.output_input, mesh)
    melon.save(cfg.output_target, make_target_mesh(target, target_ids, active_ids))

    (
        displacement,
        _activation_inv,
        recovered_local_activation_inv_delta,
        trace,
        final,
    ) = solve_inverse(mesh.copy(deep=True), target, target_ids, active_ids, cfg)
    _, local_activation_inv_delta = local_deltas_from_activation_inv(
        recovered_local_activation_inv_delta
    )
    activation, activation_inv, local_activation_delta = full_activation_fields_from_local(
        mesh,
        active_ids,
        local_activation_inv_delta,
    )
    total_elapsed_s = time.perf_counter() - total_start
    summary = summarize(
        mesh,
        target,
        displacement,
        activation_inv,
        local_activation_delta,
        local_activation_inv_delta,
        target_ids,
        active_ids,
        trace,
        final,
        total_elapsed_s,
        cfg,
    )
    result = make_result_mesh(
        mesh,
        target,
        displacement,
        activation,
        activation_inv,
        local_activation_delta,
        local_activation_inv_delta,
        target_ids,
        active_ids,
        numeric_metrics(summary, exclude=frozenset({"trace"})),
    )
    melon.save(cfg.output, result)
    try:
        save_snapshot(cfg.output_snapshot, result)
    except (OSError, RuntimeError, ValueError):
        logger.warning("failed to save snapshot: %s", cfg.output_snapshot, exc_info=True)
    save_json(cfg.output_summary, summary)
    cherries.log_metrics(numeric_metrics(summary, exclude=frozenset({"trace"})))
    print(
        "inverse result:",
        f"passed={summary['passed']}",
        f"stop={summary['stop_reason']}",
        f"best_step={summary['best/step']}",
        f"target_rms_error={summary['target/error_rms']:.3e}cm",
        f"target_max_error={summary['target/error_max']:.3e}cm",
        f"local_activation=({summary['local_activation_delta/xx']:.3f},"
        f"{summary['local_activation_delta/yy']:.3f},"
        f"{summary['local_activation_delta/zz']:.3f},"
        f"{summary['local_activation_delta/xy']:.3f},"
        f"{summary['local_activation_delta/xz']:.3f},"
        f"{summary['local_activation_delta/yz']:.3f})",
    )
    print(f"saved: {cfg.output}")
    print(f"saved: {cfg.output_summary}")
    if cfg.require_success and not summary["passed"]:
        msg = (
            "inverse solve did not meet max point error: "
            f"{summary['target/error_max']:.6g} cm > "
            f"{cfg.max_point_error_cm:.6g} cm"
        )
        raise RuntimeError(msg)


if __name__ == "__main__":
    cherries.main(main)
