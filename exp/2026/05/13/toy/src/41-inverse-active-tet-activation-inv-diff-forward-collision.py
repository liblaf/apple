import json
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

OUTPUT_STEM = "41-inverse-active-tet-activation-inv-diff-forward-collision"
TARGET_STEM = "20-activated-boxes-collision"


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    target_input: Path = cherries.output(f"{TARGET_STEM}-input.vtu")
    target_output: Path = cherries.output(f"{TARGET_STEM}-collision.vtu")
    output_input: Path = cherries.output(f"{OUTPUT_STEM}-input.vtu")
    output_target: Path = cherries.output(f"{OUTPUT_STEM}-target.vtu")
    output: Path = cherries.output(f"{OUTPUT_STEM}.vtu")
    output_series: Path = cherries.output(f"{OUTPUT_STEM}-steps.vtu.series")
    output_summary: Path = cherries.output(f"{OUTPUT_STEM}-summary.json")

    forward_rtol: float = 1.0e-3
    forward_atol: float = 1.0e-3
    forward_max_steps: int = 5000
    inverse_max_steps: int = 120
    inverse_lr: float = 0.5
    activation_inv_residual_scale: float = 0.05
    adam_beta1: float = 0.0
    adam_beta2: float = 0.9
    adam_eps: float = 1.0e-8
    activation_inv_xz_abs_max: float = 0.02
    activation_inv_y_min: float = -0.8
    activation_inv_y_max: float = 0.0
    activation_inv_shear_abs_max: float = 0.02
    active_fraction_tol: float = 1.0e-6
    point_error_rel_tol: float = 1.0e-3


def configure_runtime() -> None:
    if not torch.cuda.is_available():
        msg = "This experiment uses Warp kernels through Torch and needs CUDA."
        raise RuntimeError(msg)
    warnings.filterwarnings(
        "ignore",
        message=r"The \.grad attribute of a Tensor that is not a leaf Tensor.*",
        category=UserWarning,
    )
    torch.set_default_dtype(torch.float64)
    torch.set_default_device("cuda")
    wp.config.mode = "release"
    wp.init()


def to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def require_path(path: Path) -> None:
    if path.exists():
        return
    msg = f"missing target file: {path}. Run {TARGET_STEM}.py first."
    raise FileNotFoundError(msg)


def load_target(cfg: Config) -> tuple[pv.UnstructuredGrid, pv.UnstructuredGrid]:
    require_path(cfg.target_input)
    require_path(cfg.target_output)
    input_mesh = pv.read(cfg.target_input)
    target_mesh = pv.read(cfg.target_output)
    if input_mesh.n_points != target_mesh.n_points:
        msg = (
            "target input/output point counts differ: "
            f"{input_mesh.n_points} != {target_mesh.n_points}"
        )
        raise ValueError(msg)
    if not np.allclose(input_mesh.points, target_mesh.points):
        msg = "target input/output rest points differ"
        raise ValueError(msg)
    if "Displacement" not in target_mesh.point_data:
        msg = f"{cfg.target_output} has no point_data['Displacement']"
        raise KeyError(msg)
    return input_mesh, target_mesh


def surface_point_ids(mesh: pv.UnstructuredGrid) -> np.ndarray:
    surface = mesh.extract_surface(algorithm=None)
    point_ids = np.asarray(surface.point_data["vtkOriginalPointIds"], dtype=np.int64)
    return np.unique(point_ids)


def zero_activation(mesh: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    from liblaf.apple.common import ACTIVATION, ACTIVATION_INV

    result = mesh.copy(deep=True)
    activation = np.zeros((result.n_cells, 6), dtype=np.float64)
    result.cell_data[ACTIVATION.vtk] = activation
    if ACTIVATION_INV.vtk in result.cell_data:
        result.cell_data[ACTIVATION_INV.vtk] = activation.copy()
    result.field_data["Activation"] = np.zeros((1, 6), dtype=np.float64)
    result.field_data["ActivationInv"] = np.zeros((1, 6), dtype=np.float64)
    result.field_data["InitialActivation"] = np.zeros((1, 6), dtype=np.float64)
    result.field_data["TargetSource"] = np.asarray([TARGET_STEM])
    return result


def target_activation_field(target_mesh: pv.UnstructuredGrid) -> np.ndarray:
    from liblaf.apple.common import ACTIVATION

    if ACTIVATION.vtk in target_mesh.cell_data:
        return np.asarray(target_mesh.cell_data[ACTIVATION.vtk], dtype=np.float64)
    if "Activation" in target_mesh.field_data:
        activation = np.asarray(target_mesh.field_data["Activation"]).reshape(-1)[:6]
        return np.repeat(activation[np.newaxis, :], target_mesh.n_cells, axis=0)
    msg = "target mesh has no cell activation field"
    raise KeyError(msg)


def active_cell_ids(mesh: pv.UnstructuredGrid, cfg: Config) -> np.ndarray:
    if "ActivationMask" in mesh.cell_data:
        active = np.asarray(mesh.cell_data["ActivationMask"], dtype=bool)
    else:
        active_fraction = np.asarray(mesh.cell_data["ActiveFraction"], dtype=np.float64)
        active = active_fraction > cfg.active_fraction_tol
    ids = np.flatnonzero(active).astype(np.int64)
    if ids.size == 0:
        msg = "target has no active tets"
        raise ValueError(msg)
    return ids


def activation_matrices_torch(activation: torch.Tensor) -> torch.Tensor:
    matrices = torch.zeros(
        (*activation.shape[:-1], 3, 3),
        dtype=activation.dtype,
        device=activation.device,
    )
    matrices[..., 0, 0] = 1.0 + activation[..., 0]
    matrices[..., 1, 1] = 1.0 + activation[..., 1]
    matrices[..., 2, 2] = 1.0 + activation[..., 2]
    matrices[..., 0, 1] = activation[..., 3]
    matrices[..., 1, 0] = activation[..., 3]
    matrices[..., 0, 2] = activation[..., 4]
    matrices[..., 2, 0] = activation[..., 4]
    matrices[..., 1, 2] = activation[..., 5]
    matrices[..., 2, 1] = activation[..., 5]
    return matrices


def pack_activation_matrices_torch(matrices: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (
            matrices[..., 0, 0] - 1.0,
            matrices[..., 1, 1] - 1.0,
            matrices[..., 2, 2] - 1.0,
            matrices[..., 0, 1],
            matrices[..., 0, 2],
            matrices[..., 1, 2],
        ),
        dim=-1,
    )


def activation_to_activation_inv_torch(activation: torch.Tensor) -> torch.Tensor:
    matrices = activation_matrices_torch(activation)
    return pack_activation_matrices_torch(torch.linalg.inv(matrices))


def activation_inv_to_activation_torch(activation_inv: torch.Tensor) -> torch.Tensor:
    matrices = activation_matrices_torch(activation_inv)
    return pack_activation_matrices_torch(torch.linalg.inv(matrices))


def activation_to_activation_inv_numpy(activation: np.ndarray) -> np.ndarray:
    activation_t = torch.as_tensor(activation, dtype=torch.float64, device="cpu")
    return activation_to_activation_inv_torch(activation_t).numpy()


def activation_inv_to_activation_numpy(activation_inv: np.ndarray) -> np.ndarray:
    activation_inv_t = torch.as_tensor(activation_inv, dtype=torch.float64, device="cpu")
    return activation_inv_to_activation_torch(activation_inv_t).numpy()


def full_activation_from_active(
    active_activation: torch.Tensor,
    active_ids_t: torch.Tensor,
    n_cells: int,
) -> torch.Tensor:
    full_activation = torch.zeros(
        (n_cells, 6),
        dtype=active_activation.dtype,
        device=active_activation.device,
    )
    return full_activation.index_copy(0, active_ids_t, active_activation)


def full_activation_from_active_inv(
    active_activation_inv: torch.Tensor,
    active_ids_t: torch.Tensor,
    n_cells: int,
) -> torch.Tensor:
    active_activation = activation_inv_to_activation_torch(active_activation_inv)
    return full_activation_from_active(active_activation, active_ids_t, n_cells)


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


def precondition_active_activation_inv(
    raw_active_activation_inv: torch.Tensor, cfg: Config
) -> torch.Tensor:
    active_mean = raw_active_activation_inv.mean(dim=0, keepdim=True)
    active_residual = raw_active_activation_inv - active_mean
    return active_mean + cfg.activation_inv_residual_scale * active_residual


def clamp_active_activation_inv_(
    active_activation_inv: torch.Tensor, cfg: Config
) -> None:
    active_activation_inv[:, 0].clamp_(
        -cfg.activation_inv_xz_abs_max, cfg.activation_inv_xz_abs_max
    )
    active_activation_inv[:, 2].clamp_(
        -cfg.activation_inv_xz_abs_max, cfg.activation_inv_xz_abs_max
    )
    active_activation_inv[:, 1].clamp_(
        cfg.activation_inv_y_min, cfg.activation_inv_y_max
    )
    active_activation_inv[:, 3:].clamp_(
        -cfg.activation_inv_shear_abs_max, cfg.activation_inv_shear_abs_max
    )


def set_fraction(mesh: pv.UnstructuredGrid, name: str) -> None:
    from liblaf.apple.common import FRACTION

    mesh.cell_data[FRACTION.vtk] = np.asarray(mesh.cell_data[name])


def build_forward(mesh: pv.UnstructuredGrid, cfg: Config):
    from liblaf.apple.collision import CollisionBuilder
    from liblaf.apple.forward import Forward, ModelBuilder
    from liblaf.apple.warp.fem import StableNeoHookean, StableNeoHookeanActive

    collision_stiffness = float(np.asarray(mesh.field_data["CollisionStiffness"])[0])
    builder = ModelBuilder()
    builder.add_vertices(mesh)
    builder.add_fixed(mesh)

    set_fraction(mesh, "PassiveFraction")
    builder.add_potential(StableNeoHookean.from_pyvista(mesh, name="passive"))

    set_fraction(mesh, "ActiveFraction")
    builder.add_potential(StableNeoHookeanActive.from_pyvista(mesh, name="active"))

    collision_builder = CollisionBuilder(stiffness=collision_stiffness)
    collision_builder.add_tetmesh(mesh)
    builder.collision = collision_builder

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
    materials["active"]["activation_inv"] = full_activation_inv_from_active(
        active_activation_inv, active_ids_t, n_cells
    )
    return materials


def point_error_stats(residual: torch.Tensor) -> dict[str, torch.Tensor]:
    point_error = torch.linalg.vector_norm(residual, dim=1)
    return {
        "mean": point_error.mean(),
        "rms": torch.linalg.vector_norm(residual) / math.sqrt(residual.shape[0]),
        "max": point_error.max(),
    }


def inverse_tensors(
    mesh: pv.UnstructuredGrid,
    target_displacement: np.ndarray,
    surface_ids: np.ndarray,
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor, torch.Tensor]:
    from liblaf.apple.common import GLOBAL_POINT_ID

    global_ids = np.asarray(mesh.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64)
    surface_global_ids = global_ids[surface_ids]
    target = torch.as_tensor(
        target_displacement,
        dtype=torch.get_default_dtype(),
        device=torch.get_default_device(),
    )
    surface_ids_t = torch.as_tensor(
        surface_ids,
        dtype=torch.long,
        device=torch.get_default_device(),
    )
    surface_global_ids_t = torch.as_tensor(
        surface_global_ids,
        dtype=torch.long,
        device=torch.get_default_device(),
    )
    return global_ids, target, surface_ids_t, surface_global_ids_t


def solve_inverse(  # noqa: PLR0915
    mesh: pv.UnstructuredGrid,
    target_activation: np.ndarray,
    target_displacement: np.ndarray,
    surface_ids: np.ndarray,
    active_ids: np.ndarray,
    point_error_tol: float,
    cfg: Config,
    series_writer: Any,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]], str, int, dict[str, float]]:
    from liblaf.apple.inverse import DifferentiableForward

    forward = build_forward(mesh, cfg)
    differentiable_forward = DifferentiableForward(forward)
    base_materials = forward.model.get_materials()
    global_ids, target, surface_ids_t, surface_global_ids_t = inverse_tensors(
        mesh, target_displacement, surface_ids
    )
    active_ids_t = torch.as_tensor(
        active_ids,
        dtype=torch.long,
        device=torch.get_default_device(),
    )
    raw_active_activation_inv = torch.nn.Parameter(
        torch.zeros(
            (active_ids.size, 6),
            dtype=torch.get_default_dtype(),
            device=torch.get_default_device(),
        )
    )
    optimizer = torch.optim.Adam(
        [raw_active_activation_inv],
        lr=cfg.inverse_lr,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_eps,
    )

    trace: list[dict[str, float]] = []
    stop_reason = "inverse_max_steps"
    optimizer_steps = 0
    best_displacement: np.ndarray | None = None
    best_activation_inv: np.ndarray | None = None
    best_surface_mean_error = math.inf
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
        active_activation_inv = precondition_active_activation_inv(
            raw_active_activation_inv,
            cfg,
        )
        materials = material_tree(
            base_materials, active_activation_inv, active_ids_t, mesh.n_cells
        )
        with torch.no_grad():
            active_inv_values = active_activation_inv.detach()
            activation_inv_y_mean = float(active_inv_values[:, 1].mean().cpu())
            activation_inv_min = float(active_inv_values.min().cpu())
            activation_inv_max = float(active_inv_values.max().cpu())
            activation_inv_rms = float(
                torch.linalg.vector_norm(active_inv_values).cpu()
                / math.sqrt(active_inv_values.numel())
            )
        print(
            "inverse forward:",
            f"step={step:03d}",
            f"active_activation_inv_y_mean={activation_inv_y_mean:.8f}",
            f"activation_inv_range=[{activation_inv_min:.3e}, {activation_inv_max:.3e}]",
            flush=True,
        )
        forward_start = time.perf_counter()
        output = differentiable_forward.forward(materials)
        forward_elapsed = time.perf_counter() - forward_start
        timing["forward_elapsed_s"] += forward_elapsed
        print("inverse backward:", f"step={step:03d}", flush=True)
        backward_start = time.perf_counter()
        residual = output[surface_global_ids_t] - target[surface_ids_t]
        loss = residual.square().mean()
        loss.backward()
        backward_elapsed = time.perf_counter() - backward_start
        timing["backward_elapsed_s"] += backward_elapsed
        if raw_active_activation_inv.grad is None:
            msg = "differentiable forward did not produce activation_inv gradients"
            raise RuntimeError(msg)
        grad = raw_active_activation_inv.grad.detach()
        if not torch.isfinite(grad).all():
            nonfinite = int((~torch.isfinite(grad)).sum().detach().cpu())
            msg = f"non-finite inverse gradient at step {step}: {nonfinite} entries"
            raise FloatingPointError(msg)

        error_stats = point_error_stats(residual.detach())
        grad_norm = float(torch.linalg.vector_norm(grad).cpu())
        grad_abs_max = float(grad.abs().max().cpu())
        loss_value = float(loss.detach().cpu())
        surface_mean_error = float(error_stats["mean"].cpu())
        displacement = to_numpy(output)[global_ids]
        if surface_mean_error < best_surface_mean_error:
            best_surface_mean_error = surface_mean_error
            best_activation_inv = to_numpy(
                full_activation_inv_from_active(
                    active_activation_inv, active_ids_t, mesh.n_cells
                )
            )
            best_displacement = displacement
        stopped = surface_mean_error <= point_error_tol
        did_optimizer_step = False
        if stopped:
            stop_reason = "point_error_tol"
        elif optimizer_steps < cfg.inverse_max_steps:
            optimizer_start = time.perf_counter()
            optimizer.step()
            optimizer_steps += 1
            did_optimizer_step = True
            timing["optimizer_elapsed_s"] += time.perf_counter() - optimizer_start
            with torch.no_grad():
                clamp_active_activation_inv_(raw_active_activation_inv, cfg)
        next_active_inv = precondition_active_activation_inv(
            raw_active_activation_inv,
            cfg,
        ).detach()
        next_activation_inv_y_mean = float(next_active_inv[:, 1].mean().cpu())
        next_activation_inv_min = float(next_active_inv.min().cpu())
        next_activation_inv_max = float(next_active_inv.max().cpu())
        trace.append(
            {
                "step": float(step),
                "loss": loss_value,
                "surface_mean_error": surface_mean_error,
                "surface_rms_error": float(error_stats["rms"].cpu()),
                "surface_max_error": float(error_stats["max"].cpu()),
                "activation_inv_y_mean": activation_inv_y_mean,
                "activation_inv_min": activation_inv_min,
                "activation_inv_max": activation_inv_max,
                "activation_inv_rms": activation_inv_rms,
                "next_activation_inv_y_mean": next_activation_inv_y_mean,
                "next_activation_inv_min": next_activation_inv_min,
                "next_activation_inv_max": next_activation_inv_max,
                "grad_norm": grad_norm,
                "grad_abs_max": grad_abs_max,
                "optimizer_steps": float(optimizer_steps),
                "stopped": float(stopped),
                "best_surface_mean_error": best_surface_mean_error,
                "forward_elapsed_s": forward_elapsed,
                "backward_elapsed_s": backward_elapsed,
            }
        )
        evaluated_activation_inv = to_numpy(
            full_activation_inv_from_active(
                active_inv_values, active_ids_t, mesh.n_cells
            )
        )
        evaluated_activation = activation_inv_to_activation_numpy(
            evaluated_activation_inv
        )
        series_start = time.perf_counter()
        step_mesh = make_result_mesh(
            mesh,
            target_displacement,
            displacement,
            target_activation,
            evaluated_activation,
            evaluated_activation_inv,
            surface_ids,
            active_ids,
            {
                "inverse_step": step,
                "optimizer_steps": optimizer_steps,
                "loss": loss_value,
                "surface_mean_error": surface_mean_error,
                "surface_rms_error": float(error_stats["rms"].cpu()),
                "surface_max_error": float(error_stats["max"].cpu()),
                "point_error_tol": point_error_tol,
                "activation_inv_y_mean": activation_inv_y_mean,
                "activation_inv_min": activation_inv_min,
                "activation_inv_max": activation_inv_max,
                "activation_inv_rms": activation_inv_rms,
                "next_activation_inv_y_mean": next_activation_inv_y_mean,
                "next_activation_inv_min": next_activation_inv_min,
                "next_activation_inv_max": next_activation_inv_max,
                "grad_norm": grad_norm,
                "grad_abs_max": grad_abs_max,
                "best_surface_mean_error": best_surface_mean_error,
                "stopped": stopped,
            },
        )
        series_writer.append(step_mesh, time=float(step))
        series_elapsed = time.perf_counter() - series_start
        timing["series_elapsed_s"] += series_elapsed
        step_elapsed = time.perf_counter() - step_start
        trace[-1]["series_elapsed_s"] = series_elapsed
        trace[-1]["step_elapsed_s"] = step_elapsed
        print(
            "inverse step:",
            f"{step:03d}",
            f"loss={loss_value:.3e}",
            f"mean_error={surface_mean_error:.3e}",
            f"tol={point_error_tol:.3e}",
            f"activation_inv_y_mean={activation_inv_y_mean:.8f}",
            f"next_activation_inv_y_mean={next_activation_inv_y_mean:.8f}",
            f"grad_norm={grad_norm:.3e}",
            f"step_elapsed={step_elapsed:.3f}s",
            f"optimizer_steps={optimizer_steps}",
            flush=True,
        )
        if stopped or not did_optimizer_step:
            break

    if best_displacement is None:
        msg = "inverse solve did not evaluate any forward states"
        raise RuntimeError(msg)
    if best_activation_inv is None:
        msg = "inverse solve did not record any activation_inv states"
        raise RuntimeError(msg)
    timing["inverse_elapsed_s"] = time.perf_counter() - inverse_start
    return (
        best_displacement,
        best_activation_inv,
        trace,
        stop_reason,
        optimizer_steps,
        timing,
    )


def add_surface_mask(mesh: pv.UnstructuredGrid, surface_ids: np.ndarray) -> None:
    mask = np.zeros(mesh.n_points, dtype=np.int8)
    mask[surface_ids] = 1
    mesh.point_data["SurfaceMask"] = mask


def add_metric_fields(
    mesh: pv.UnstructuredGrid,
    metrics: dict[str, float | int | bool | str],
) -> None:
    for name, value in metrics.items():
        if isinstance(value, str):
            continue
        mesh.field_data[name] = np.asarray([value])


def make_target_mesh(
    target_mesh: pv.UnstructuredGrid,
    surface_ids: np.ndarray,
) -> pv.UnstructuredGrid:
    result = target_mesh.copy(deep=True)
    add_surface_mask(result, surface_ids)
    result.point_data["TargetDisplacement"] = target_mesh.point_data["Displacement"]
    result.point_data["TargetPoint"] = (
        target_mesh.points + result.point_data["TargetDisplacement"]
    )
    return result


def make_result_mesh(
    mesh: pv.UnstructuredGrid,
    target_displacement: np.ndarray,
    displacement: np.ndarray,
    target_activation: np.ndarray,
    recovered_activation: np.ndarray,
    recovered_activation_inv: np.ndarray,
    surface_ids: np.ndarray,
    active_ids: np.ndarray,
    metrics: dict[str, float | int | bool | str],
) -> pv.UnstructuredGrid:
    result = mesh.copy(deep=True)
    add_surface_mask(result, surface_ids)
    error = displacement - target_displacement
    active_mask = np.zeros(result.n_cells, dtype=np.int8)
    active_mask[active_ids] = 1
    target_activation_inv = activation_to_activation_inv_numpy(target_activation)

    result.point_data["Displacement"] = displacement
    result.point_data["TargetDisplacement"] = target_displacement
    result.point_data["DisplacementError"] = error
    result.point_data["DisplacementErrorNorm"] = np.linalg.norm(error, axis=1)
    result.point_data["DeformedPoint"] = result.points + displacement
    result.point_data["TargetPoint"] = result.points + target_displacement

    result.cell_data["InverseActiveMask"] = active_mask
    result.cell_data["TargetActivation"] = target_activation
    result.cell_data["RecoveredActivation"] = recovered_activation
    result.cell_data["ActivationError"] = (
        result.cell_data["RecoveredActivation"] - result.cell_data["TargetActivation"]
    )
    result.cell_data["ActivationErrorNorm"] = np.linalg.norm(
        result.cell_data["ActivationError"], axis=1
    )
    result.cell_data["TargetActivationInv"] = target_activation_inv
    result.cell_data["RecoveredActivationInv"] = recovered_activation_inv
    result.cell_data["ActivationInvError"] = (
        recovered_activation_inv - target_activation_inv
    )
    add_metric_fields(result, metrics)
    return result


def summarize(
    target_activation: np.ndarray,
    recovered_activation: np.ndarray,
    target_activation_inv: np.ndarray,
    recovered_activation_inv: np.ndarray,
    target_displacement: np.ndarray,
    displacement: np.ndarray,
    surface_ids: np.ndarray,
    active_ids: np.ndarray,
    trace: list[dict[str, float]],
    stop_reason: str,
    optimizer_steps: int,
    bbox_diagonal: float,
    point_error_tol: float,
    timing: dict[str, float],
    total_elapsed_s: float,
    cfg: Config,
) -> dict[str, Any]:
    error = displacement - target_displacement
    error_norm = np.linalg.norm(error, axis=1)
    surface_error = error[surface_ids]
    surface_error_norm = np.linalg.norm(surface_error, axis=1)
    active_target = target_activation[active_ids]
    active_recovered = recovered_activation[active_ids]
    active_target_inv = target_activation_inv[active_ids]
    active_recovered_inv = recovered_activation_inv[active_ids]
    activation_error = active_recovered - active_target
    activation_error_norm = np.linalg.norm(activation_error, axis=1)
    activation_inv_error = active_recovered_inv - active_target_inv
    activation_inv_error_norm = np.linalg.norm(activation_inv_error, axis=1)
    final_loss = float(np.mean(np.square(surface_error)))
    surface_mean = float(surface_error_norm.mean())
    surface_rms = float(np.linalg.norm(surface_error) / math.sqrt(surface_ids.size))
    all_rms = float(np.linalg.norm(error) / math.sqrt(error.shape[0]))
    frame_count = len(trace)
    metrics: dict[str, Any] = {
        "target_input": str(cfg.target_input),
        "target_output": str(cfg.target_output),
        "output_input": str(cfg.output_input),
        "output_target": str(cfg.output_target),
        "output": str(cfg.output),
        "output_series": str(cfg.output_series),
        "n_points": int(displacement.shape[0]),
        "n_cells": int(target_activation.shape[0]),
        "n_surface_points": int(surface_ids.size),
        "n_active_tets": int(active_ids.size),
        "n_activation_params": int(active_ids.size * 6),
        "optimized_parameter": "activation_inv",
        "series_frames": int(frame_count),
        "inverse_max_steps": int(cfg.inverse_max_steps),
        "optimizer_steps": int(optimizer_steps),
        "stop_reason": stop_reason,
        "inverse_lr": float(cfg.inverse_lr),
        "activation_inv_residual_scale": float(cfg.activation_inv_residual_scale),
        "adam_beta1": float(cfg.adam_beta1),
        "adam_beta2": float(cfg.adam_beta2),
        "adam_eps": float(cfg.adam_eps),
        "forward_rtol": float(cfg.forward_rtol),
        "forward_atol": float(cfg.forward_atol),
        "forward_max_steps": int(cfg.forward_max_steps),
        "activation_inv_xz_abs_max": float(cfg.activation_inv_xz_abs_max),
        "activation_inv_y_min": float(cfg.activation_inv_y_min),
        "activation_inv_y_max": float(cfg.activation_inv_y_max),
        "activation_inv_shear_abs_max": float(cfg.activation_inv_shear_abs_max),
        "bbox_diagonal": bbox_diagonal,
        "point_error_rel_tol": float(cfg.point_error_rel_tol),
        "point_error_tol": point_error_tol,
        "total_elapsed_s": float(total_elapsed_s),
        **{name: float(value) for name, value in timing.items()},
        "target_active_activation_mean": active_target.mean(axis=0).tolist(),
        "recovered_active_activation_mean": active_recovered.mean(axis=0).tolist(),
        "target_active_activation_inv_mean": active_target_inv.mean(axis=0).tolist(),
        "recovered_active_activation_inv_mean": active_recovered_inv.mean(
            axis=0
        ).tolist(),
        "target_active_activation_min": active_target.min(axis=0).tolist(),
        "target_active_activation_max": active_target.max(axis=0).tolist(),
        "recovered_active_activation_min": active_recovered.min(axis=0).tolist(),
        "recovered_active_activation_max": active_recovered.max(axis=0).tolist(),
        "target_active_activation_inv_min": active_target_inv.min(axis=0).tolist(),
        "target_active_activation_inv_max": active_target_inv.max(axis=0).tolist(),
        "recovered_active_activation_inv_min": active_recovered_inv.min(
            axis=0
        ).tolist(),
        "recovered_active_activation_inv_max": active_recovered_inv.max(
            axis=0
        ).tolist(),
        "active_activation_mean_abs_error": float(np.abs(activation_error).mean()),
        "active_activation_rms_error": float(
            np.linalg.norm(activation_error) / math.sqrt(activation_error.size)
        ),
        "active_activation_max_abs_error": float(np.abs(activation_error).max()),
        "active_activation_mean_vector_error": float(activation_error_norm.mean()),
        "active_activation_max_vector_error": float(activation_error_norm.max()),
        "active_activation_inv_mean_abs_error": float(
            np.abs(activation_inv_error).mean()
        ),
        "active_activation_inv_rms_error": float(
            np.linalg.norm(activation_inv_error)
            / math.sqrt(activation_inv_error.size)
        ),
        "active_activation_inv_max_abs_error": float(
            np.abs(activation_inv_error).max()
        ),
        "active_activation_inv_mean_vector_error": float(
            activation_inv_error_norm.mean()
        ),
        "active_activation_inv_max_vector_error": float(
            activation_inv_error_norm.max()
        ),
        "final_loss": final_loss,
        "surface_mean_error": surface_mean,
        "surface_rms_error": surface_rms,
        "surface_max_error": float(surface_error_norm.max()),
        "all_rms_error": all_rms,
        "all_max_error": float(error_norm.max()),
        "trace": trace,
    }
    metrics["passed"] = bool(
        stop_reason == "point_error_tol" and surface_mean <= point_error_tol
    )
    return metrics


def save_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main(cfg: Config) -> None:
    total_start = time.perf_counter()
    configure_runtime()

    input_mesh, target_mesh = load_target(cfg)
    surface_ids = surface_point_ids(input_mesh)
    active_ids = active_cell_ids(input_mesh, cfg)
    target_displacement = np.asarray(target_mesh.point_data["Displacement"])
    target_activation = target_activation_field(input_mesh)
    target_activation_inv = activation_to_activation_inv_numpy(target_activation)
    bbox_diagonal = float(input_mesh.length)
    point_error_tol = cfg.point_error_rel_tol * bbox_diagonal

    inverse_input = zero_activation(input_mesh)
    add_surface_mask(inverse_input, surface_ids)
    melon.save(cfg.output_input, inverse_input)
    melon.save(cfg.output_target, make_target_mesh(target_mesh, surface_ids))

    inverse_mesh = inverse_input.copy(deep=True)
    with melon.SeriesWriter(cfg.output_series, clear=True) as series_writer:
        (
            displacement,
            recovered_activation_inv,
            trace,
            stop_reason,
            optimizer_steps,
            timing,
        ) = solve_inverse(
            inverse_mesh,
            target_activation,
            target_displacement,
            surface_ids,
            active_ids,
            point_error_tol,
            cfg,
            series_writer,
        )
    recovered_activation = activation_inv_to_activation_numpy(recovered_activation_inv)
    total_elapsed_s = time.perf_counter() - total_start
    summary = summarize(
        target_activation,
        recovered_activation,
        target_activation_inv,
        recovered_activation_inv,
        target_displacement,
        displacement,
        surface_ids,
        active_ids,
        trace,
        stop_reason,
        optimizer_steps,
        bbox_diagonal,
        point_error_tol,
        timing,
        total_elapsed_s,
        cfg,
    )
    metrics = {
        name: value
        for name, value in summary.items()
        if isinstance(value, int | float | bool)
    }
    result = make_result_mesh(
        inverse_mesh,
        target_displacement,
        displacement,
        target_activation,
        recovered_activation,
        recovered_activation_inv,
        surface_ids,
        active_ids,
        metrics,
    )
    melon.save(cfg.output, result)
    save_summary(cfg.output_summary, summary)
    cherries.log_metrics(metrics)
    target_y_mean = summary["target_active_activation_mean"][1]
    recovered_y_mean = summary["recovered_active_activation_mean"][1]
    target_inv_y_mean = summary["target_active_activation_inv_mean"][1]
    recovered_inv_y_mean = summary["recovered_active_activation_inv_mean"][1]
    print(
        "active activation_inv:",
        f"params={summary['n_activation_params']}",
        f"target_y_mean={target_inv_y_mean:.8f}",
        f"recovered_y_mean={recovered_inv_y_mean:.8f}",
        f"rms_error={summary['active_activation_inv_rms_error']:.3e}",
    )
    print(
        "derived activation:",
        f"target_y_mean={target_y_mean:.8f}",
        f"recovered_y_mean={recovered_y_mean:.8f}",
        f"rms_error={summary['active_activation_rms_error']:.3e}",
    )
    print(
        "surface error:",
        f"mean={summary['surface_mean_error']:.3e}",
        f"rms={summary['surface_rms_error']:.3e}",
        f"max={summary['surface_max_error']:.3e}",
        f"tol={summary['point_error_tol']:.3e}",
        f"loss={summary['final_loss']:.3e}",
    )
    print(
        "inverse stop:",
        f"reason={summary['stop_reason']}",
        f"optimizer_steps={summary['optimizer_steps']}",
        f"bbox_diagonal={summary['bbox_diagonal']:.6g}",
        f"total_elapsed={summary['total_elapsed_s']:.3f}s",
    )
    print(f"saved: {cfg.output}")
    print(f"saved: {cfg.output_series}")
    print(f"saved: {cfg.output_summary}")
    if not summary["passed"]:
        msg = "inverse activation_inv recovery missed the geometry-scaled tolerance"
        raise RuntimeError(msg)


if __name__ == "__main__":
    cherries.main(main, profile="debug")
