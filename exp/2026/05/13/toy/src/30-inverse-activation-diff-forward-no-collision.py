import contextlib
import io
import json
import logging
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
import pyvista as pv
import torch
import warp as wp

from liblaf import cherries, melon

OUTPUT_STEM = "30-inverse-activation-diff-forward-no-collision"
TARGET_STEM = "10-activation-fixed-side-no-collision"


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    target_input: Path = cherries.output(f"{TARGET_STEM}-input.vtu")
    target_output: Path = cherries.output(f"{TARGET_STEM}.vtu")
    output_input: Path = cherries.output(f"{OUTPUT_STEM}-input.vtu")
    output_target: Path = cherries.output(f"{OUTPUT_STEM}-target.vtu")
    output: Path = cherries.output(f"{OUTPUT_STEM}.vtu")
    output_summary: Path = cherries.output(f"{OUTPUT_STEM}-summary.json")

    forward_rtol: float = 1.0e-5
    forward_max_steps: int = 800
    inverse_max_steps: int = 80
    inverse_lr: float = 0.1
    adam_beta1: float = 0.3
    adam_beta2: float = 0.8
    activation_min: float = -0.75
    activation_max: float = 1.5
    point_error_rel_tol: float = 1.0e-3
    activation_tol: float = 5.0e-3


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
    result.field_data["InitialActivation"] = np.zeros((1, 6), dtype=np.float64)
    result.field_data["InitialActivationY"] = np.asarray([0.0])
    result.field_data["TargetSource"] = np.asarray([TARGET_STEM])
    return result


def target_activation(target_mesh: pv.UnstructuredGrid) -> np.ndarray:
    from liblaf.apple.common import ACTIVATION

    if "Activation" in target_mesh.field_data:
        return np.asarray(target_mesh.field_data["Activation"]).reshape(-1)[:6]
    if ACTIVATION.vtk in target_mesh.cell_data:
        return np.asarray(target_mesh.cell_data[ACTIVATION.vtk]).mean(axis=0)
    msg = "target mesh has no activation field"
    raise KeyError(msg)


def activation_vector(activation_y: float) -> np.ndarray:
    activation = np.zeros(6, dtype=np.float64)
    activation[1] = activation_y
    return activation


def activation_inv_vector(activation_y: float) -> np.ndarray:
    activation_inv = np.zeros(6, dtype=np.float64)
    activation_inv[1] = (1.0 / (1.0 + activation_y)) - 1.0
    return activation_inv


def activation_inv_field(activation_y: torch.Tensor, n_cells: int) -> torch.Tensor:
    activation_inv = torch.zeros(
        (n_cells, 6), dtype=activation_y.dtype, device=activation_y.device
    )
    activation_inv[:, 1] = (1.0 / (1.0 + activation_y)) - 1.0
    return activation_inv


def build_forward(mesh: pv.UnstructuredGrid, cfg: Config):
    from liblaf.apple.forward import Forward, ModelBuilder
    from liblaf.apple.warp.fem import StableNeoHookeanActive

    builder = ModelBuilder()
    builder.add_vertices(mesh)
    builder.add_fixed(mesh)
    builder.add_potential(StableNeoHookeanActive.from_pyvista(mesh, name="active"))
    forward = Forward(builder.finalize())
    forward.optimizer = forward.default_optimizer(
        max_steps=cfg.forward_max_steps,
        rtol=cfg.forward_rtol,
    )
    return forward


def material_tree(
    base_materials: dict[str, dict[str, torch.Tensor]],
    activation_y: torch.Tensor,
    n_cells: int,
) -> dict[str, dict[str, torch.Tensor]]:
    materials = {name: dict(values) for name, values in base_materials.items()}
    materials["active"]["activation_inv"] = activation_inv_field(activation_y, n_cells)
    return materials


def forward_quiet(differentiable_forward: Any, materials: Any) -> torch.Tensor:
    with contextlib.redirect_stdout(io.StringIO()):
        return differentiable_forward.forward(materials)


def surface_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    point_ids: torch.Tensor,
) -> torch.Tensor:
    residual = output[point_ids] - target[point_ids]
    return residual.square().mean()


def point_error_stats(residual: torch.Tensor) -> dict[str, torch.Tensor]:
    point_error = torch.linalg.vector_norm(residual, dim=1)
    return {
        "mean": point_error.mean(),
        "rms": torch.linalg.vector_norm(residual) / math.sqrt(residual.shape[0]),
        "max": point_error.max(),
    }


def solve_inverse(
    mesh: pv.UnstructuredGrid,
    target_displacement: np.ndarray,
    surface_ids: np.ndarray,
    point_error_tol: float,
    cfg: Config,
) -> tuple[np.ndarray, float, list[dict[str, float]], str, int]:
    from liblaf.apple.inverse import DifferentiableForward

    forward = build_forward(mesh, cfg)
    differentiable_forward = DifferentiableForward(forward)
    base_materials = forward.model.get_materials()

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
    activation_y = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.Adam(
        [activation_y],
        lr=cfg.inverse_lr,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
    )

    trace: list[dict[str, float]] = []
    stop_reason = "inverse_max_steps"
    optimizer_steps = 0
    for step in range(cfg.inverse_max_steps + 1):
        optimizer.zero_grad()
        materials = material_tree(base_materials, activation_y, mesh.n_cells)
        output = forward_quiet(differentiable_forward, materials)
        loss = surface_loss(output, target, surface_ids_t)
        loss.backward()
        if activation_y.grad is None:
            msg = "differentiable forward did not produce activation gradients"
            raise RuntimeError(msg)
        if not torch.isfinite(activation_y.grad):
            grad = float(activation_y.grad.detach().cpu())
            msg = f"non-finite inverse gradient at step {step}: {grad}"
            raise FloatingPointError(msg)

        residual = output.detach()[surface_ids_t] - target[surface_ids_t]
        error_stats = point_error_stats(residual)
        grad_value = float(activation_y.grad.detach().cpu())
        loss_value = float(loss.detach().cpu())
        surface_mean_error = float(error_stats["mean"].cpu())
        stopped = surface_mean_error <= point_error_tol
        if stopped:
            stop_reason = "point_error_tol"
        elif optimizer_steps < cfg.inverse_max_steps:
            optimizer.step()
            optimizer_steps += 1
            with torch.no_grad():
                activation_y.clamp_(cfg.activation_min, cfg.activation_max)
        trace.append(
            {
                "step": float(step),
                "loss": loss_value,
                "surface_mean_error": surface_mean_error,
                "surface_rms_error": float(error_stats["rms"].cpu()),
                "surface_max_error": float(error_stats["max"].cpu()),
                "activation_y": float(activation_y.detach().cpu()),
                "grad_activation_y": grad_value,
                "optimizer_steps": float(optimizer_steps),
                "stopped": float(stopped),
            }
        )
        if stopped or optimizer_steps >= cfg.inverse_max_steps:
            break

    final_activation_y = float(activation_y.detach().cpu())
    final_y = torch.as_tensor(
        final_activation_y,
        dtype=torch.get_default_dtype(),
        device=torch.get_default_device(),
    )
    final_materials = material_tree(base_materials, final_y, mesh.n_cells)
    with torch.no_grad():
        output = forward_quiet(differentiable_forward, final_materials)
    return to_numpy(output), final_activation_y, trace, stop_reason, optimizer_steps


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
    target_activation_value: np.ndarray,
    recovered_activation_y: float,
    surface_ids: np.ndarray,
    metrics: dict[str, float | int | bool | str],
) -> pv.UnstructuredGrid:
    result = mesh.copy(deep=True)
    add_surface_mask(result, surface_ids)
    error = displacement - target_displacement
    target_activation_inv = activation_inv_vector(float(target_activation_value[1]))
    recovered_activation = activation_vector(recovered_activation_y)
    recovered_activation_inv = activation_inv_vector(recovered_activation_y)

    result.point_data["Displacement"] = displacement
    result.point_data["TargetDisplacement"] = target_displacement
    result.point_data["DisplacementError"] = error
    result.point_data["DisplacementErrorNorm"] = np.linalg.norm(error, axis=1)
    result.point_data["DeformedPoint"] = result.points + displacement
    result.point_data["TargetPoint"] = result.points + target_displacement

    result.cell_data["TargetActivation"] = np.repeat(
        target_activation_value[np.newaxis, :], result.n_cells, axis=0
    )
    result.cell_data["RecoveredActivation"] = np.repeat(
        recovered_activation[np.newaxis, :], result.n_cells, axis=0
    )
    result.cell_data["ActivationError"] = (
        result.cell_data["RecoveredActivation"] - result.cell_data["TargetActivation"]
    )
    result.cell_data["TargetActivationInv"] = np.repeat(
        target_activation_inv[np.newaxis, :], result.n_cells, axis=0
    )
    result.cell_data["RecoveredActivationInv"] = np.repeat(
        recovered_activation_inv[np.newaxis, :], result.n_cells, axis=0
    )
    add_metric_fields(result, metrics)
    return result


def summarize(
    target_activation_value: np.ndarray,
    recovered_activation_y: float,
    target_displacement: np.ndarray,
    displacement: np.ndarray,
    surface_ids: np.ndarray,
    trace: list[dict[str, float]],
    stop_reason: str,
    optimizer_steps: int,
    bbox_diagonal: float,
    point_error_tol: float,
    cfg: Config,
) -> dict[str, Any]:
    error = displacement - target_displacement
    error_norm = np.linalg.norm(error, axis=1)
    surface_error = error[surface_ids]
    surface_error_norm = np.linalg.norm(surface_error, axis=1)
    target_activation_y = float(target_activation_value[1])
    recovered_activation = activation_vector(recovered_activation_y)
    activation_abs_error = abs(recovered_activation_y - target_activation_y)
    final_loss = float(np.mean(np.square(surface_error)))
    surface_mean = float(surface_error_norm.mean())
    surface_rms = float(np.linalg.norm(surface_error) / math.sqrt(surface_ids.size))
    all_rms = float(np.linalg.norm(error) / math.sqrt(error.shape[0]))
    metrics: dict[str, Any] = {
        "target_input": str(cfg.target_input),
        "target_output": str(cfg.target_output),
        "n_points": int(displacement.shape[0]),
        "n_surface_points": int(surface_ids.size),
        "inverse_max_steps": int(cfg.inverse_max_steps),
        "optimizer_steps": int(optimizer_steps),
        "stop_reason": stop_reason,
        "inverse_lr": float(cfg.inverse_lr),
        "adam_beta1": float(cfg.adam_beta1),
        "adam_beta2": float(cfg.adam_beta2),
        "forward_rtol": float(cfg.forward_rtol),
        "bbox_diagonal": bbox_diagonal,
        "point_error_rel_tol": float(cfg.point_error_rel_tol),
        "point_error_tol": point_error_tol,
        "target_activation_y": target_activation_y,
        "recovered_activation_y": recovered_activation_y,
        "activation_abs_error": activation_abs_error,
        "target_activation": target_activation_value.tolist(),
        "recovered_activation": recovered_activation.tolist(),
        "final_loss": final_loss,
        "surface_mean_error": surface_mean,
        "surface_rms_error": surface_rms,
        "surface_max_error": float(surface_error_norm.max()),
        "all_rms_error": all_rms,
        "all_max_error": float(error_norm.max()),
        "trace": trace,
    }
    metrics["passed"] = bool(
        stop_reason == "point_error_tol"
        and surface_mean <= point_error_tol
        and activation_abs_error <= cfg.activation_tol
    )
    return metrics


def save_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main(cfg: Config) -> None:
    configure_runtime()

    input_mesh, target_mesh = load_target(cfg)
    surface_ids = surface_point_ids(input_mesh)
    target_displacement = np.asarray(target_mesh.point_data["Displacement"])
    target_activation_value = target_activation(target_mesh)
    bbox_diagonal = float(input_mesh.length)
    point_error_tol = cfg.point_error_rel_tol * bbox_diagonal

    inverse_input = zero_activation(input_mesh)
    add_surface_mask(inverse_input, surface_ids)
    melon.save(cfg.output_input, inverse_input)
    melon.save(cfg.output_target, make_target_mesh(target_mesh, surface_ids))

    inverse_mesh = inverse_input.copy(deep=True)
    displacement, recovered_activation_y, trace, stop_reason, optimizer_steps = (
        solve_inverse(
            inverse_mesh, target_displacement, surface_ids, point_error_tol, cfg
        )
    )
    summary = summarize(
        target_activation_value,
        recovered_activation_y,
        target_displacement,
        displacement,
        surface_ids,
        trace,
        stop_reason,
        optimizer_steps,
        bbox_diagonal,
        point_error_tol,
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
        target_activation_value,
        recovered_activation_y,
        surface_ids,
        metrics,
    )
    melon.save(cfg.output, result)
    save_summary(cfg.output_summary, summary)

    cherries.log_metrics(metrics)
    print(
        "activation_y:",
        f"target={summary['target_activation_y']:.8f}",
        f"recovered={summary['recovered_activation_y']:.8f}",
        f"abs_error={summary['activation_abs_error']:.3e}",
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
    )
    print(f"saved: {cfg.output}")
    print(f"saved: {cfg.output_summary}")
    if not summary["passed"]:
        msg = "inverse activation recovery missed the geometry-scaled tolerance"
        raise RuntimeError(msg)


if __name__ == "__main__":
    cherries.main(main, profile="debug")
