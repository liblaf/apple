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
    report: Path = EXPERIMENT_DIR / "docs" / "README.md"

    E: float = 1.0
    nu: float = 0.49
    smas_stiffness_ratio: float = 1.0e2
    active_fraction_tol: float = 1.0e-3

    forward_rtol: float = 1.0e-2
    forward_atol: float = 1.0e-4
    forward_max_steps: int = 600

    inverse_lr: float = 0.75
    adam_beta1: float = 0.0
    adam_beta2: float = 0.8
    adam_eps: float = 1.0e-8
    inverse_max_steps: int = 160
    inverse_min_steps: int = 4
    stagnation_patience: int = 10
    stagnation_rel_tol: float = 1.0e-3
    stagnation_abs_tol: float = 1.0e-8
    loss_tol: float = 1.0e-5
    point_error_tol: float = 0.1
    activation_l2_weight: float = 1.0e-8
    activation_det_min: float = 1.0e-3
    activation_det_weight: float = 1.0e-7
    adjoint_maxiter: int = 80
    adjoint_rtol: float = 1.0e-2
    adjoint_atol: float = 0.0

    activation_diag_min: float = -0.95
    activation_diag_max: float = 4.0
    activation_shear_abs_max: float = 0.75
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


def target_point_ids(target: pv.UnstructuredGrid) -> np.ndarray:
    if TARGET_SURFACE_MASK in target.point_data:
        mask = np.asarray(target.point_data[TARGET_SURFACE_MASK], dtype=bool)
        ids = np.flatnonzero(mask)
        if ids.size > 0:
            return ids.astype(np.int64)
    surface = target.extract_surface(algorithm=None)
    ids = np.asarray(surface.point_data["vtkOriginalPointIds"], dtype=np.int64)
    return np.unique(ids)


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
    inverse = torch.linalg.inv(matrices)
    return pack_activation_matrices_torch(inverse)


def activation_to_activation_inv_numpy(activation: np.ndarray) -> np.ndarray:
    activation_t = torch.as_tensor(activation, dtype=torch.float64, device="cpu")
    return activation_to_activation_inv_torch(activation_t).numpy()


def clamp_active_activation_(active_activation: torch.Tensor, cfg: Config) -> None:
    active_activation[:, :3].clamp_(cfg.activation_diag_min, cfg.activation_diag_max)
    active_activation[:, 3:].clamp_(
        -cfg.activation_shear_abs_max, cfg.activation_shear_abs_max
    )


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

    background_fraction = np.asarray(mesh.cell_data[BACKGROUND_FRACTION])
    active_fraction = np.asarray(mesh.cell_data[ACTIVE_FRACTION])
    smas_fraction = np.asarray(mesh.cell_data[SMAS_STIFFNESS_FRACTION])

    builder = ModelBuilder()
    builder.add_vertices(mesh)
    builder.add_fixed(mesh)

    set_material(mesh, E=cfg.E, nu=cfg.nu, fraction=background_fraction)
    builder.add_potential(StableNeoHookean.from_pyvista(mesh, name="background"))

    set_material(mesh, E=cfg.E, nu=cfg.nu, fraction=active_fraction)
    builder.add_potential(StableNeoHookeanActive.from_pyvista(mesh, name="muscle"))

    set_material(
        mesh,
        E=cfg.smas_stiffness_ratio * cfg.E,
        nu=cfg.nu,
        fraction=smas_fraction,
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
    active_activation: torch.Tensor,
    active_ids_t: torch.Tensor,
    n_cells: int,
) -> dict[str, dict[str, torch.Tensor]]:
    materials = {name: dict(values) for name, values in base_materials.items()}
    active_activation_inv = activation_to_activation_inv_torch(active_activation)
    materials["muscle"]["activation_inv"] = full_activation_inv_from_active(
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


def add_point_masks(
    mesh: pv.UnstructuredGrid, target_point_ids: np.ndarray, active_ids: np.ndarray
) -> None:
    target_mask = np.zeros(mesh.n_points, dtype=np.int8)
    target_mask[target_point_ids] = 1
    active_mask = np.zeros(mesh.n_cells, dtype=np.int8)
    active_mask[active_ids] = 1
    mesh.point_data[TARGET_SURFACE_MASK] = target_mask
    mesh.cell_data["InverseActiveMask"] = active_mask


def make_target_mesh(
    target: pv.UnstructuredGrid, target_point_ids: np.ndarray, active_ids: np.ndarray
) -> pv.UnstructuredGrid:
    result = target.copy(deep=True)
    add_point_masks(result, target_point_ids, active_ids)
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
    target_point_ids: np.ndarray,
    active_ids: np.ndarray,
    metrics: dict[str, float | int | bool | str],
) -> pv.UnstructuredGrid:
    from liblaf.apple.common import ACTIVATION, ACTIVATION_INV

    result = mesh.copy(deep=True)
    add_point_masks(result, target_point_ids, active_ids)
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
    step: int, *, improved: bool, stopped: bool, cfg: Config
) -> bool:
    stride = max(1, cfg.series_stride)
    return step == 0 or improved or stopped or step % stride == 0


def regularization_losses(
    active_activation: torch.Tensor, cfg: Config
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    activation_l2 = cfg.activation_l2_weight * active_activation.square().mean()
    matrices = activation_matrices_torch(active_activation)
    det = torch.linalg.det(matrices)
    det_penalty = torch.relu(cfg.activation_det_min - det).square().mean()
    det_loss = cfg.activation_det_weight * det_penalty
    return activation_l2, det_loss, det


def solve_inverse(  # noqa: C901, PLR0912, PLR0915
    mesh: pv.UnstructuredGrid,
    target_displacement: np.ndarray,
    target_point_ids: np.ndarray,
    active_ids: np.ndarray,
    cfg: Config,
    series_writer: Any,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]], str, int, dict[str, float]]:
    from liblaf.peach.linalg import FallbackSolver
    from liblaf.peach.linalg.cupy import CupyCG, CupyMinRes

    from liblaf.apple.inverse import DifferentiableForward

    forward = build_forward(mesh, cfg)
    differentiable_forward = DifferentiableForward(forward)
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
        mesh, target_displacement, target_point_ids
    )
    active_ids_t = torch.as_tensor(
        active_ids,
        dtype=torch.long,
        device=torch.get_default_device(),
    )
    active_activation = torch.nn.Parameter(
        torch.zeros(
            (active_ids.size, 6),
            dtype=torch.get_default_dtype(),
            device=torch.get_default_device(),
        )
    )
    optimizer = torch.optim.Adam(
        [active_activation],
        lr=cfg.inverse_lr,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_eps,
    )

    trace: list[dict[str, float]] = []
    stop_reason = "inverse_max_steps"
    optimizer_steps = 0
    best_displacement: np.ndarray | None = None
    best_activation: np.ndarray | None = None
    best_surface_mean_error = math.inf
    best_loss = math.inf
    no_improve_steps = 0
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
            base_materials, active_activation, active_ids_t, mesh.n_cells
        )

        forward_start = time.perf_counter()
        output = forward_quiet(differentiable_forward, materials)
        forward_elapsed = time.perf_counter() - forward_start
        timing["forward_elapsed_s"] += forward_elapsed

        backward_start = time.perf_counter()
        residual = output[point_global_ids_t] - target[point_ids_t]
        data_loss = residual.square().mean()
        activation_l2, det_loss, det = regularization_losses(active_activation, cfg)
        loss = data_loss + activation_l2 + det_loss
        loss.backward()
        backward_elapsed = time.perf_counter() - backward_start
        timing["backward_elapsed_s"] += backward_elapsed

        if active_activation.grad is None:
            msg = "differentiable forward did not produce activation gradients"
            raise RuntimeError(msg)
        grad = active_activation.grad.detach()
        if not torch.isfinite(grad).all():
            nonfinite = int((~torch.isfinite(grad)).sum().detach().cpu())
            msg = f"non-finite inverse gradient at step {step}: {nonfinite} entries"
            raise FloatingPointError(msg)

        error_stats = point_error_stats(residual.detach())
        data_loss_value = float(data_loss.detach().cpu())
        loss_value = float(loss.detach().cpu())
        activation_l2_value = float(activation_l2.detach().cpu())
        det_loss_value = float(det_loss.detach().cpu())
        surface_mean_error = float(error_stats["mean"].cpu())
        surface_rms_error = float(error_stats["rms"].cpu())
        surface_max_error = float(error_stats["max"].cpu())
        grad_norm = float(torch.linalg.vector_norm(grad).cpu())
        grad_abs_max = float(grad.abs().max().cpu())
        active_values = active_activation.detach()
        active_inv_values = activation_to_activation_inv_torch(active_values)
        activation_rms = float(
            torch.linalg.vector_norm(active_values).cpu()
            / math.sqrt(active_values.numel())
        )
        activation_min = float(active_values.min().cpu())
        activation_max = float(active_values.max().cpu())
        activation_inv_rms = float(
            torch.linalg.vector_norm(active_inv_values).cpu()
            / math.sqrt(active_inv_values.numel())
        )
        det_min = float(det.detach().min().cpu())
        det_mean = float(det.detach().mean().cpu())
        det_max = float(det.detach().max().cpu())
        displacement = to_numpy(output)[global_ids]

        improved = surface_mean_error < (
            best_surface_mean_error * (1.0 - cfg.stagnation_rel_tol)
            - cfg.stagnation_abs_tol
        )
        if surface_mean_error < best_surface_mean_error:
            best_surface_mean_error = surface_mean_error
            best_loss = data_loss_value
            best_activation = to_numpy(
                full_activation_from_active(active_values, active_ids_t, mesh.n_cells)
            )
            best_displacement = displacement
        if improved or step == 0:
            no_improve_steps = 0
        else:
            no_improve_steps += 1

        stopped = False
        if surface_mean_error <= cfg.point_error_tol:
            stop_reason = "point_error_tol"
            stopped = True
        elif data_loss_value <= cfg.loss_tol:
            stop_reason = "loss_tol"
            stopped = True
        elif (
            step >= cfg.inverse_min_steps
            and no_improve_steps >= cfg.stagnation_patience
        ):
            stop_reason = "stagnation"
            stopped = True

        did_optimizer_step = False
        if not stopped and optimizer_steps < cfg.inverse_max_steps:
            optimizer_start = time.perf_counter()
            optimizer.step()
            optimizer_steps += 1
            did_optimizer_step = True
            timing["optimizer_elapsed_s"] += time.perf_counter() - optimizer_start
            with torch.no_grad():
                clamp_active_activation_(active_activation, cfg)

        trace_record = {
            "step": float(step),
            "loss": loss_value,
            "data_loss": data_loss_value,
            "activation_l2_loss": activation_l2_value,
            "activation_det_loss": det_loss_value,
            "target_mean_error": surface_mean_error,
            "target_rms_error": surface_rms_error,
            "target_max_error": surface_max_error,
            "point_error_tol": cfg.point_error_tol,
            "activation_rms": activation_rms,
            "activation_min": activation_min,
            "activation_max": activation_max,
            "activation_inv_rms": activation_inv_rms,
            "activation_det_min": det_min,
            "activation_det_mean": det_mean,
            "activation_det_max": det_max,
            "grad_norm": grad_norm,
            "grad_abs_max": grad_abs_max,
            "optimizer_steps": float(optimizer_steps),
            "best_loss": best_loss,
            "best_target_mean_error": best_surface_mean_error,
            "no_improve_steps": float(no_improve_steps),
            "stopped": float(stopped),
            "forward_elapsed_s": forward_elapsed,
            "backward_elapsed_s": backward_elapsed,
        }
        trace.append(trace_record)

        if should_write_series(step, improved=improved, stopped=stopped, cfg=cfg):
            series_start = time.perf_counter()
            evaluated_activation = to_numpy(
                full_activation_from_active(active_values, active_ids_t, mesh.n_cells)
            )
            evaluated_activation_inv = to_numpy(
                full_activation_inv_from_active(
                    active_inv_values, active_ids_t, mesh.n_cells
                )
            )
            step_mesh = make_result_mesh(
                mesh,
                target_displacement,
                displacement,
                evaluated_activation,
                evaluated_activation_inv,
                target_point_ids,
                active_ids,
                {
                    "inverse_step": step,
                    "optimizer_steps": optimizer_steps,
                    "loss": loss_value,
                    "data_loss": data_loss_value,
                    "target_mean_error": surface_mean_error,
                    "target_rms_error": surface_rms_error,
                    "target_max_error": surface_max_error,
                    "point_error_tol": cfg.point_error_tol,
                    "activation_rms": activation_rms,
                    "activation_min": activation_min,
                    "activation_max": activation_max,
                    "activation_inv_rms": activation_inv_rms,
                    "activation_det_min": det_min,
                    "activation_det_mean": det_mean,
                    "activation_det_max": det_max,
                    "grad_norm": grad_norm,
                    "grad_abs_max": grad_abs_max,
                    "best_loss": best_loss,
                    "best_target_mean_error": best_surface_mean_error,
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
            f"data={data_loss_value:.3e}",
            f"mean={surface_mean_error:.3e}",
            f"tol={cfg.point_error_tol:.3e}",
            f"best={best_surface_mean_error:.3e}",
            f"act={activation_rms:.3e}",
            f"det_min={det_min:.3e}",
            f"grad={grad_norm:.3e}",
            f"no_improve={no_improve_steps}",
            f"elapsed={step_elapsed:.2f}s",
            flush=True,
        )
        if stopped or not did_optimizer_step:
            break

    if best_displacement is None or best_activation is None:
        msg = "inverse solve did not evaluate any forward states"
        raise RuntimeError(msg)
    timing["inverse_elapsed_s"] = time.perf_counter() - inverse_start
    return (
        best_displacement,
        best_activation,
        trace,
        stop_reason,
        optimizer_steps,
        timing,
    )


def summarize(
    mesh: pv.UnstructuredGrid,
    target_displacement: np.ndarray,
    displacement: np.ndarray,
    recovered_activation: np.ndarray,
    recovered_activation_inv: np.ndarray,
    target_point_ids: np.ndarray,
    active_ids: np.ndarray,
    trace: list[dict[str, float]],
    stop_reason: str,
    optimizer_steps: int,
    bbox_diagonal: float,
    timing: dict[str, float],
    total_elapsed_s: float,
    cfg: Config,
) -> dict[str, Any]:
    error = displacement - target_displacement
    error_norm = np.linalg.norm(error, axis=1)
    target_error = error[target_point_ids]
    target_error_norm = np.linalg.norm(target_error, axis=1)
    target_norm = np.linalg.norm(target_displacement[target_point_ids], axis=1)
    active_activation = recovered_activation[active_ids]
    active_activation_inv = recovered_activation_inv[active_ids]
    active_det = np.linalg.det(to_activation_matrices_numpy(active_activation))
    final_loss = float(np.mean(np.square(target_error)))
    target_rms = float(np.linalg.norm(target_error) / math.sqrt(target_point_ids.size))
    all_rms = float(np.linalg.norm(error) / math.sqrt(error.shape[0]))
    metrics: dict[str, Any] = {
        "input": str(cfg.input),
        "target": str(cfg.target),
        "output": str(cfg.output),
        "output_series": str(cfg.output_series),
        "output_summary": str(cfg.output_summary),
        "report": str(cfg.report),
        "n_points": int(mesh.n_points),
        "n_cells": int(mesh.n_cells),
        "n_target_points": int(target_point_ids.size),
        "n_active_tets": int(active_ids.size),
        "n_activation_params": int(active_ids.size * 6),
        "optimized_parameter": "Activation",
        "E": float(cfg.E),
        "nu": float(cfg.nu),
        "smas_stiffness_ratio": float(cfg.smas_stiffness_ratio),
        "inverse_lr": float(cfg.inverse_lr),
        "adam_beta1": float(cfg.adam_beta1),
        "adam_beta2": float(cfg.adam_beta2),
        "adam_eps": float(cfg.adam_eps),
        "forward_rtol": float(cfg.forward_rtol),
        "forward_atol": float(cfg.forward_atol),
        "forward_max_steps": int(cfg.forward_max_steps),
        "inverse_max_steps": int(cfg.inverse_max_steps),
        "inverse_min_steps": int(cfg.inverse_min_steps),
        "optimizer_steps": int(optimizer_steps),
        "series_stride": int(cfg.series_stride),
        "series_frames": int(
            sum(1 for record in trace if "series_elapsed_s" in record)
        ),
        "stop_reason": stop_reason,
        "bbox_diagonal": bbox_diagonal,
        "point_error_tol": float(cfg.point_error_tol),
        "loss_tol": float(cfg.loss_tol),
        "stagnation_patience": int(cfg.stagnation_patience),
        "stagnation_rel_tol": float(cfg.stagnation_rel_tol),
        "activation_l2_weight": float(cfg.activation_l2_weight),
        "activation_det_min": float(cfg.activation_det_min),
        "activation_det_weight": float(cfg.activation_det_weight),
        "activation_diag_min": float(cfg.activation_diag_min),
        "activation_diag_max": float(cfg.activation_diag_max),
        "activation_shear_abs_max": float(cfg.activation_shear_abs_max),
        "adjoint_maxiter": int(cfg.adjoint_maxiter),
        "adjoint_rtol": float(cfg.adjoint_rtol),
        "adjoint_atol": float(cfg.adjoint_atol),
        "total_elapsed_s": float(total_elapsed_s),
        **{name: float(value) for name, value in timing.items()},
        "target_displacement_mean": float(target_norm.mean()),
        "target_displacement_rms": float(
            np.linalg.norm(target_displacement[target_point_ids])
            / math.sqrt(target_point_ids.size)
        ),
        "target_displacement_max": float(target_norm.max()),
        "final_loss": final_loss,
        "best_loss": float(min(record["data_loss"] for record in trace)),
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
        "active_activation_det_min": float(active_det.min()),
        "active_activation_det_mean": float(active_det.mean()),
        "active_activation_det_max": float(active_det.max()),
        "trace": trace,
    }
    metrics["passed"] = bool(metrics["target_mean_error"] <= cfg.point_error_tol)
    return metrics


def to_activation_matrices_numpy(activation: np.ndarray) -> np.ndarray:
    matrices = np.zeros((*activation.shape[:-1], 3, 3), dtype=np.float64)
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
        "# Inverse Face",
        "",
        "## Result",
        "",
        f"- stop reason: `{summary['stop_reason']}`",
        f"- passed: `{summary['passed']}`",
        f"- target mean error: `{fmt(summary['target_mean_error'])} cm`",
        f"- target mean error tolerance: `{fmt(summary['point_error_tol'])} cm`",
        f"- target RMS error: `{fmt(summary['target_rms_error'])} cm`",
        f"- target max error: `{fmt(summary['target_max_error'])} cm`",
        f"- final loss: `{fmt(summary['final_loss'])}`",
        f"- optimizer steps: `{summary['optimizer_steps']}`",
        f"- series frames: `{summary['series_frames']}`",
        "",
        "## Problem",
        "",
        f"- input: `{summary['input']}`",
        f"- target: `{summary['target']}`",
        f"- output: `{summary['output']}`",
        f"- optimization series: `{summary['output_series']}`",
        f"- points: `{summary['n_points']}`",
        f"- tetrahedra: `{summary['n_cells']}`",
        f"- target surface points: `{summary['n_target_points']}`",
        f"- active muscle tetrahedra: `{summary['n_active_tets']}`",
        f"- activation parameters: `{summary['n_activation_params']}`",
        "",
        "## Model",
        "",
        f"- material: stable neo-Hookean, `nu = {summary['nu']}`",
        f"- SMAS stiffness ratio: `{summary['smas_stiffness_ratio']}`",
        "- collisions: `off`",
        f"- optimized field: `{summary['optimized_parameter']}`",
        f"- Adam: `lr={summary['inverse_lr']}`, "
        f"`betas=({summary['adam_beta1']}, {summary['adam_beta2']})`",
        "",
        "## Trace",
        "",
        "| step | loss | target mean error | best mean error | activation RMS | grad norm |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        (
            "| "
            f"{int(record['step'])} | "
            f"{fmt(record['data_loss'])} | "
            f"{fmt(record['target_mean_error'])} | "
            f"{fmt(record['best_target_mean_error'])} | "
            f"{fmt(record['activation_rms'])} | "
            f"{fmt(record['grad_norm'])} |"
        )
        for record in summary["trace"]
    )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


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
    active_ids = active_cell_ids(mesh, cfg)
    target_ids = target_point_ids(target)
    target_displacement = np.asarray(
        target.point_data["Displacement"], dtype=np.float64
    )
    bbox_diagonal = float(mesh.length)

    add_point_masks(mesh, target_ids, active_ids)
    melon.save(cfg.output_input, mesh)
    melon.save(cfg.output_target, make_target_mesh(target, target_ids, active_ids))

    inverse_mesh = mesh.copy(deep=True)
    with melon.SeriesWriter(cfg.output_series, clear=True) as series_writer:
        (
            displacement,
            recovered_activation,
            trace,
            stop_reason,
            optimizer_steps,
            timing,
        ) = solve_inverse(
            inverse_mesh,
            target_displacement,
            target_ids,
            active_ids,
            cfg,
            series_writer,
        )
    recovered_activation_inv = activation_to_activation_inv_numpy(recovered_activation)
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
        bbox_diagonal,
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
    cherries.log_metrics(numeric_metrics(summary))

    print(
        "inverse result:",
        f"stop={summary['stop_reason']}",
        f"passed={summary['passed']}",
        f"target_mean_error={summary['target_mean_error']:.3e}",
        f"target_rms_error={summary['target_rms_error']:.3e}",
        f"loss={summary['final_loss']:.3e}",
        f"steps={summary['optimizer_steps']}",
    )
    print(f"saved: {cfg.output}")
    print(f"saved: {cfg.output_series}")
    print(f"saved: {cfg.output_summary}")
    print(f"saved: {cfg.report}")
    if not summary["passed"]:
        msg = (
            "inverse solve did not reach the required target mean error "
            f"({summary['target_mean_error']:.6g} cm > {cfg.point_error_tol:.6g} cm)"
        )
        raise RuntimeError(msg)


if __name__ == "__main__":
    cherries.main(main)
