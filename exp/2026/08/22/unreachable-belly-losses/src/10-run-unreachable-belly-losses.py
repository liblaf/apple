from __future__ import annotations

# ruff: noqa: C901, EM101, EM102, FBT003, I001, PLR0912, PLR0915, TRY003

import csv
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pydantic_settings as ps
import scipy.optimize as spo
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from matplotlib import animation
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from liblaf import cherries

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
DESIGN = "fixed-rim-linear-membrane-with-orthogonal-unreachable-target"
LOSS_NAMES = ("l1", "l2", "linf")
LOSS_LABELS = {"l1": "L1 / MAE", "l2": "L2 / RMS", "linf": "L-inf / max"}
LOSS_COLORS = {"l1": "#009E73", "l2": "#0072B2", "linf": "#E69F00"}
LOSS_METRIC_KEYS = {"l1": "mae", "l2": "rmse", "linf": "linf"}
TARGET_COLOR = "#CC79A7"

LossName = Literal["l1", "l2", "linf"]
DimensionName = Literal["2d", "3d"]


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)

    output_dir: Path = cherries.output("10-unreachable-belly-losses", mkdir=True)
    steps: int = 600
    learning_rate: float = 0.08
    final_learning_rate: float = 5.0e-4
    initial_control: float = 0.05
    control_min: float = 0.0
    control_max: float = 1.0
    grid_2d: int = 101
    grid_3d: int = 31
    actuator_scale: float = 0.055
    unreachable_amplitude_2d: float = 0.035
    unreachable_amplitude_3d: float = 0.030
    foundation_stiffness: float = 1.0
    membrane_stiffness: float = 0.9
    bending_stiffness: float = 0.15
    snapshot_every: int = 20
    video_fps: int = 8
    render_video: bool = True


@dataclass(frozen=True)
class BellyModel:
    dimension: DimensionName
    grid_shape: tuple[int, ...]
    x: np.ndarray
    z: np.ndarray | None
    rest_height: np.ndarray
    response: np.ndarray
    target_displacement: np.ndarray
    interior_indices: np.ndarray
    actuator_centers: np.ndarray
    teacher_controls: np.ndarray
    unreachable_component: np.ndarray
    metadata: dict[str, Any]

    @property
    def target_height(self) -> np.ndarray:
        return self.rest_height + self.target_displacement

    @property
    def n_points(self) -> int:
        return int(self.rest_height.size)

    @property
    def n_controls(self) -> int:
        return int(self.response.shape[1])


@dataclass(frozen=True)
class ReferenceOptimum:
    loss_name: LossName
    controls: np.ndarray
    displacement: np.ndarray
    metrics: dict[str, float]
    objective: float
    solver: str
    solver_status: str


@dataclass(frozen=True)
class CaseResult:
    dimension: DimensionName
    loss_name: LossName
    steps: np.ndarray
    controls: np.ndarray
    displacements: np.ndarray
    objectives: np.ndarray
    mae: np.ndarray
    rmse: np.ndarray
    linf: np.ndarray
    grad_norm: np.ndarray
    learning_rate: np.ndarray
    worst_index: np.ndarray
    worst_ties: np.ndarray
    best_index: int
    reference: ReferenceOptimum
    summary: dict[str, Any]


def validate_config(cfg: Config) -> None:
    if cfg.steps < 1:
        raise ValueError("steps must be positive")
    if cfg.learning_rate <= 0.0 or cfg.final_learning_rate < 0.0:
        raise ValueError("learning rates must be nonnegative, with initial > 0")
    if cfg.final_learning_rate > cfg.learning_rate:
        raise ValueError("final learning rate must not exceed initial learning rate")
    if not cfg.control_min <= cfg.initial_control <= cfg.control_max:
        raise ValueError("initial control must lie inside the control interval")
    if cfg.control_min >= cfg.control_max:
        raise ValueError("control interval must have positive width")
    if cfg.grid_2d < 17 or cfg.grid_2d % 2 == 0:
        raise ValueError("grid_2d must be an odd integer of at least 17")
    if cfg.grid_3d < 11 or cfg.grid_3d % 2 == 0:
        raise ValueError("grid_3d must be an odd integer of at least 11")
    if cfg.actuator_scale <= 0.0:
        raise ValueError("actuator scale must be positive")
    if cfg.unreachable_amplitude_2d <= 0.0 or cfg.unreachable_amplitude_3d <= 0.0:
        raise ValueError("unreachable amplitudes must be positive")
    if (
        min(
            cfg.foundation_stiffness,
            cfg.membrane_stiffness,
            cfg.bending_stiffness,
        )
        < 0.0
    ):
        raise ValueError("stiffness coefficients must be nonnegative")
    if cfg.foundation_stiffness == 0.0 and cfg.membrane_stiffness == 0.0:
        raise ValueError("the equilibrium operator must be nonsingular")
    if cfg.snapshot_every < 1 or cfg.video_fps < 1:
        raise ValueError("snapshot_every and video_fps must be positive")
    if cfg.output_dir.exists() and any(cfg.output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite nonempty output: {cfg.output_dir}"
        )


def graph_laplacian_1d(n: int) -> sp.csr_matrix:
    if n < 1:
        raise ValueError("interior grid size must be positive")
    return sp.diags(
        diagonals=(-np.ones(n - 1), 2.0 * np.ones(n), -np.ones(n - 1)),
        offsets=(-1, 0, 1),
        format="csr",
    )


def equilibrium_operator(laplacian: sp.csr_matrix, cfg: Config) -> sp.csc_matrix:
    identity = sp.eye(laplacian.shape[0], format="csr")
    operator = (
        cfg.foundation_stiffness * identity
        + cfg.membrane_stiffness * laplacian
        + cfg.bending_stiffness * (laplacian @ laplacian)
    )
    return operator.tocsc()


def normalized_responses(
    operator: sp.csc_matrix, loads: np.ndarray, actuator_scale: float
) -> np.ndarray:
    solved = np.asarray(spla.spsolve(operator, loads), dtype=np.float64)
    if solved.ndim == 1:
        solved = solved[:, None]
    column_max = np.max(np.abs(solved), axis=0)
    if np.any(column_max <= 1.0e-14):
        raise ValueError("an actuator produced a numerically zero response")
    return -actuator_scale * solved / column_max[None, :]


def make_unreachable_target(
    response_interior: np.ndarray,
    seed: np.ndarray,
    teacher_controls: np.ndarray,
    amplitude: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    seed_coefficients = np.linalg.lstsq(response_interior, seed, rcond=None)[0]
    unreachable = seed - response_interior @ seed_coefficients
    scale = float(np.max(np.abs(unreachable)))
    if scale <= 1.0e-10:
        raise ValueError("the unreachable seed lies inside the actuator response space")
    unreachable /= scale
    target = response_interior @ teacher_controls + amplitude * unreachable

    projection_controls = np.linalg.lstsq(response_interior, target, rcond=None)[0]
    projection_residual = response_interior @ projection_controls - target
    orthogonality = response_interior.T @ unreachable
    target_rms = float(np.sqrt(np.mean(np.square(target))))
    projection_rms = float(np.sqrt(np.mean(np.square(projection_residual))))
    certificate = {
        "orthogonality_max_abs": float(np.max(np.abs(orthogonality))),
        "projection_control_error_max_abs": float(
            np.max(np.abs(projection_controls - teacher_controls))
        ),
        "unconstrained_projection_mae": float(np.mean(np.abs(projection_residual))),
        "unconstrained_projection_rmse": projection_rms,
        "unconstrained_projection_linf": float(np.max(np.abs(projection_residual))),
        "unconstrained_projection_rmse_fraction_of_target": projection_rms / target_rms,
    }
    return target, unreachable, certificate


def make_2d_model(cfg: Config) -> BellyModel:
    x = np.linspace(-1.0, 1.0, cfg.grid_2d, dtype=np.float64)
    interior_indices = np.arange(1, cfg.grid_2d - 1, dtype=np.int64)
    x_interior = x[interior_indices]
    envelope = np.cos(0.5 * np.pi * x) ** 2
    rest_height = 0.05 + 0.18 * envelope

    centers = np.asarray([-0.55, 0.0, 0.55], dtype=np.float64)[:, None]
    width = 0.26
    loads = np.column_stack(
        [
            np.exp(-0.5 * np.square((x_interior - center) / width))
            for center in centers[:, 0]
        ]
    )
    laplacian = graph_laplacian_1d(x_interior.size)
    response_interior = normalized_responses(
        equilibrium_operator(laplacian, cfg), loads, cfg.actuator_scale
    )
    response = np.zeros((cfg.grid_2d, centers.shape[0]), dtype=np.float64)
    response[interior_indices] = response_interior

    teacher_controls = np.asarray([0.45, 0.80, 0.35], dtype=np.float64)
    narrow_seed = -np.exp(-0.5 * np.square(x_interior / 0.115))
    target_interior, unreachable_interior, certificate = make_unreachable_target(
        response_interior,
        narrow_seed,
        teacher_controls,
        cfg.unreachable_amplitude_2d,
    )
    target_displacement = np.zeros(cfg.grid_2d, dtype=np.float64)
    target_displacement[interior_indices] = target_interior
    unreachable = np.zeros(cfg.grid_2d, dtype=np.float64)
    unreachable[interior_indices] = unreachable_interior

    rank = int(np.linalg.matrix_rank(response_interior))
    metadata: dict[str, Any] = {
        "dimension": "2d",
        "interpretation": "fixed-rim belly profile embedded in the x-height plane",
        "grid_shape": [cfg.grid_2d],
        "n_points": int(cfg.grid_2d),
        "n_interior_points": int(interior_indices.size),
        "n_controls": int(centers.shape[0]),
        "response_rank": rank,
        "actuator_centers": centers[:, 0].tolist(),
        "actuator_width": width,
        "teacher_controls": teacher_controls.tolist(),
        "unreachable_amplitude": cfg.unreachable_amplitude_2d,
        "rest_height_min": float(rest_height.min()),
        "rest_height_max": float(rest_height.max()),
        "target_height_min": float((rest_height + target_displacement).min()),
        "target_height_max": float((rest_height + target_displacement).max()),
        **certificate,
    }
    return BellyModel(
        dimension="2d",
        grid_shape=(cfg.grid_2d,),
        x=x,
        z=None,
        rest_height=rest_height,
        response=response,
        target_displacement=target_displacement,
        interior_indices=interior_indices,
        actuator_centers=centers,
        teacher_controls=teacher_controls,
        unreachable_component=unreachable,
        metadata=metadata,
    )


def make_3d_model(cfg: Config) -> BellyModel:
    x = np.linspace(-1.0, 1.0, cfg.grid_3d, dtype=np.float64)
    z = np.linspace(-1.0, 1.0, cfg.grid_3d, dtype=np.float64)
    xx, zz = np.meshgrid(x, z, indexing="ij")
    envelope = np.square(np.cos(0.5 * np.pi * xx)) * np.square(np.cos(0.5 * np.pi * zz))
    rest_height = (0.05 + 0.18 * envelope).reshape(-1)

    interior_axis = np.arange(1, cfg.grid_3d - 1, dtype=np.int64)
    ii, jj = np.meshgrid(interior_axis, interior_axis, indexing="ij")
    interior_indices = (ii * cfg.grid_3d + jj).reshape(-1)
    x_interior = xx.reshape(-1)[interior_indices]
    z_interior = zz.reshape(-1)[interior_indices]

    center_axis = (-0.55, 0.0, 0.55)
    centers = np.asarray(
        [(cx, cz) for cx in center_axis for cz in center_axis], dtype=np.float64
    )
    width = 0.31
    loads = np.column_stack(
        [
            np.exp(
                -0.5
                * (
                    np.square((x_interior - cx) / width)
                    + np.square((z_interior - cz) / width)
                )
            )
            for cx, cz in centers
        ]
    )

    lap_axis = graph_laplacian_1d(cfg.grid_3d - 2)
    identity_axis = sp.eye(cfg.grid_3d - 2, format="csr")
    laplacian = sp.kron(lap_axis, identity_axis, format="csr") + sp.kron(
        identity_axis, lap_axis, format="csr"
    )
    response_interior = normalized_responses(
        equilibrium_operator(laplacian, cfg), loads, cfg.actuator_scale
    )
    response = np.zeros((cfg.grid_3d**2, centers.shape[0]), dtype=np.float64)
    response[interior_indices] = response_interior

    teacher_controls = np.asarray(
        [0.30, 0.45, 0.25, 0.50, 0.85, 0.55, 0.20, 0.40, 0.30],
        dtype=np.float64,
    )
    narrow_seed = -np.exp(
        -0.5 * (np.square(x_interior / 0.15) + np.square((z_interior + 0.05) / 0.12))
    )
    target_interior, unreachable_interior, certificate = make_unreachable_target(
        response_interior,
        narrow_seed,
        teacher_controls,
        cfg.unreachable_amplitude_3d,
    )
    target_displacement = np.zeros(cfg.grid_3d**2, dtype=np.float64)
    target_displacement[interior_indices] = target_interior
    unreachable = np.zeros(cfg.grid_3d**2, dtype=np.float64)
    unreachable[interior_indices] = unreachable_interior

    rank = int(np.linalg.matrix_rank(response_interior))
    metadata: dict[str, Any] = {
        "dimension": "3d",
        "interpretation": "fixed-rim belly height surface embedded in x-z-height space",
        "grid_shape": [cfg.grid_3d, cfg.grid_3d],
        "n_points": int(cfg.grid_3d**2),
        "n_interior_points": int(interior_indices.size),
        "n_controls": int(centers.shape[0]),
        "response_rank": rank,
        "actuator_centers": centers.tolist(),
        "actuator_width": width,
        "teacher_controls": teacher_controls.reshape(3, 3).tolist(),
        "unreachable_amplitude": cfg.unreachable_amplitude_3d,
        "rest_height_min": float(rest_height.min()),
        "rest_height_max": float(rest_height.max()),
        "target_height_min": float((rest_height + target_displacement).min()),
        "target_height_max": float((rest_height + target_displacement).max()),
        **certificate,
    }
    return BellyModel(
        dimension="3d",
        grid_shape=(cfg.grid_3d, cfg.grid_3d),
        x=x,
        z=z,
        rest_height=rest_height,
        response=response,
        target_displacement=target_displacement,
        interior_indices=interior_indices,
        actuator_centers=centers,
        teacher_controls=teacher_controls,
        unreachable_component=unreachable,
        metadata=metadata,
    )


def numpy_metrics(residual: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "linf": float(np.max(np.abs(residual))),
    }


def metric_objective(metrics: dict[str, float], loss_name: LossName) -> float:
    return metrics[LOSS_METRIC_KEYS[loss_name]]


def solve_reference_optimum(
    model: BellyModel, loss_name: LossName, cfg: Config
) -> ReferenceOptimum:
    response = model.response[model.interior_indices]
    target = model.target_displacement[model.interior_indices]
    n_controls = response.shape[1]
    bounds = (cfg.control_min, cfg.control_max)

    if loss_name == "l2":
        solved = spo.lsq_linear(
            response,
            target,
            bounds=bounds,
            tol=1.0e-14,
            lsmr_tol=1.0e-14,
            max_iter=10_000,
        )
        if not solved.success:
            raise RuntimeError(f"bounded L2 reference solve failed: {solved.message}")
        controls = np.asarray(solved.x, dtype=np.float64)
        solver = "scipy.optimize.lsq_linear"
        status = str(solved.message)
    elif loss_name == "l1":
        n_points = response.shape[0]
        objective = np.concatenate(
            (np.zeros(n_controls), np.full(n_points, 1.0 / n_points))
        )
        identity = sp.eye(n_points, format="csr")
        constraints = sp.vstack(
            (
                sp.hstack((sp.csr_matrix(response), -identity), format="csr"),
                sp.hstack((sp.csr_matrix(-response), -identity), format="csr"),
            ),
            format="csr",
        )
        rhs = np.concatenate((target, -target))
        variable_bounds = [bounds] * n_controls + [(0.0, None)] * n_points
        solved = spo.linprog(
            objective,
            A_ub=constraints,
            b_ub=rhs,
            bounds=variable_bounds,
            method="highs",
        )
        if not solved.success:
            raise RuntimeError(f"bounded L1 reference solve failed: {solved.message}")
        controls = np.asarray(solved.x[:n_controls], dtype=np.float64)
        solver = "scipy.optimize.linprog-highs"
        status = str(solved.message)
    else:
        objective = np.concatenate((np.zeros(n_controls), np.ones(1)))
        minus_radius = -np.ones((response.shape[0], 1), dtype=np.float64)
        constraints = np.vstack(
            (
                np.hstack((response, minus_radius)),
                np.hstack((-response, minus_radius)),
            )
        )
        rhs = np.concatenate((target, -target))
        variable_bounds = [bounds] * n_controls + [(0.0, None)]
        solved = spo.linprog(
            objective,
            A_ub=constraints,
            b_ub=rhs,
            bounds=variable_bounds,
            method="highs",
        )
        if not solved.success:
            raise RuntimeError(
                f"bounded L-inf reference solve failed: {solved.message}"
            )
        controls = np.asarray(solved.x[:n_controls], dtype=np.float64)
        solver = "scipy.optimize.linprog-highs"
        status = str(solved.message)

    displacement = model.response @ controls
    residual = displacement[model.interior_indices] - target
    metrics = numpy_metrics(residual)
    return ReferenceOptimum(
        loss_name=loss_name,
        controls=controls,
        displacement=displacement,
        metrics=metrics,
        objective=metric_objective(metrics, loss_name),
        solver=solver,
        solver_status=status,
    )


def torch_metrics(residual: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "mae": residual.abs().mean(),
        "rmse": residual.square().mean().sqrt(),
        "linf": residual.abs().max(),
    }


def optimize_case(
    model: BellyModel,
    loss_name: LossName,
    cfg: Config,
    case_index: int,
) -> CaseResult:
    response = torch.as_tensor(
        model.response[model.interior_indices], dtype=torch.float64, device="cpu"
    )
    target = torch.as_tensor(
        model.target_displacement[model.interior_indices],
        dtype=torch.float64,
        device="cpu",
    )
    controls = torch.nn.Parameter(
        torch.full((model.n_controls,), cfg.initial_control, dtype=torch.float64)
    )
    optimizer = torch.optim.Adam([controls], lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.steps, eta_min=cfg.final_learning_rate
    )
    reference = solve_reference_optimum(model, loss_name, cfg)

    steps: list[int] = []
    control_history: list[np.ndarray] = []
    displacement_history: list[np.ndarray] = []
    objective_history: list[float] = []
    mae_history: list[float] = []
    rmse_history: list[float] = []
    linf_history: list[float] = []
    grad_history: list[float] = []
    lr_history: list[float] = []
    worst_history: list[int] = []
    ties_history: list[int] = []
    best_index = 0
    best_objective = math.inf

    for step in range(cfg.steps + 1):
        optimizer.zero_grad()
        displacement_interior = response @ controls
        residual = displacement_interior - target
        metrics = torch_metrics(residual)
        objective = metrics[LOSS_METRIC_KEYS[loss_name]]
        objective.backward()
        if controls.grad is None or not torch.isfinite(controls.grad).all():
            raise FloatingPointError(
                f"{model.dimension}/{loss_name}: non-finite or missing gradient at {step}"
            )

        controls_np = controls.detach().numpy().copy()
        displacement_full = model.response @ controls_np
        objective_value = float(objective.detach())
        abs_residual = np.abs(residual.detach().numpy())
        worst_local = int(np.argmax(abs_residual))
        worst_value = float(abs_residual[worst_local])
        worst_ties = int(
            np.count_nonzero(
                np.isclose(abs_residual, worst_value, rtol=1.0e-10, atol=1.0e-12)
            )
        )

        steps.append(step)
        control_history.append(controls_np)
        displacement_history.append(displacement_full)
        objective_history.append(objective_value)
        mae_history.append(float(metrics["mae"].detach()))
        rmse_history.append(float(metrics["rmse"].detach()))
        linf_history.append(float(metrics["linf"].detach()))
        grad_history.append(float(torch.linalg.vector_norm(controls.grad).detach()))
        lr_history.append(float(optimizer.param_groups[0]["lr"]))
        worst_history.append(int(model.interior_indices[worst_local]))
        ties_history.append(worst_ties)

        if objective_value < best_objective:
            best_objective = objective_value
            best_index = step

        if step % cfg.snapshot_every == 0 or step == cfg.steps:
            global_step = case_index * (cfg.steps + 1) + step
            cherries.set_step(global_step)
            prefix = f"{model.dimension}/{loss_name}"
            cherries.log_metrics(
                {
                    f"{prefix}/objective": objective_value,
                    f"{prefix}/mae": mae_history[-1],
                    f"{prefix}/rmse": rmse_history[-1],
                    f"{prefix}/linf": linf_history[-1],
                    f"{prefix}/grad_norm": grad_history[-1],
                    f"{prefix}/learning_rate": lr_history[-1],
                }
            )
        if step % max(1, cfg.steps // 12) == 0 or step == cfg.steps:
            logger.info(
                "%s/%s step %04d objective %.7g mae %.7g rmse %.7g linf %.7g",
                model.dimension,
                loss_name,
                step,
                objective_value,
                mae_history[-1],
                rmse_history[-1],
                linf_history[-1],
            )

        if step < cfg.steps:
            optimizer.step()
            with torch.no_grad():
                controls.clamp_(cfg.control_min, cfg.control_max)
            scheduler.step()

    steps_np = np.asarray(steps, dtype=np.int64)
    controls_np = np.stack(control_history)
    displacements_np = np.stack(displacement_history)
    objectives_np = np.asarray(objective_history, dtype=np.float64)
    mae_np = np.asarray(mae_history, dtype=np.float64)
    rmse_np = np.asarray(rmse_history, dtype=np.float64)
    linf_np = np.asarray(linf_history, dtype=np.float64)
    grad_np = np.asarray(grad_history, dtype=np.float64)
    lr_np = np.asarray(lr_history, dtype=np.float64)
    worst_np = np.asarray(worst_history, dtype=np.int64)
    ties_np = np.asarray(ties_history, dtype=np.int64)

    best_controls = controls_np[best_index]
    best_metrics = {
        "mae": float(mae_np[best_index]),
        "rmse": float(rmse_np[best_index]),
        "linf": float(linf_np[best_index]),
    }
    final_metrics = {
        "mae": float(mae_np[-1]),
        "rmse": float(rmse_np[-1]),
        "linf": float(linf_np[-1]),
    }
    initial_metrics = {
        "mae": float(mae_np[0]),
        "rmse": float(rmse_np[0]),
        "linf": float(linf_np[0]),
    }
    objective_gap = max(0.0, best_objective - reference.objective)
    relative_gap = objective_gap / max(reference.objective, 1.0e-15)
    worst_switches = int(np.count_nonzero(np.diff(worst_np)))
    bound_tolerance = 1.0e-8
    bound_hits = int(
        np.count_nonzero(
            (best_controls <= cfg.control_min + bound_tolerance)
            | (best_controls >= cfg.control_max - bound_tolerance)
        )
    )
    summary: dict[str, Any] = {
        "dimension": model.dimension,
        "loss": loss_name,
        "loss_definition": LOSS_LABELS[loss_name],
        "optimizer": "projected Adam with cosine learning-rate decay",
        "evaluations": int(cfg.steps + 1),
        "updates": int(cfg.steps),
        "initial": {
            "step": 0,
            "objective": float(objectives_np[0]),
            **initial_metrics,
            "controls": controls_np[0].tolist(),
        },
        "best": {
            "step": int(best_index),
            "objective": float(best_objective),
            **best_metrics,
            "controls": best_controls.tolist(),
            "bound_hits": bound_hits,
        },
        "final": {
            "step": int(steps_np[-1]),
            "objective": float(objectives_np[-1]),
            **final_metrics,
            "controls": controls_np[-1].tolist(),
        },
        "reference": {
            "objective": float(reference.objective),
            **reference.metrics,
            "controls": reference.controls.tolist(),
            "solver": reference.solver,
            "solver_status": reference.solver_status,
        },
        "optimizer_gap_to_reference": {
            "absolute": float(objective_gap),
            "relative": float(relative_gap),
        },
        "worst_point_switches": worst_switches,
        "worst_point_ties_max": int(ties_np.max()),
        "validation": {
            "finite_history": bool(
                np.isfinite(
                    np.concatenate(
                        (
                            controls_np.reshape(-1),
                            objectives_np,
                            mae_np,
                            rmse_np,
                            linf_np,
                            grad_np,
                        )
                    )
                ).all()
            ),
            "best_improves_initial": bool(best_objective < objectives_np[0]),
            "best_not_below_reference": bool(
                best_objective >= reference.objective - 1.0e-9
            ),
            "controls_within_bounds": bool(
                np.all(controls_np >= cfg.control_min - 1.0e-12)
                and np.all(controls_np <= cfg.control_max + 1.0e-12)
            ),
            "target_residual_nonzero": bool(best_metrics["linf"] > 1.0e-8),
        },
    }
    return CaseResult(
        dimension=model.dimension,
        loss_name=loss_name,
        steps=steps_np,
        controls=controls_np,
        displacements=displacements_np,
        objectives=objectives_np,
        mae=mae_np,
        rmse=rmse_np,
        linf=linf_np,
        grad_norm=grad_np,
        learning_rate=lr_np,
        worst_index=worst_np,
        worst_ties=ties_np,
        best_index=best_index,
        reference=reference,
        summary=summary,
    )


def set_plot_style() -> None:
    plt.style.use("default")
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "#F8FAFC",
            "axes.edgecolor": "#475569",
            "axes.labelcolor": "#0F172A",
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "grid.color": "#94A3B8",
            "grid.alpha": 0.28,
            "font.size": 10,
        }
    )


def frame_indices(result: CaseResult, cfg: Config) -> list[int]:
    indices = set(range(0, cfg.steps + 1, cfg.snapshot_every))
    indices.update((0, result.best_index, cfg.steps))
    return sorted(indices)


def residual_plot_limit(model: BellyModel, result: CaseResult) -> float:
    initial_height = model.rest_height + result.displacements[0]
    return max(float(np.max(np.abs(initial_height - model.target_height))), 1.0e-6)


def configure_2d_axes(
    height_ax: plt.Axes,
    residual_ax: plt.Axes,
    model: BellyModel,
    height: np.ndarray,
    result: CaseResult,
    step_index: int,
    residual_limit: float,
) -> None:
    height_ax.clear()
    residual_ax.clear()
    residual = height - model.target_height
    height_ax.plot(
        model.x,
        model.rest_height,
        color="#64748B",
        linestyle=":",
        linewidth=1.7,
        label="rest",
    )
    height_ax.plot(
        model.x,
        model.target_height,
        color=TARGET_COLOR,
        linestyle="--",
        linewidth=2.3,
        label="unreachable target",
    )
    height_ax.plot(
        model.x,
        height,
        color=LOSS_COLORS[result.loss_name],
        linewidth=2.8,
        label=f"current ({LOSS_LABELS[result.loss_name]})",
    )
    height_ax.fill_between(
        model.x,
        model.target_height,
        height,
        color=LOSS_COLORS[result.loss_name],
        alpha=0.10,
    )
    all_heights = np.concatenate((model.rest_height, model.target_height))
    pad = 0.08 * float(np.ptp(all_heights))
    height_ax.set_xlim(-1.03, 1.03)
    height_ax.set_ylim(float(all_heights.min() - pad), float(all_heights.max() + pad))
    height_ax.set_ylabel("belly height (dimensionless)")
    height_ax.grid(True)
    height_ax.legend(loc="upper right", frameon=True)
    height_ax.set_title(
        f"2D fixed-rim linear belly | {LOSS_LABELS[result.loss_name]} | "
        f"step {int(result.steps[step_index])}\n"
        "orthogonally unreachable target | "
        f"MAE {result.mae[step_index]:.5f}  RMS {result.rmse[step_index]:.5f}  "
        f"max {result.linf[step_index]:.5f}"
    )
    residual_ax.plot(
        model.x, residual, color=LOSS_COLORS[result.loss_name], linewidth=1.8
    )
    residual_ax.fill_between(
        model.x,
        np.zeros_like(residual),
        residual,
        color=LOSS_COLORS[result.loss_name],
        alpha=0.16,
    )
    residual_ax.axhline(0.0, color="#475569", linewidth=0.9)
    residual_ax.set_xlim(-1.03, 1.03)
    residual_ax.set_ylim(-1.05 * residual_limit, 1.05 * residual_limit)
    residual_ax.set_xlabel("lateral coordinate x (dimensionless)")
    residual_ax.set_ylabel("current - target")
    residual_ax.set_title("signed height residual (shared scale across losses)")
    residual_ax.grid(True)


def configure_3d_axes(
    surface_ax: Any,
    residual_ax: plt.Axes,
    model: BellyModel,
    height: np.ndarray,
    result: CaseResult,
    step_index: int,
    residual_limit: float,
) -> None:
    surface_ax.clear()
    residual_ax.clear()
    assert model.z is not None
    xx, zz = np.meshgrid(model.x, model.z, indexing="ij")
    current = height.reshape(model.grid_shape)
    target = model.target_height.reshape(model.grid_shape)
    residual = current - target
    surface_ax.plot_surface(
        xx,
        zz,
        current,
        color=LOSS_COLORS[result.loss_name],
        alpha=0.82,
        linewidth=0.15,
        edgecolor="#E2E8F0",
        antialiased=True,
    )
    surface_ax.plot_wireframe(
        xx,
        zz,
        target,
        rstride=2,
        cstride=2,
        color=TARGET_COLOR,
        linewidth=0.65,
        alpha=0.9,
    )
    all_heights = np.concatenate((model.rest_height, model.target_height))
    pad = 0.08 * float(np.ptp(all_heights))
    surface_ax.set_xlim(-1.0, 1.0)
    surface_ax.set_ylim(-1.0, 1.0)
    surface_ax.set_zlim(float(all_heights.min() - pad), float(all_heights.max() + pad))
    surface_ax.set_xlabel("x")
    surface_ax.set_ylabel("z")
    surface_ax.set_zlabel("height", labelpad=1)
    surface_ax.tick_params(labelsize=8, pad=1)
    surface_ax.view_init(elev=27.0, azim=-56.0)
    surface_ax.set_box_aspect((1.0, 1.0, 0.38))
    surface_ax.legend(
        handles=[
            Patch(
                facecolor=LOSS_COLORS[result.loss_name],
                alpha=0.82,
                label="current surface",
            ),
            Line2D(
                [0],
                [0],
                color=TARGET_COLOR,
                linestyle="--",
                linewidth=1.6,
                label="unreachable target wireframe",
            ),
        ],
        loc="upper right",
        fontsize=8,
    )
    surface_ax.text2D(
        0.02,
        0.02,
        "vertical display scale exaggerated for visibility",
        transform=surface_ax.transAxes,
        fontsize=7,
        color="#475569",
    )
    surface_ax.set_title(
        f"current vs target | {LOSS_LABELS[result.loss_name]} | "
        f"step {int(result.steps[step_index])}\n"
        f"MAE {result.mae[step_index]:.5f}  RMS {result.rmse[step_index]:.5f}  "
        f"max {result.linf[step_index]:.5f}",
        fontsize=10,
    )
    residual_ax.pcolormesh(
        xx,
        zz,
        residual,
        shading="auto",
        cmap="coolwarm",
        vmin=-residual_limit,
        vmax=residual_limit,
    )
    residual_ax.set_aspect("equal")
    residual_ax.set_xlabel("x (dimensionless)")
    residual_ax.set_ylabel("z (dimensionless)")
    residual_ax.set_title(
        "signed height residual\nblue: below target; red: above target"
    )


def render_case(
    model: BellyModel, result: CaseResult, case_dir: Path, cfg: Config
) -> dict[str, str]:
    set_plot_style()
    case_dir.mkdir(parents=True, exist_ok=False)
    best_height = model.rest_height + result.displacements[result.best_index]
    residual_limit = residual_plot_limit(model, result)
    best_png = case_dir / "best.png"
    if model.dimension == "2d":
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(8.0, 6.4),
            constrained_layout=True,
            sharex=True,
            gridspec_kw={"height_ratios": (3.0, 1.0)},
        )
        configure_2d_axes(
            axes[0],
            axes[1],
            model,
            best_height,
            result,
            result.best_index,
            residual_limit,
        )
    else:
        fig = plt.figure(figsize=(12.4, 5.6), constrained_layout=True)
        fig.suptitle(
            "3D fixed-rim linear belly with orthogonally unreachable target "
            "(dimensionless coordinates)"
        )
        surface_ax = fig.add_subplot(121, projection="3d")
        residual_ax = fig.add_subplot(122)
        configure_3d_axes(
            surface_ax,
            residual_ax,
            model,
            best_height,
            result,
            result.best_index,
            residual_limit,
        )
        scalar_mappable = mpl.cm.ScalarMappable(
            norm=mpl.colors.Normalize(-residual_limit, residual_limit),
            cmap="coolwarm",
        )
        fig.colorbar(
            scalar_mappable,
            ax=residual_ax,
            shrink=0.72,
            label="current - target (dimensionless height)",
        )
    fig.savefig(best_png, dpi=180)
    plt.close(fig)

    video_path = case_dir / "evolution.mp4"
    if cfg.render_video:
        if not animation.writers.is_available("ffmpeg"):
            raise RuntimeError("Matplotlib cannot find the ffmpeg animation writer")
        if model.dimension == "2d":
            fig, axes = plt.subplots(
                2,
                1,
                figsize=(8.0, 6.4),
                constrained_layout=True,
                sharex=True,
                gridspec_kw={"height_ratios": (3.0, 1.0)},
            )
        else:
            fig = plt.figure(figsize=(12.4, 5.6), constrained_layout=True)
            fig.suptitle(
                "3D fixed-rim linear belly with orthogonally unreachable target "
                "(dimensionless coordinates)"
            )
            surface_ax = fig.add_subplot(121, projection="3d")
            residual_ax = fig.add_subplot(122)
            scalar_mappable = mpl.cm.ScalarMappable(
                norm=mpl.colors.Normalize(-residual_limit, residual_limit),
                cmap="coolwarm",
            )
            fig.colorbar(
                scalar_mappable,
                ax=residual_ax,
                shrink=0.72,
                label="current - target (dimensionless height)",
            )
        writer = animation.FFMpegWriter(
            fps=cfg.video_fps,
            codec="libx264",
            bitrate=1800,
            metadata={
                "title": f"{model.dimension} {LOSS_LABELS[result.loss_name]} optimization",
                "artist": "liblaf.apple experiment",
            },
            extra_args=["-pix_fmt", "yuv420p"],
        )
        with writer.saving(fig, str(video_path), dpi=120):
            for index in frame_indices(result, cfg):
                height = model.rest_height + result.displacements[index]
                if model.dimension == "2d":
                    configure_2d_axes(
                        axes[0],
                        axes[1],
                        model,
                        height,
                        result,
                        index,
                        residual_limit,
                    )
                else:
                    configure_3d_axes(
                        surface_ax,
                        residual_ax,
                        model,
                        height,
                        result,
                        index,
                        residual_limit,
                    )
                writer.grab_frame()
        plt.close(fig)

    control_png = case_dir / "control-evolution.png"
    fig, ax = plt.subplots(figsize=(8.0, 4.8), constrained_layout=True)
    for control_id in range(model.n_controls):
        center = model.actuator_centers[control_id]
        if model.dimension == "2d":
            control_label = f"a{control_id} (x={center[0]:+.2f})"
        else:
            control_label = f"a{control_id} (x,z={center[0]:+.2f},{center[1]:+.2f})"
        ax.plot(
            result.steps,
            result.controls[:, control_id],
            linewidth=1.25,
            alpha=0.88,
            label=control_label,
        )
    ax.axhline(cfg.control_min, color="#64748B", linestyle=":", linewidth=1.0)
    ax.axhline(cfg.control_max, color="#64748B", linestyle=":", linewidth=1.0)
    ax.set_xlabel("optimization step")
    ax.set_ylabel("bounded actuator control")
    ax.set_ylim(cfg.control_min - 0.04, cfg.control_max + 0.04)
    ax.set_title(f"{model.dimension.upper()} {LOSS_LABELS[result.loss_name]} controls")
    ax.grid(True)
    ax.legend(ncol=3, fontsize=8, loc="best")
    fig.savefig(control_png, dpi=180)
    plt.close(fig)

    artifacts = {
        "best_png": str(best_png),
        "control_evolution_png": str(control_png),
    }
    if cfg.render_video:
        artifacts["evolution_mp4"] = str(video_path)
    return artifacts


def write_case_data(
    model: BellyModel,
    result: CaseResult,
    case_dir: Path,
    artifacts: dict[str, str],
) -> dict[str, str]:
    csv_path = case_dir / "trace.csv"
    control_names = [f"control_{i}" for i in range(model.n_controls)]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = [
            "step",
            "objective",
            "mae",
            "rmse",
            "linf",
            "grad_norm",
            "learning_rate",
            "worst_point_index",
            "worst_point_ties",
            *control_names,
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index, step in enumerate(result.steps):
            row: dict[str, Any] = {
                "step": int(step),
                "objective": float(result.objectives[index]),
                "mae": float(result.mae[index]),
                "rmse": float(result.rmse[index]),
                "linf": float(result.linf[index]),
                "grad_norm": float(result.grad_norm[index]),
                "learning_rate": float(result.learning_rate[index]),
                "worst_point_index": int(result.worst_index[index]),
                "worst_point_ties": int(result.worst_ties[index]),
            }
            row.update(
                {
                    name: float(value)
                    for name, value in zip(
                        control_names, result.controls[index], strict=True
                    )
                }
            )
            writer.writerow(row)

    history_path = case_dir / "history.npz"
    np.savez_compressed(
        history_path,
        steps=result.steps,
        x=model.x,
        z=np.asarray([]) if model.z is None else model.z,
        rest_height=model.rest_height,
        target_displacement=model.target_displacement,
        target_height=model.target_height,
        unreachable_component=model.unreachable_component,
        response=model.response,
        controls=result.controls,
        displacements=result.displacements,
        heights=model.rest_height[None, :] + result.displacements,
        objective=result.objectives,
        mae=result.mae,
        rmse=result.rmse,
        linf=result.linf,
        grad_norm=result.grad_norm,
        learning_rate=result.learning_rate,
        worst_index=result.worst_index,
        worst_ties=result.worst_ties,
        reference_controls=result.reference.controls,
        reference_displacement=result.reference.displacement,
    )

    summary_path = case_dir / "summary.json"
    summary = dict(result.summary)
    summary["artifacts"] = {
        "trace_csv": str(csv_path),
        "history_npz": str(history_path),
        **artifacts,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {
        "summary_json": str(summary_path),
        "trace_csv": str(csv_path),
        "history_npz": str(history_path),
        **artifacts,
    }


def plot_metric_comparison(
    dimension: DimensionName,
    results: list[CaseResult],
    output_dir: Path,
) -> tuple[Path, Path]:
    set_plot_style()
    metric_specs = (("mae", "MAE"), ("rmse", "RMS error"), ("linf", "max error"))
    paths: list[Path] = []
    for suffix, step_limit in (("", None), ("-early", 100)):
        fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4), constrained_layout=True)
        for ax, (attribute, label) in zip(axes, metric_specs, strict=True):
            for result in results:
                values = getattr(result, attribute)
                ax.plot(
                    result.steps,
                    values,
                    color=LOSS_COLORS[result.loss_name],
                    linewidth=1.8,
                    label=f"optimize {LOSS_LABELS[result.loss_name]}",
                )
                ax.axhline(
                    result.reference.metrics[attribute],
                    color=LOSS_COLORS[result.loss_name],
                    linestyle=":",
                    linewidth=0.9,
                    alpha=0.75,
                )
            if step_limit is not None:
                ax.set_xlim(0, step_limit)
            ax.set_xlabel("optimization step")
            ax.set_ylabel(label)
            ax.set_yscale("log")
            ax.grid(True, which="both")
        axes[0].legend(fontsize=8, loc="best")
        window = " (first 100 steps)" if step_limit is not None else ""
        fig.suptitle(
            f"{dimension.upper()} fixed-rim linear belly: common metrics{window}\n"
            "solid = Adam trace; dotted = displayed metric at that loss's exact "
            "bounded reference solution"
        )
        path = output_dir / f"metric-comparison{suffix}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths[0], paths[1]


def model_artifact(model: BellyModel, output_dir: Path) -> Path:
    path = output_dir / "model.npz"
    np.savez_compressed(
        path,
        x=model.x,
        z=np.asarray([]) if model.z is None else model.z,
        rest_height=model.rest_height,
        response=model.response,
        target_displacement=model.target_displacement,
        target_height=model.target_height,
        interior_indices=model.interior_indices,
        actuator_centers=model.actuator_centers,
        teacher_controls=model.teacher_controls,
        unreachable_component=model.unreachable_component,
    )
    return path


def make_table(cases: list[dict[str, Any]]) -> str:
    lines = [
        "| model | optimized loss | best step | objective | MAE | RMS | max | exact optimum | relative gap | worst-point switches |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in cases:
        best = case["best"]
        reference = case["reference"]
        lines.append(
            "| {dimension} | {loss} | {step} | {objective:.7g} | {mae:.7g} | "
            "{rmse:.7g} | {linf:.7g} | {reference:.7g} | {gap:.3%} | {switches} |".format(
                dimension=case["dimension"],
                loss=case["loss"],
                step=best["step"],
                objective=best["objective"],
                mae=best["mae"],
                rmse=best["rmse"],
                linf=best["linf"],
                reference=reference["objective"],
                gap=case["optimizer_gap_to_reference"]["relative"],
                switches=case["worst_point_switches"],
            )
        )
    return "\n".join(lines) + "\n"


def compare_cases(results: list[CaseResult]) -> dict[str, Any]:
    by_loss = {result.loss_name: result for result in results}
    endpoints = {
        loss: {
            "mae": float(result.mae[result.best_index]),
            "rmse": float(result.rmse[result.best_index]),
            "linf": float(result.linf[result.best_index]),
        }
        for loss, result in by_loss.items()
    }
    winners = {
        metric: min(endpoints, key=lambda loss: endpoints[loss][metric])
        for metric in ("mae", "rmse", "linf")
    }
    return {"best_state_metrics": endpoints, "winner_by_metric": winners}


def relative_artifact_paths(summary: dict[str, Any], output_dir: Path) -> None:
    for case in summary["cases"]:
        case["artifacts"] = {
            key: str(Path(value).relative_to(output_dir))
            for key, value in case["artifacts"].items()
        }
    for model in summary["models"].values():
        model["artifacts"] = {
            key: str(Path(value).relative_to(output_dir))
            for key, value in model["artifacts"].items()
        }


def main(cfg: Config) -> None:
    validate_config(cfg)
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)

    models = (make_2d_model(cfg), make_3d_model(cfg))
    for model in models:
        projection_floor = float(model.metadata["unconstrained_projection_rmse"])
        if projection_floor <= 1.0e-8:
            raise AssertionError(
                f"{model.dimension} target is not demonstrably outside the response space"
            )
        if model.metadata["response_rank"] != model.n_controls:
            raise AssertionError(
                f"{model.dimension} actuator responses are rank deficient"
            )

    case_summaries: list[dict[str, Any]] = []
    model_summaries: dict[str, Any] = {}
    case_index = 0
    for model in models:
        dimension_dir = cfg.output_dir / model.dimension
        dimension_dir.mkdir(parents=True, exist_ok=False)
        model_path = model_artifact(model, dimension_dir)
        dimension_results: list[CaseResult] = []
        for loss_name_value in LOSS_NAMES:
            loss_name: LossName = loss_name_value  # pyright: ignore[reportAssignmentType]
            result = optimize_case(model, loss_name, cfg, case_index)
            case_index += 1
            case_dir = dimension_dir / loss_name
            artifacts = render_case(model, result, case_dir, cfg)
            artifacts = write_case_data(model, result, case_dir, artifacts)
            case_summary = dict(result.summary)
            case_summary["artifacts"] = artifacts
            case_summaries.append(case_summary)
            dimension_results.append(result)
        comparison_path, early_comparison_path = plot_metric_comparison(
            model.dimension, dimension_results, dimension_dir
        )
        model_summaries[model.dimension] = {
            **model.metadata,
            "comparison": compare_cases(dimension_results),
            "artifacts": {
                "model_npz": str(model_path),
                "metric_comparison_png": str(comparison_path),
                "metric_comparison_early_png": str(early_comparison_path),
            },
        }

    validation_cases = [
        check for case in case_summaries for check in case["validation"].values()
    ]
    expected_winners = {
        dimension: model_summaries[dimension]["comparison"]["winner_by_metric"]
        for dimension in ("2d", "3d")
    }
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "design": DESIGN,
        "status": "ok",
        "complete": True,
        "scope": (
            "reduced fixed-rim linear membrane surrogate; not volumetric FEM or "
            "a calibrated anatomical tissue model"
        ),
        "loss_definitions": {
            "l1": "mean(abs(surface-height residual))",
            "l2": "sqrt(mean(square(surface-height residual)))",
            "linf": "max(abs(surface-height residual))",
        },
        "target_construction": (
            "reachable teacher response plus a normalized narrow component projected "
            "orthogonal to the actuator response space"
        ),
        "config": cfg.model_dump(mode="json"),
        "models": model_summaries,
        "cases": case_summaries,
        "validation": {
            "all_case_checks_pass": bool(all(validation_cases)),
            "case_checks_total": len(validation_cases),
            "case_checks_passed": int(sum(validation_cases)),
            "expected_metric_winners": expected_winners,
            "each_loss_wins_its_metric": bool(
                all(
                    winners == {"mae": "l1", "rmse": "l2", "linf": "linf"}
                    for winners in expected_winners.values()
                )
            ),
        },
    }
    relative_artifact_paths(summary, cfg.output_dir)
    summary_path = cfg.output_dir / "summary.json"
    table_path = cfg.output_dir / "results.md"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    table_path.write_text(make_table(summary["cases"]), encoding="utf-8")

    if not summary["validation"]["all_case_checks_pass"]:
        raise AssertionError("one or more optimization validation checks failed")
    logger.info("Wrote %s", summary_path)
    logger.info("Wrote %s", table_path)


if __name__ == "__main__":
    cherries.main(main)
