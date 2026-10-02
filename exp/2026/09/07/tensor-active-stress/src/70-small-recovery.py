# Copyright (c) 2026 liblaf
# ruff: noqa: C901, PLR0915
"""Recover a generated interior-PSD target on the validated two-tet fixture."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import logging
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pydantic_settings as ps
import tensor_controls as controls
import torch
from experiment_profile import ProfileCometNoCommit

from liblaf import cherries

LOG = logging.getLogger(__name__)
DTYPE = torch.float64
HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
SOURCE12 = HERE / "12-small-model-study.py"
SOURCE12_RECEIPT = EXPERIMENT / "data/12-small-model-study-v4/summary.json"
FACE_STEP64_FIT_GRADIENT_RMS = 4.783391513852665e-5
EPSILONS = (1.0e-2, 1.0e-4, 1.0e-6)
BASELINE_LEARNING_RATE = 0.3
CANDIDATE_STEP_MULTIPLIER = 2.0
COMPLETED = False


def load_script(name: str, path: Path) -> ModuleType:
    """Load a numbered source module without executing its CLI."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


study = load_script("validated_two_tet_study", SOURCE12)


class Config(cherries.BaseConfig):
    """Fixed-budget recovery settings."""

    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output_dir: Path = cherries.output("70-small-recovery", mkdir=True)
    muscle_E_mpa: float = 0.03
    fat_E_mpa: float = 0.003
    poisson_ratio: float = 0.49
    inner_maxiter: int = 400
    updates: int = 256
    recovery_recalibration_step: int = 4096
    recovery_max_updates: int = 8192
    recovery_relative_displacement_tolerance: float = 1.0e-3
    finite_difference_step: float = 1.0e-2


@dataclass(frozen=True)
class ImplicitResult:
    """One equilibrium loss and its dense implicit control gradient."""

    raw_mse: float
    scaled_loss: float
    gradient: np.ndarray
    equilibrium_gradient_inf: float
    hessian_eigen_min: float
    hessian_eigen_max: float
    hessian_condition: float


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, Any]:
    """Return an immutable file receipt."""
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def write_json(path: Path, value: Any) -> None:
    """Atomically write strict JSON."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def append_csv(path: Path, row: dict[str, Any]) -> None:
    """Append one durable trace row, writing the header on first use."""
    first = not path.exists()
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if first:
            writer.writeheader()
        writer.writerow(row)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    """Append one durable accepted-forward receipt."""
    with path.open("a") as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def rotation_x(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(((1, 0, 0), (0, cosine, -sine), (0, sine, cosine)))


def rotation_y(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(((cosine, 0, sine), (0, 1, 0), (-sine, 0, cosine)))


def rotation_z(angle: float) -> np.ndarray:
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.asarray(((cosine, -sine, 0), (sine, cosine, 0), (0, 0, 1)))


def generated_z_star() -> np.ndarray:
    """Return a fixed rotated stress field strictly inside 0 < Z < 10 I."""
    rotations = (
        rotation_z(math.radians(25)) @ rotation_y(math.radians(-15)),
        rotation_x(math.radians(-20)) @ rotation_z(math.radians(35)),
    )
    eigenvalues = ((0.7, 1.5, 2.4), (0.5, 1.2, 2.0))
    matrices = np.asarray(
        [
            rotation @ np.diag(values) @ rotation.T
            for rotation, values in zip(rotations, eigenvalues, strict=True)
        ]
    )
    return controls.coordinates(torch.as_tensor(matrices, dtype=DTYPE)).numpy()


def positions_from_free(free: torch.Tensor) -> torch.Tensor:
    """Insert nine free coordinates without detaching their graph."""
    base = torch.as_tensor(study.X0.ravel(), dtype=DTYPE)
    indices = torch.as_tensor(study.FREE, dtype=torch.int64)
    return base.index_copy(0, indices, free).reshape(study.X0.shape)


def differentiable_energy(
    free: torch.Tensor,
    z: torch.Tensor,
    fixture: Any,
    cfg: Config,
    q_ref_mpa: float,
) -> torch.Tensor:
    """Exact source-12 mixture energy with differentiable face-style controls."""
    x = positions_from_free(free)
    gradients = study.deformation_gradients(x, fixture)
    q_cells = q_ref_mpa * controls.matrices(z)
    identity = torch.eye(3, dtype=DTYPE)
    energy = torch.zeros((), dtype=DTYPE)
    for cell, deformation in enumerate(gradients):
        fraction = float(study.FRACTION[cell])
        mixture = fraction * study.passive_energy(
            deformation, cfg.muscle_E_mpa, cfg.poisson_ratio
        )
        mixture += (1 - fraction) * study.passive_energy(
            deformation, cfg.fat_E_mpa, cfg.poisson_ratio
        )
        mixture += (
            fraction
            * 0.5
            * torch.sum(q_cells[cell] * (deformation.T @ deformation - identity))
        )
        energy += fixture.volumes[cell] * mixture
    return energy


def implicit_gradient(
    x: np.ndarray,
    z: np.ndarray,
    target: np.ndarray,
    fixture: Any,
    cfg: Config,
    q_ref_mpa: float,
    loss_multiplier: float,
) -> ImplicitResult:
    """Differentiate the displacement loss through a 9-DOF equilibrium."""
    free = torch.tensor(x.ravel()[study.FREE], dtype=DTYPE, requires_grad=True)
    control = torch.tensor(z, dtype=DTYPE, requires_grad=True)
    energy = differentiable_energy(free, control, fixture, cfg, q_ref_mpa)
    (residual,) = torch.autograd.grad(energy, free, create_graph=True)
    hessian = torch.stack(
        [
            torch.autograd.grad(residual[index], free, retain_graph=True)[0]
            for index in range(len(free))
        ]
    )
    positions = positions_from_free(free)
    target_tensor = torch.as_tensor(target, dtype=DTYPE)
    raw_mse = torch.mean((positions - target_tensor).square())
    loss = loss_multiplier * raw_mse
    (loss_free,) = torch.autograd.grad(loss, free, retain_graph=True)
    adjoint = torch.linalg.solve(hessian.T, loss_free)
    (gradient,) = torch.autograd.grad(residual, control, grad_outputs=-adjoint)
    symmetric_hessian = 0.5 * (hessian + hessian.T)
    eigenvalues = torch.linalg.eigvalsh(symmetric_hessian)
    singular_values = torch.linalg.svdvals(hessian)
    return ImplicitResult(
        raw_mse=float(raw_mse.detach()),
        scaled_loss=float(loss.detach()),
        gradient=gradient.detach().numpy(),
        equilibrium_gradient_inf=float(residual.detach().abs().max()),
        hessian_eigen_min=float(eigenvalues.min()),
        hessian_eigen_max=float(eigenvalues.max()),
        hessian_condition=float(singular_values.max() / singular_values.min()),
    )


def q_from_z(z: np.ndarray, q_ref_mpa: float) -> np.ndarray:
    """Convert dimensionless orthonormal controls to physical stress matrices."""
    return (q_ref_mpa * controls.matrices(torch.as_tensor(z, dtype=DTYPE))).numpy()


def solve_checked(
    fixture: Any,
    cfg: Config,
    z: np.ndarray,
    q_ref_mpa: float,
    *,
    start: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run the unchanged source-12 forward acceptance contract."""
    x, solver = study.solve_equilibrium(
        fixture,
        cfg,
        q_from_z(z, q_ref_mpa),
        start=start,
    )
    return x, solver


def physical_delta_rms(
    before: torch.Tensor, after: torch.Tensor, q_ref: float
) -> float:
    """Return per-cell Frobenius RMS active-stress change in MPa."""
    return float(q_ref * (after - before).square().sum(-1).mean().sqrt())


def simulate_first_update(
    gradient: np.ndarray,
    *,
    epsilon: float,
    learning_rate: float,
    q_ref_mpa: float,
    maximum: float,
) -> dict[str, Any]:
    """Apply exactly one projected torch Adam update from zero."""
    z = torch.zeros(gradient.shape, dtype=DTYPE, requires_grad=True)
    optimizer = torch.optim.Adam(
        (z,), lr=learning_rate, eps=epsilon, betas=(0.9, 0.999)
    )
    before = z.detach().clone()
    z.grad = torch.as_tensor(gradient, dtype=DTYPE)
    optimizer.step()
    unprojected = z.detach().clone()
    projection = controls.project(z, maximum)
    after = z.detach().clone()
    state = optimizer.state[z]
    sqrt_v_hat = (state["exp_avg_sq"] / (1 - 0.999 ** int(state["step"]))).sqrt()
    return {
        "epsilon": epsilon,
        "learning_rate": learning_rate,
        "unprojected_delta_Q_rms_MPa": physical_delta_rms(
            before, unprojected, q_ref_mpa
        ),
        "actual_projected_delta_Q_rms_MPa": physical_delta_rms(
            before, after, q_ref_mpa
        ),
        "projection": projection,
        "sqrt_v_hat": sqrt_v_hat.numpy(),
        "z_after": after.numpy(),
    }


def calibrate_learning_rate(
    gradient: np.ndarray,
    *,
    epsilon: float,
    target_delta_q_rms_mpa: float,
    q_ref_mpa: float,
    maximum: float,
) -> tuple[float, dict[str, Any]]:
    """Biselect learning rate using only the pre-run first projected update."""
    lower, upper = 0.0, BASELINE_LEARNING_RATE
    while (
        simulate_first_update(
            gradient,
            epsilon=epsilon,
            learning_rate=upper,
            q_ref_mpa=q_ref_mpa,
            maximum=maximum,
        )["actual_projected_delta_Q_rms_MPa"]
        < target_delta_q_rms_mpa
    ):
        upper *= 2
    for _ in range(80):
        middle = 0.5 * (lower + upper)
        value = simulate_first_update(
            gradient,
            epsilon=epsilon,
            learning_rate=middle,
            q_ref_mpa=q_ref_mpa,
            maximum=maximum,
        )["actual_projected_delta_Q_rms_MPa"]
        if value < target_delta_q_rms_mpa:
            lower = middle
        else:
            upper = middle
    learning_rate = 0.5 * (lower + upper)
    receipt = simulate_first_update(
        gradient,
        epsilon=epsilon,
        learning_rate=learning_rate,
        q_ref_mpa=q_ref_mpa,
        maximum=maximum,
    )
    return learning_rate, receipt


def simulate_continuation_update(
    z: torch.Tensor,
    optimizer: torch.optim.Adam,
    gradient: np.ndarray,
    learning_rate: float,
    q_ref_mpa: float,
    maximum: float,
) -> dict[str, Any]:
    """Evaluate the next Adam update without changing controls or moments."""
    group = optimizer.param_groups[0]
    state = optimizer.state[z]
    beta1, beta2 = group["betas"]
    step = int(state["step"]) + 1
    gradient_tensor = torch.as_tensor(gradient, dtype=DTYPE)
    first_moment = beta1 * state["exp_avg"] + (1 - beta1) * gradient_tensor
    second_moment = beta2 * state["exp_avg_sq"] + (1 - beta2) * gradient_tensor.square()
    first_hat = first_moment / (1 - beta1**step)
    second_hat = second_moment / (1 - beta2**step)
    before = z.detach().clone()
    unprojected = before - learning_rate * first_hat / (
        second_hat.sqrt() + group["eps"]
    )
    after = unprojected.clone()
    projection = controls.project(after, maximum)
    return {
        "learning_rate": learning_rate,
        "unprojected_delta_Q_rms_MPa": physical_delta_rms(
            before, unprojected, q_ref_mpa
        ),
        "actual_projected_delta_Q_rms_MPa": physical_delta_rms(
            before, after, q_ref_mpa
        ),
        "projection": projection,
    }


def calibrate_continuation_learning_rate(
    z: torch.Tensor,
    optimizer: torch.optim.Adam,
    gradient: np.ndarray,
    target_delta_q_rms_mpa: float,
    q_ref_mpa: float,
    maximum: float,
) -> tuple[float, dict[str, Any]]:
    """Biselect the next-step rate while preserving accumulated Adam moments."""
    lower = 0.0
    upper = float(optimizer.param_groups[0]["lr"])
    while (
        simulate_continuation_update(z, optimizer, gradient, upper, q_ref_mpa, maximum)[
            "actual_projected_delta_Q_rms_MPa"
        ]
        < target_delta_q_rms_mpa
    ):
        upper *= 2
    for _ in range(80):
        middle = 0.5 * (lower + upper)
        value = simulate_continuation_update(
            z, optimizer, gradient, middle, q_ref_mpa, maximum
        )["actual_projected_delta_Q_rms_MPa"]
        if value < target_delta_q_rms_mpa:
            lower = middle
        else:
            upper = middle
    learning_rate = 0.5 * (lower + upper)
    receipt = simulate_continuation_update(
        z, optimizer, gradient, learning_rate, q_ref_mpa, maximum
    )
    return learning_rate, receipt


def denominator_receipt(gradient: np.ndarray, epsilon: float) -> dict[str, float]:
    """Record the first-step Adam denominator scale."""
    values = np.abs(gradient).ravel()
    median = float(np.median(values))
    return {
        "sqrt_v_hat_min": float(values.min()),
        "sqrt_v_hat_median": median,
        "sqrt_v_hat_p99": float(np.quantile(values, 0.99)),
        "sqrt_v_hat_max": float(values.max()),
        "epsilon": epsilon,
        "epsilon_fraction_of_median_denominator": epsilon / (median + epsilon),
    }


def projected_gradient_mapping_rms(z: np.ndarray, gradient: np.ndarray) -> float:
    """Return ||Z-Pi(Z-grad)|| RMS with a declared unit mapping step."""
    current = torch.as_tensor(z, dtype=DTYPE)
    trial = current - torch.as_tensor(gradient, dtype=DTYPE)
    projected = trial.clone()
    controls.project(projected, 10.0)
    return float(torch.mean((current - projected).square()).sqrt())


def finite_difference_check(
    z_star: np.ndarray,
    target: np.ndarray,
    fixture: Any,
    cfg: Config,
    q_ref_mpa: float,
    loss_multiplier: float,
) -> dict[str, Any]:
    """Check one dense implicit directional derivative away from cone boundaries."""
    z = 0.5 * z_star
    x, solver = solve_checked(fixture, cfg, z, q_ref_mpa)
    if not solver["accepted"]:
        message = f"finite-difference center solve rejected: {solver}"
        raise RuntimeError(message)
    implicit = implicit_gradient(x, z, target, fixture, cfg, q_ref_mpa, loss_multiplier)
    direction = np.linspace(-1.0, 1.0, z.size).reshape(z.shape)
    direction /= np.linalg.norm(direction)
    implicit_directional = float(np.sum(implicit.gradient * direction))
    sweep = []
    for step in (3.0e-2, 1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4, 1.0e-4, 3.0e-5):
        losses = []
        perturbation_solvers = []
        perturbed_eigenvalues = []
        for sign in (-1.0, 1.0):
            perturbed = z + sign * step * direction
            eigenvalues = np.linalg.eigvalsh(q_from_z(perturbed, 1.0))
            perturbed_eigenvalues.append(eigenvalues.tolist())
            candidate, candidate_solver = solve_checked(
                fixture, cfg, perturbed, q_ref_mpa, start=x
            )
            perturbation_solvers.append(candidate_solver)
            if not candidate_solver["accepted"]:
                message = (
                    f"finite-difference perturbation solve rejected: {candidate_solver}"
                )
                raise RuntimeError(message)
            losses.append(loss_multiplier * float(np.mean((candidate - target) ** 2)))
        finite_difference = (losses[1] - losses[0]) / (2 * step)
        absolute_error = abs(finite_difference - implicit_directional)
        relative_error = absolute_error / max(
            abs(finite_difference), abs(implicit_directional), 1.0e-14
        )
        sweep.append(
            {
                "step_dimensionless_Z": step,
                "finite_difference_directional_derivative": finite_difference,
                "implicit_directional_derivative": implicit_directional,
                "absolute_error": absolute_error,
                "relative_error": relative_error,
                "perturbed_Z_eigenvalues": perturbed_eigenvalues,
                "perturbation_solvers": perturbation_solvers,
            }
        )
    selected = next(
        row
        for row in sweep
        if row["step_dimensionless_Z"] == cfg.finite_difference_step
    )
    return {
        "control_state": "0.5 * Z_star; strictly interior and unprojected",
        "step_dimensionless_Z": cfg.finite_difference_step,
        "direction": direction.tolist(),
        "perturbed_Z_eigenvalues": selected["perturbed_Z_eigenvalues"],
        "center_solver": solver,
        "perturbation_solvers": selected["perturbation_solvers"],
        "finite_difference_directional_derivative": selected[
            "finite_difference_directional_derivative"
        ],
        "implicit_directional_derivative": implicit_directional,
        "absolute_error": selected["absolute_error"],
        "relative_error": selected["relative_error"],
        "tolerance": 2.0e-4,
        "passed": selected["relative_error"] <= 2.0e-4,
        "implicit_hessian_condition": implicit.hessian_condition,
        "step_sweep": sweep,
        "interpretation": (
            "The unchanged accepted forward solve has a finite residual floor; "
            "agreement improves at moderate central-difference steps and degrades "
            "again when differenced loss changes approach solver noise."
        ),
    }


def save_mesh(
    path: Path,
    x: np.ndarray,
    target: np.ndarray,
    z: np.ndarray,
    q_ref_mpa: float,
    solver: dict[str, Any],
) -> None:
    """Save one exact endpoint with target and tensor controls."""
    grid = study.pv.UnstructuredGrid(
        np.c_[np.full(2, 4), study.TETS].ravel(),
        np.full(2, study.pv.CellType.TETRA),
        x,
    )
    grid.point_data["RestPosition"] = study.X0
    grid.point_data["Displacement"] = x - study.X0
    grid.point_data["TargetDisplacement"] = target - study.X0
    fixed = np.zeros(study.X0.size, dtype=bool)
    fixed[study.FIXED] = True
    grid.point_data["IsFixedCoordinate"] = fixed.reshape(study.X0.shape)
    grid.cell_data["MuscleFraction"] = study.FRACTION
    grid.cell_data["ZCoordinates"] = z
    grid.cell_data["ActiveStressMatrixMPa"] = q_from_z(z, q_ref_mpa).reshape(2, 9)
    grid.cell_data["detF"] = np.asarray(solver["detF"])
    grid.save(path)


def run_arm(
    identity: str,
    epsilon: float,
    learning_rate: float,
    target: np.ndarray,
    z_star: np.ndarray,
    fixture: Any,
    cfg: Config,
    q_ref_mpa: float,
    loss_multiplier: float,
    output: Path,
    updates: int,
    target_motion_rms: float,
    stop_relative_displacement_tolerance: float | None = None,
    recalibrate_at_step: int | None = None,
    recalibration_target_delta_q_rms_mpa: float | None = None,
) -> dict[str, Any]:
    """Run one projected-Adam arm, optionally stopping at declared recovery."""
    arm_dir = output / identity
    arm_dir.mkdir()
    z = torch.zeros(z_star.shape, dtype=DTYPE, requires_grad=True)
    optimizer = torch.optim.Adam(
        (z,), lr=learning_rate, eps=epsilon, betas=(0.9, 0.999)
    )
    traces: list[dict[str, Any]] = []
    solvers: list[dict[str, Any]] = []
    previous_x: np.ndarray | None = None
    cumulative_delta_q = 0.0
    final_x = study.X0.copy()
    final_solver: dict[str, Any] = {}
    recalibrations: list[dict[str, Any]] = []
    recovered = False
    for step in range(updates + 1):
        current = z.detach().numpy().copy()
        x, solver = solve_checked(fixture, cfg, current, q_ref_mpa, start=previous_x)
        solver_receipt = {"step": step, **solver}
        solvers.append(solver_receipt)
        append_jsonl(arm_dir / "solver-receipts.jsonl", solver_receipt)
        if not solver["accepted"]:
            failure = {
                "status": "failed_forward",
                "arm": identity,
                "step": step,
                "solver": solver,
            }
            write_json(arm_dir / "failure.json", failure)
            message = f"{identity} forward solve rejected at step {step}: {solver}"
            raise RuntimeError(message)
        implicit = implicit_gradient(
            x,
            current,
            target,
            fixture,
            cfg,
            q_ref_mpa,
            loss_multiplier,
        )
        physical_q = q_from_z(current, q_ref_mpa)
        eigenvalues = np.linalg.eigvalsh(physical_q)
        row = {
            "step": step,
            "raw_displacement_mse_fixture_unit2": implicit.raw_mse,
            "raw_displacement_rms_fixture_unit": math.sqrt(implicit.raw_mse),
            "scaled_loss": implicit.scaled_loss,
            "implicit_gradient_rms": float(np.sqrt(np.mean(implicit.gradient**2))),
            "projected_gradient_mapping_rms": projected_gradient_mapping_rms(
                current, implicit.gradient
            ),
            "equilibrium_gradient_inf": implicit.equilibrium_gradient_inf,
            "hessian_eigen_min": implicit.hessian_eigen_min,
            "hessian_eigen_max": implicit.hessian_eigen_max,
            "hessian_condition": implicit.hessian_condition,
            "Q_eigen_min_MPa": float(eigenvalues.min()),
            "Q_eigen_max_MPa": float(eigenvalues.max()),
            "Q_frobenius_rms_MPa": float(
                np.sqrt(np.mean(np.sum(physical_q**2, axis=(1, 2))))
            ),
            "Q_star_difference_rms_MPa_diagnostic_only": float(
                np.sqrt(np.mean((physical_q - q_from_z(z_star, q_ref_mpa)) ** 2))
            ),
            "detF_min_diagnostic_only": float(min(solver["detF"])),
            "detF_max_diagnostic_only": float(max(solver["detF"])),
            "cumulative_projected_delta_Q_rms_MPa": cumulative_delta_q,
            "outgoing_learning_rate": None,
            "outgoing_unprojected_delta_Q_rms_MPa": None,
            "outgoing_actual_projected_delta_Q_rms_MPa": None,
            "outgoing_projection_rms_dimensionless_Z": None,
            "outgoing_projected_negative_eigenvalue_fraction": None,
            "outgoing_projected_upper_eigenvalue_fraction": None,
            "outgoing_sqrt_v_hat_median": None,
            "outgoing_sqrt_v_hat_p99": None,
        }
        relative_displacement_error = (
            row["raw_displacement_rms_fixture_unit"] / target_motion_rms
        )
        initial_projected_gradient_mapping = (
            traces[0]["projected_gradient_mapping_rms"]
            if traces
            else row["projected_gradient_mapping_rms"]
        )
        projected_gradient_reduced = (
            row["projected_gradient_mapping_rms"] < initial_projected_gradient_mapping
        )
        recovered = (
            stop_relative_displacement_tolerance is not None
            and relative_displacement_error <= stop_relative_displacement_tolerance
            and projected_gradient_reduced
        )
        if not recovered and step < updates:
            if step == recalibrate_at_step:
                if recalibration_target_delta_q_rms_mpa is None:
                    message = "recalibration target is required"
                    raise ValueError(message)
                checkpoint_path = arm_dir / f"checkpoint-step-{step}.pt"
                torch.save(
                    {
                        "step": step,
                        "z": z.detach().clone(),
                        "x": x,
                        "optimizer": optimizer.state_dict(),
                        "implicit_gradient": implicit.gradient,
                        "raw_displacement_mse_fixture_unit2": implicit.raw_mse,
                    },
                    checkpoint_path,
                )
                previous_learning_rate = float(optimizer.param_groups[0]["lr"])
                new_learning_rate, recalibration = calibrate_continuation_learning_rate(
                    z,
                    optimizer,
                    implicit.gradient,
                    recalibration_target_delta_q_rms_mpa,
                    q_ref_mpa,
                    10.0,
                )
                optimizer.param_groups[0]["lr"] = new_learning_rate
                recalibration_receipt = {
                    "step": step,
                    "epsilon_unchanged": epsilon,
                    "previous_learning_rate": previous_learning_rate,
                    "new_learning_rate": new_learning_rate,
                    "target_actual_projected_delta_Q_rms_MPa": (
                        recalibration_target_delta_q_rms_mpa
                    ),
                    "optimizer_checkpoint": record(checkpoint_path),
                    "preserved_state": "q, u, Adam m/v/t, target, and loss multiplier",
                    **recalibration,
                }
                recalibrations.append(recalibration_receipt)
                write_json(
                    arm_dir / f"recalibration-step-{step}.json",
                    recalibration_receipt,
                )
            before = z.detach().clone()
            z.grad = torch.as_tensor(implicit.gradient, dtype=DTYPE)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            unprojected = z.detach().clone()
            projection = controls.project(z, 10.0)
            after = z.detach().clone()
            unprojected_delta = physical_delta_rms(before, unprojected, q_ref_mpa)
            actual_delta = physical_delta_rms(before, after, q_ref_mpa)
            cumulative_delta_q += actual_delta
            state = optimizer.state[z]
            bias = 1 - 0.999 ** int(state["step"])
            sqrt_v_hat = (state["exp_avg_sq"] / bias).sqrt().numpy().ravel()
            row.update(
                {
                    "outgoing_learning_rate": float(optimizer.param_groups[0]["lr"]),
                    "outgoing_unprojected_delta_Q_rms_MPa": unprojected_delta,
                    "outgoing_actual_projected_delta_Q_rms_MPa": actual_delta,
                    "outgoing_projection_rms_dimensionless_Z": projection[
                        "projection_rms"
                    ],
                    "outgoing_projected_negative_eigenvalue_fraction": projection[
                        "projected_negative_eigenvalue_fraction"
                    ],
                    "outgoing_projected_upper_eigenvalue_fraction": projection[
                        "projected_upper_eigenvalue_fraction"
                    ],
                    "outgoing_sqrt_v_hat_median": float(np.median(sqrt_v_hat)),
                    "outgoing_sqrt_v_hat_p99": float(np.quantile(sqrt_v_hat, 0.99)),
                }
            )
        traces.append(row)
        append_csv(arm_dir / "trace.csv", row)
        write_json(
            output / "progress.json",
            {"phase": "recovery", "arm": identity, "step": step},
        )
        previous_x = x
        final_x, final_solver = x, solver
        if recovered:
            break
    final_z = z.detach().numpy().copy()
    np.savez_compressed(
        arm_dir / "endpoint.npz",
        step=traces[-1]["step"],
        z=final_z,
        Q_MPa=q_from_z(final_z, q_ref_mpa),
        u=final_x - study.X0,
        target_u=target - study.X0,
        z_star=z_star,
        solver_valid=final_solver["accepted"],
    )
    save_mesh(
        arm_dir / "endpoint.vtu",
        final_x,
        target,
        final_z,
        q_ref_mpa,
        final_solver,
    )
    final = traces[-1]
    phase_receipts: dict[str, Any] = {}
    if recalibrate_at_step is not None and len(traces) > recalibrate_at_step:
        phase_receipts = {
            "initial_fixed_phase": {
                "updates": recalibrate_at_step,
                "epsilon": epsilon,
                "learning_rate": learning_rate,
                "final": traces[recalibrate_at_step],
            },
            "post_recalibration_phase": {
                "updates": traces[-1]["step"] - recalibrate_at_step,
                "epsilon": epsilon,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "final": traces[-1],
            },
        }
    return {
        "id": identity,
        "epsilon": epsilon,
        "learning_rate": learning_rate,
        "status": "recovered" if recovered else "completed_budget",
        "updates": traces[-1]["step"],
        "maximum_updates": updates,
        "accepted_forward_states": len(solvers),
        "initial": traces[0],
        "final": final,
        "projected_gradient_mapping_reduction": (
            final["projected_gradient_mapping_rms"]
            / traces[0]["projected_gradient_mapping_rms"]
        ),
        "relative_displacement_error": (
            final["raw_displacement_rms_fixture_unit"] / target_motion_rms
        ),
        "recovery_tolerance": stop_relative_displacement_tolerance,
        "recovered": recovered,
        "recalibrations": recalibrations,
        "phases": phase_receipts,
        "trace": record(arm_dir / "trace.csv"),
        "solver_receipts": record(arm_dir / "solver-receipts.jsonl"),
        "endpoint_npz": record(arm_dir / "endpoint.npz"),
        "endpoint_vtu": record(arm_dir / "endpoint.vtu"),
    }


def plot_recovery(arms: list[dict[str, Any]], output: Path) -> None:
    """Plot displacement recovery and projected-gradient progress."""
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for arm in arms:
        with Path(arm["trace"]["path"]).open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        steps = np.asarray([int(row["step"]) for row in rows])
        rms = np.asarray(
            [float(row["raw_displacement_rms_fixture_unit"]) for row in rows]
        )
        projected = np.asarray(
            [float(row["projected_gradient_mapping_rms"]) for row in rows]
        )
        label = f"eps={arm['epsilon']:.0e}, lr={arm['learning_rate']:.3g}"
        axes[0].semilogy(steps, rms, label=label)
        axes[1].semilogy(steps, projected, label=label)
    axes[0].set_title("Known-reachable displacement recovery")
    axes[0].set_xlabel("projected Adam update")
    axes[0].set_ylabel("RMS [fixture length unit]")
    axes[1].set_title("Projected-gradient progress")
    axes[1].set_xlabel("projected Adam update")
    axes[1].set_ylabel("unit-step mapping RMS in Z")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output, dpi=200, facecolor="white")
    figure.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(figure)


def validate_source12_fixture() -> tuple[dict[str, bool], dict[str, Any]]:
    """Bind this run to the previously passed source-12 fixture receipt."""
    receipt = json.loads(SOURCE12_RECEIPT.read_text())
    fixture = receipt["fixture"]
    checks = {
        "source12_status_completed": receipt["status"] == "completed",
        "points_exact": np.array_equal(np.asarray(fixture["points"]), study.X0),
        "tetrahedra_exact": np.array_equal(np.asarray(fixture["tets"]), study.TETS),
        "fractions_exact": np.array_equal(
            np.asarray(fixture["muscle_fractions"]), study.FRACTION
        ),
        "fixed_dofs_exact": np.array_equal(
            np.asarray(fixture["fixed_dofs"]), study.FIXED
        ),
        "rigid_constraint_rank_six": (
            fixture["rigid_mode_constraint_rank"]
            == study.rigid_mode_constraint_rank()
            == 6
        ),
        "source12_zero_Q_exact": (
            receipt["zero_Q_check"]["both_cells_Q_zero_max_position_abs_error"] == 0.0
        ),
        "source12_passive_loaded_solver_accepted": receipt["zero_Q_check"][
            "passive_solver"
        ]["accepted"],
        "source12_zero_Q_loaded_solver_accepted": receipt["zero_Q_check"][
            "zero_Q_solver"
        ]["accepted"],
    }
    return checks, receipt


def main(cfg: Config) -> None:
    """Run the generated-target recovery comparison."""
    global COMPLETED  # noqa: PLW0603
    torch.set_num_threads(1)
    output = cfg.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        message = f"choose an empty output directory: {output}"
        raise FileExistsError(message)
    for source in (
        Path(__file__),
        SOURCE12,
        HERE / "tensor_controls.py",
        HERE / "experiment_profile.py",
    ):
        destination = output / "sources" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    write_json(output / "config.json", cfg.model_dump(mode="json"))
    fixture_checks, _source12_receipt = validate_source12_fixture()
    if not all(fixture_checks.values()):
        message = f"source-12 fixture validation failed: {fixture_checks}"
        raise RuntimeError(message)
    fixture = study.Fixture.build()
    muscle_mu = cfg.muscle_E_mpa / (2 * (1 + cfg.poisson_ratio))
    q_ref_mpa = 3 * muscle_mu
    cap_mpa = 10 * q_ref_mpa
    z_star = generated_z_star()
    q_star_mpa = q_from_z(z_star, q_ref_mpa)
    z_star_eigenvalues = np.linalg.eigvalsh(q_from_z(z_star, 1.0))
    target, target_solver = study.solve_equilibrium(fixture, cfg, q_star_mpa)
    if not target_solver["accepted"]:
        message = f"generated target equilibrium rejected: {target_solver}"
        raise RuntimeError(message)
    target_motion_rms = study.shape_rms(target, study.X0)
    np.savez_compressed(
        output / "generated-target.npz",
        rest=study.X0,
        target=target,
        target_displacement=target - study.X0,
        z_star=z_star,
        Q_star_MPa=q_star_mpa,
        fractions=study.FRACTION,
        fixed_dofs=study.FIXED,
    )
    save_mesh(
        output / "generated-target.vtu",
        target,
        target,
        z_star,
        q_ref_mpa,
        target_solver,
    )
    zero_z = np.zeros_like(z_star)
    zero_x, zero_solver = solve_checked(fixture, cfg, zero_z, q_ref_mpa)
    if not zero_solver["accepted"]:
        message = f"zero-control equilibrium rejected: {zero_solver}"
        raise RuntimeError(message)
    raw_initial = implicit_gradient(
        zero_x, zero_z, target, fixture, cfg, q_ref_mpa, 1.0
    )
    raw_gradient_rms = float(np.sqrt(np.mean(raw_initial.gradient**2)))
    loss_multiplier = FACE_STEP64_FIT_GRADIENT_RMS / raw_gradient_rms
    scaled_initial = implicit_gradient(
        zero_x,
        zero_z,
        target,
        fixture,
        cfg,
        q_ref_mpa,
        loss_multiplier,
    )
    scaled_gradient_rms = float(np.sqrt(np.mean(scaled_initial.gradient**2)))
    baseline_update = simulate_first_update(
        scaled_initial.gradient,
        epsilon=EPSILONS[0],
        learning_rate=BASELINE_LEARNING_RATE,
        q_ref_mpa=q_ref_mpa,
        maximum=10.0,
    )
    candidate_target_delta = (
        CANDIDATE_STEP_MULTIPLIER * baseline_update["actual_projected_delta_Q_rms_MPa"]
    )
    calibrations: list[dict[str, Any]] = []
    settings: list[tuple[str, float, float]] = [
        ("baseline-eps-1e-2", EPSILONS[0], BASELINE_LEARNING_RATE)
    ]
    baseline_receipt = {
        key: value
        for key, value in baseline_update.items()
        if key not in {"sqrt_v_hat", "z_after"}
    }
    calibrations.append(
        {
            "id": settings[0][0],
            "role": "historical face optimizer setting and one-times baseline step",
            **baseline_receipt,
            "denominator": denominator_receipt(scaled_initial.gradient, EPSILONS[0]),
            "target_actual_projected_delta_Q_rms_MPa": baseline_update[
                "actual_projected_delta_Q_rms_MPa"
            ],
            "calibration_relative_to_baseline": 1.0,
        }
    )
    for epsilon in EPSILONS[1:]:
        learning_rate, update = calibrate_learning_rate(
            scaled_initial.gradient,
            epsilon=epsilon,
            target_delta_q_rms_mpa=candidate_target_delta,
            q_ref_mpa=q_ref_mpa,
            maximum=10.0,
        )
        identity = f"candidate-eps-{epsilon:.0e}".replace("e-0", "e-")
        settings.append((identity, epsilon, learning_rate))
        update_receipt = {
            key: value
            for key, value in update.items()
            if key not in {"sqrt_v_hat", "z_after"}
        }
        calibrations.append(
            {
                "id": identity,
                "role": "reduced-epsilon candidate calibrated before recovery outcomes",
                **update_receipt,
                "denominator": denominator_receipt(scaled_initial.gradient, epsilon),
                "target_actual_projected_delta_Q_rms_MPa": candidate_target_delta,
                "calibration_relative_to_baseline": CANDIDATE_STEP_MULTIPLIER,
                "relative_calibration_error": abs(
                    update["actual_projected_delta_Q_rms_MPa"] - candidate_target_delta
                )
                / candidate_target_delta,
            }
        )
    calibration_document = {
        "schema_version": 1,
        "status": "calibrated_before_recovery_outcomes",
        "control_coordinates": (
            "dimensionless orthonormal Z=Q/Qref: xx, yy, zz, sqrt(2)xy, "
            "sqrt(2)yz, sqrt(2)xz"
        ),
        "loss": {
            "raw": "mean squared displacement over all 15 fixture coordinates",
            "raw_units": "squared fixture-length units; the unit edge is 1",
            "positive_multiplier": loss_multiplier,
            "rule": (
                "chosen once so the zero-control 12-coordinate implicit-gradient "
                "RMS equals the saved face PSD step-64 fit-gradient RMS"
            ),
            "raw_initial_gradient_rms": raw_gradient_rms,
            "scaled_initial_gradient_rms": scaled_gradient_rms,
            "face_PSD_step64_fit_gradient_rms_reference": (
                FACE_STEP64_FIT_GRADIENT_RMS
            ),
            "minimizer_unchanged": True,
        },
        "candidate_rule": (
            "both reduced-epsilon learning rates biselect the first post-projection "
            "physical delta-Q RMS to exactly 2x the baseline first step; no recovery "
            "outcome or Q-star mismatch enters calibration"
        ),
        "calibrations": calibrations,
    }
    write_json(output / "calibration.json", calibration_document)
    fd_check = finite_difference_check(
        z_star,
        target,
        fixture,
        cfg,
        q_ref_mpa,
        loss_multiplier,
    )
    write_json(output / "implicit-gradient-check.json", fd_check)
    if not fd_check["passed"]:
        message = f"implicit finite-difference check failed: {fd_check}"
        raise RuntimeError(message)
    arms: list[dict[str, Any]] = []
    for index, (identity, epsilon, learning_rate) in enumerate(settings):
        cherries.set_step(index)
        arm = run_arm(
            identity,
            epsilon,
            learning_rate,
            target,
            z_star,
            fixture,
            cfg,
            q_ref_mpa,
            loss_multiplier,
            output,
            cfg.updates,
            target_motion_rms,
        )
        arms.append(arm)
        cherries.log_metrics(
            {
                f"{identity}/final_displacement_rms": arm["final"][
                    "raw_displacement_rms_fixture_unit"
                ],
                f"{identity}/final_projected_gradient_mapping_rms": arm["final"][
                    "projected_gradient_mapping_rms"
                ],
            }
        )
    ranked = sorted(
        arms,
        key=lambda arm: (
            arm["final"]["raw_displacement_mse_fixture_unit2"],
            arm["projected_gradient_mapping_reduction"],
        ),
    )
    selected = ranked[0]
    recovery = run_arm(
        f"recovery-{selected['id']}",
        selected["epsilon"],
        selected["learning_rate"],
        target,
        z_star,
        fixture,
        cfg,
        q_ref_mpa,
        loss_multiplier,
        output,
        cfg.recovery_max_updates,
        target_motion_rms,
        cfg.recovery_relative_displacement_tolerance,
        cfg.recovery_recalibration_step,
        candidate_target_delta,
    )
    plot_recovery([*arms, recovery], output / "recovery-traces.png")
    calibration_checks = {
        row["id"]: (
            abs(
                row["actual_projected_delta_Q_rms_MPa"]
                / row["target_actual_projected_delta_Q_rms_MPa"]
                - 1
            )
            <= 1.0e-10
        )
        for row in calibrations
    }
    tensor_eigen_min = min(arm["final"]["Q_eigen_min_MPa"] for arm in [*arms, recovery])
    tensor_eigen_max = max(arm["final"]["Q_eigen_max_MPa"] for arm in [*arms, recovery])
    checks = {
        **fixture_checks,
        "free_coordinate_count_nine": len(study.FREE) == 9,
        "target_solver_accepted": target_solver["accepted"],
        "target_Z_strictly_inside_PSD_cap": (
            float(z_star_eigenvalues.min()) > 0 and float(z_star_eigenvalues.max()) < 10
        ),
        "scaled_initial_gradient_matches_face_reference": math.isclose(
            scaled_gradient_rms,
            FACE_STEP64_FIT_GRADIENT_RMS,
            rel_tol=1.0e-12,
            abs_tol=1.0e-15,
        ),
        "implicit_directional_gradient_check": fd_check["passed"],
        "all_first_step_calibrations_match_declared_target": all(
            calibration_checks.values()
        ),
        "all_arms_completed_256_updates": all(
            arm["updates"] == cfg.updates
            and arm["accepted_forward_states"] == cfg.updates + 1
            for arm in arms
        ),
        "all_arms_reduce_displacement_error": all(
            arm["final"]["raw_displacement_mse_fixture_unit2"]
            < arm["initial"]["raw_displacement_mse_fixture_unit2"]
            for arm in arms
        ),
        "all_arms_reduce_projected_gradient_mapping": all(
            arm["projected_gradient_mapping_reduction"] < 1 for arm in arms
        ),
        "all_endpoint_tensors_in_spectral_box": (
            tensor_eigen_min >= -1.0e-12 and tensor_eigen_max <= cap_mpa + 1.0e-12
        ),
        "selected_arm_recovered_displacement": recovery["recovered"],
        "selected_arm_relative_displacement_error_at_most_1e-3": (
            recovery["relative_displacement_error"]
            <= cfg.recovery_relative_displacement_tolerance
        ),
        "selected_arm_reduced_projected_gradient_mapping": (
            recovery["projected_gradient_mapping_reduction"] < 1
        ),
    }
    if not all(checks.values()):
        write_json(
            output / "failure.json",
            {"status": "failed_checks", "checks": checks},
        )
        message = f"recovery study checks failed: {checks}"
        raise RuntimeError(message)
    summary = {
        "schema_version": 1,
        "status": "passed",
        "scope": (
            "CPU-only generated-target recovery on the validated mixed two-tet "
            "fixture; no face run, anatomy claim, unique-Q claim, or capacity claim"
        ),
        "fixture_units": (
            "coordinates use an abstract fixture length unit with rest shared edge "
            "length 1; displacement RMS is reported in that unit and is not mm"
        ),
        "fixture": {
            "points": study.X0.tolist(),
            "tets": study.TETS.tolist(),
            "muscle_fractions": study.FRACTION.tolist(),
            "fixed_dofs": study.FIXED.tolist(),
            "free_dofs": study.FREE.tolist(),
            "rigid_mode_constraint_rank": study.rigid_mode_constraint_rank(),
            "source12_receipt": record(SOURCE12_RECEIPT),
        },
        "constitutive": {
            "passive": (
                "fraction*stable(muscle)+(1-fraction)*stable(fat), exactly as source12"
            ),
            "active": "fraction*0.5*Q:(F^T F-I), exactly as source12",
            "muscle_E_MPa": cfg.muscle_E_mpa,
            "fat_E_MPa": cfg.fat_E_mpa,
            "poisson_ratio": cfg.poisson_ratio,
            "Qref_MPa": q_ref_mpa,
            "cap_MPa": cap_mpa,
        },
        "generated_target": {
            "rule": (
                "fixed rotated strictly positive-definite Z-star with eigenvalues "
                "well below the shared 10-Qref cap; target is its accepted equilibrium"
            ),
            "Z_star": z_star.tolist(),
            "Z_star_eigenvalues": z_star_eigenvalues.tolist(),
            "Q_star_MPa": q_star_mpa.tolist(),
            "target_motion_rms_fixture_unit": target_motion_rms,
            "solver": target_solver,
            "npz": record(output / "generated-target.npz"),
            "vtu": record(output / "generated-target.vtu"),
        },
        "implicit_differentiation": {
            "free_coordinates": 9,
            "formula": "H^T lambda=dL/dy; dL/dZ=-(d(E_y)/dZ)^T lambda",
            "finite_difference_check": fd_check,
        },
        "calibration": calibration_document,
        "arms": arms,
        "recovery": recovery,
        "selected": {
            "arm": selected["id"],
            "adam_eps": selected["epsilon"],
            "learning_rate": selected["learning_rate"],
            "screen_updates": cfg.updates,
            "recovery_updates": recovery["updates"],
            "recovery_relative_displacement_error": recovery[
                "relative_displacement_error"
            ],
        },
        "selection": {
            "rule_declared_before_outcomes": (
                "among completed 256-update arms, minimize final raw displacement "
                "MSE; use final/initial unit-step projected-gradient mapping ratio "
                "only as the tie-breaker; never use Q-star agreement. Then rerun the "
                "selected epsilon and initial learning rate unchanged for 4096 "
                "updates. If still unrecovered, preserve Adam state and recalibrate "
                "the learning rate once so the next physical step restores the "
                "already-declared 2x-baseline target; then allow 4096 more updates. "
                "Require displacement RMS error / target motion RMS <= 1e-3."
            ),
            "selected_arm": selected["id"],
            "recommended_epsilon": selected["epsilon"],
            "recommended_toy_learning_rate": selected["learning_rate"],
            "face_calibration_method": (
                "At the saved plain-PSD face checkpoint, preserve its true gradient "
                "and Adam moments, then choose learning rate before continuation so "
                "the first post-projection physical delta-Q RMS is 2x the baseline "
                "continuation; keep every one of the 1,729,410 controls."
            ),
            "claim_limit": (
                "The recommendation identifies a small-fixture optimizer candidate. "
                "The face probe must calibrate independently and can reject it."
            ),
        },
        "checks": checks,
        "acceptance_policy": (
            "Every measured state requires the unchanged source12 forward success "
            "and gradient-infinity tolerance 1e-8. det(F), inversion, geometry, and "
            "Q-star mismatch are recorded diagnostics and never acceptance gates."
        ),
        "provenance": {
            "source70": record(output / "sources/70-small-recovery.py"),
            "source12": record(output / "sources/12-small-model-study.py"),
            "tensor_controls": record(output / "sources/tensor_controls.py"),
            "profile": record(output / "sources/experiment_profile.py"),
            "config": record(output / "config.json"),
            "calibration": record(output / "calibration.json"),
            "gradient_check": record(output / "implicit-gradient-check.json"),
        },
    }
    write_json(output / "summary.json", summary)
    write_json(output / "progress.json", {"phase": "completed"})
    for path in (
        output / "summary.json",
        output / "calibration.json",
        output / "implicit-gradient-check.json",
        output / "recovery-traces.png",
        output / "recovery-traces.pdf",
    ):
        cherries.log_output(path)
    LOG.info("Selected %s and wrote %s", selected["id"], output)
    COMPLETED = True


if __name__ == "__main__":
    cherries.main(main, profile=ProfileCometNoCommit)
    if not COMPLETED:
        raise SystemExit(1)
