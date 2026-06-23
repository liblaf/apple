from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from _human_face_config import INVERSE_PATIENCE, InverseCase, InverseConfig
from _human_face_forward import full_activation_inv_from_active, material_tree
from _human_face_metrics import (
    adjoint_solution_metrics,
    forward_quiet,
    forward_solution_metrics,
    point_error_stats,
    to_numpy,
)
from _human_face_output import make_history_mesh, make_result_mesh
from _human_face_runtime import CaseRuntime

from liblaf import cherries, melon

logger = logging.getLogger(__name__)


@dataclass
class BestState:
    step: int = 0
    loss: float = math.inf
    displacement: np.ndarray | None = None
    activation_inv: np.ndarray | None = None

    def update(
        self,
        *,
        step: int,
        loss: float,
        displacement: np.ndarray,
        activation_inv: np.ndarray,
    ) -> None:
        if loss < self.loss:
            self.step = step
            self.loss = loss
            self.displacement = displacement
            self.activation_inv = activation_inv


@dataclass(frozen=True)
class OptimizationResult:
    trace: list[dict[str, Any]]
    best: BestState
    history_frames: int
    stop_reason: str


def make_step_row(
    *,
    step: int,
    step_start: float,
    loss_value: float,
    error_stats: dict[str, torch.Tensor],
    runtime: CaseRuntime,
    grad: torch.Tensor,
    best: BestState,
    plateau_steps: int,
    forward_elapsed: float,
    backward_elapsed: float,
) -> dict[str, Any]:
    forward_metrics = forward_solution_metrics(
        runtime.differentiable_forward.last_forward_solution
    )
    adjoint_metrics = adjoint_solution_metrics(
        runtime.differentiable_forward.last_adjoint_solution
    )
    return {
        "step": float(step),
        "loss/total": loss_value,
        "loss/data": loss_value,
        "target/error_mean": float(error_stats["mean"].detach().cpu()),
        "target/error_rms": float(error_stats["rms"].detach().cpu()),
        "target/error_max": float(error_stats["max"].detach().cpu()),
        "activation_inv/rms": float(
            torch.linalg.vector_norm(runtime.activation_parameter.detach()).cpu()
            / math.sqrt(runtime.activation_parameter.numel())
        ),
        "activation_inv/max_abs": float(
            runtime.activation_parameter.detach().abs().max().cpu()
        ),
        "grad/norm": float(torch.linalg.vector_norm(grad).detach().cpu()),
        "best/step": float(best.step),
        "best/loss": float(best.loss),
        "plateau/steps": float(plateau_steps),
        "time/forward_s": forward_elapsed,
        "time/backward_s": backward_elapsed,
        "time/step_s": time.perf_counter() - step_start,
        **forward_metrics,
        **adjoint_metrics,
    }


def log_step_metrics(case: InverseCase, step: int, row: dict[str, Any]) -> None:
    logged_metrics = {
        f"{case.stem}/loss": row["loss/total"],
        f"{case.stem}/error_rms": row["target/error_rms"],
        f"{case.stem}/error_max": row["target/error_max"],
        f"{case.stem}/activation_inv_rms": row["activation_inv/rms"],
        f"{case.stem}/grad_norm": row["grad/norm"],
        f"{case.stem}/forward_success": float(row["forward/success"]),
        f"{case.stem}/forward_steps": float(row["forward/steps"]),
        f"{case.stem}/forward_relative_grad_norm": row["forward/relative_grad_norm"],
        f"{case.stem}/adjoint_success": float(row["adjoint/success"]),
        f"{case.stem}/adjoint_best_solver": float(row["adjoint/best_solver"]),
        f"{case.stem}/adjoint_relative_residual": row["adjoint/relative_residual"],
        f"{case.stem}/adjoint_absolute_residual": row["adjoint/absolute_residual"],
    }
    for solver_i in range(int(row["adjoint/solver_count"])):
        solver_prefix = f"adjoint/solver_{solver_i}"
        metric_prefix = f"{case.stem}/{solver_prefix}"
        logged_metrics[f"{metric_prefix}_success"] = float(
            row[f"{solver_prefix}/success"]
        )
        logged_metrics[f"{metric_prefix}_relative_residual"] = row[
            f"{solver_prefix}/relative_residual"
        ]
    cherries.log_metrics(logged_metrics)
    logger.info(
        (
            "%s step %03d loss %.6g rms %.6g grad %.6g best %.6g "
            "plateau %d forward %s/%s adjoint %s/%s rel %.6g"
        ),
        case.stem,
        step,
        row["loss/total"],
        row["target/error_rms"],
        row["grad/norm"],
        row["best/loss"],
        int(row["plateau/steps"]),
        row["forward/success"],
        row["forward/result"],
        row["adjoint/success"],
        row["adjoint/result"],
        row["adjoint/relative_residual"],
    )
    if not row["forward/success"] or not row["adjoint/success"]:
        logger.warning(
            "%s step %03d solver miss: forward=%s adjoint=%s adjoint_rel=%.6g",
            case.stem,
            step,
            row["forward/result"],
            row["adjoint/result"],
            row["adjoint/relative_residual"],
        )


def append_history_frame(
    *,
    writer: Any,
    runtime: CaseRuntime,
    target: np.ndarray,
    loss_mask: np.ndarray,
    displacement: np.ndarray,
    activation_inv: np.ndarray,
    row: dict[str, Any],
    step: int,
) -> float:
    history_start = time.perf_counter()
    step_mesh = make_result_mesh(
        runtime.mesh,
        target,
        loss_mask,
        displacement,
        activation_inv,
        {
            "inverse/step": step,
            "inverse/loss": row["loss/total"],
            "inverse/error_rms": row["target/error_rms"],
        },
    )
    writer.append(make_history_mesh(step_mesh), time=float(step))
    return time.perf_counter() - history_start


def optimize_case(
    *,
    case: InverseCase,
    cfg: InverseConfig,
    runtime: CaseRuntime,
    target: np.ndarray,
    loss_mask: np.ndarray,
    history_path: Path,
) -> OptimizationResult:
    best = BestState()
    plateau_steps = 0
    trace: list[dict[str, Any]] = []
    history_frames = 0
    stop_reason = "step_limit"

    with melon.io.VTKHDFTemporalUnstructuredGridWriter(history_path) as writer:
        for step in range(cfg.inverse_max_steps + 1):
            step_start = time.perf_counter()
            runtime.optimizer.zero_grad()
            materials = material_tree(
                runtime.base_materials,
                runtime.activation_parameter,
                runtime.active_ids_t,
                runtime.mesh.n_cells,
            )
            forward_start = time.perf_counter()
            output = forward_quiet(runtime.differentiable_forward, materials)
            forward_elapsed = time.perf_counter() - forward_start
            residual = (
                output[runtime.target_global_ids_t]
                - runtime.target_t[runtime.target_ids_t]
            )
            loss = residual.square().mean()

            backward_start = time.perf_counter()
            loss.backward()
            backward_elapsed = time.perf_counter() - backward_start
            grad = runtime.activation_parameter.grad
            if grad is None:
                msg = "differentiable forward did not produce activation gradients"
                raise RuntimeError(msg)
            if not torch.isfinite(grad).all():
                nonfinite = int((~torch.isfinite(grad)).sum().detach().cpu())
                msg = f"non-finite inverse gradient at step {step}: {nonfinite}"
                raise FloatingPointError(msg)

            displacement = to_numpy(output)[runtime.global_ids]
            activation_inv = to_numpy(
                full_activation_inv_from_active(
                    runtime.activation_parameter.detach(),
                    runtime.active_ids_t,
                    runtime.mesh.n_cells,
                )
            )
            loss_value = float(loss.detach().cpu())
            prev_best_loss = best.loss
            best.update(
                step=step,
                loss=loss_value,
                displacement=displacement,
                activation_inv=activation_inv,
            )
            if math.isinf(prev_best_loss) or loss_value <= (
                prev_best_loss - cfg.inverse_loss_min_delta
            ):
                plateau_steps = 0
            else:
                plateau_steps += 1

            row = make_step_row(
                step=step,
                step_start=step_start,
                loss_value=loss_value,
                error_stats=point_error_stats(residual.detach()),
                runtime=runtime,
                grad=grad,
                best=best,
                plateau_steps=plateau_steps,
                forward_elapsed=forward_elapsed,
                backward_elapsed=backward_elapsed,
            )
            trace.append(row)
            cherries.set_step(len(trace) - 1)
            log_step_metrics(case, step, row)
            if cfg.require_solver_success and (
                not row["forward/success"] or not row["adjoint/success"]
            ):
                msg = (
                    f"{case.stem} step {step} solver failed: "
                    f"forward={row['forward/result']} "
                    f"adjoint={row['adjoint/result']}"
                )
                raise RuntimeError(msg)
            row["time/history_s"] = append_history_frame(
                writer=writer,
                runtime=runtime,
                target=target,
                loss_mask=loss_mask,
                displacement=displacement,
                activation_inv=activation_inv,
                row=row,
                step=step,
            )
            history_frames += 1

            if plateau_steps >= INVERSE_PATIENCE:
                stop_reason = f"loss_plateau_{INVERSE_PATIENCE}_steps"
                break
            if step == cfg.inverse_max_steps:
                break
            runtime.optimizer.step()

    return OptimizationResult(
        trace=trace,
        best=best,
        history_frames=history_frames,
        stop_reason=stop_reason,
    )
