from __future__ import annotations

import json
import logging
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from _human_face_config import InverseCase, InverseConfig
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
    ) -> bool:
        if math.isfinite(loss) and loss < self.loss:
            self.step = step
            self.loss = loss
            self.displacement = displacement
            self.activation_inv = activation_inv
            return True
        return False


@dataclass(frozen=True)
class OptimizationResult:
    trace: list[dict[str, Any]]
    best: BestState
    history_frames: int
    stop_reason: str
    segment_history: list[dict[str, Any]]
    lr_history: list[dict[str, Any]]
    min_delta_history: list[dict[str, Any]]
    forward_fail_count: int
    adjoint_fail_count: int
    aggressive_lr_tried: bool


def make_step_row(
    *,
    step: int,
    step_start: float,
    loss_value: float,
    raw_loss_value: float,
    loss_scale: float,
    error_stats: dict[str, torch.Tensor],
    runtime: CaseRuntime,
    grad: torch.Tensor,
    best: BestState,
    forward_elapsed: float,
    backward_elapsed: float,
    lr: float,
    effective_min_delta_rel: float,
    effective_min_delta_abs: float,
    segment: int,
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
        "loss/mm2": loss_value,
        "loss/m2": raw_loss_value,
        "loss/raw": raw_loss_value,
        "loss/scale": loss_scale,
        "target/error_mean": float(error_stats["mean"].detach().cpu()),
        "target/error_rms": float(error_stats["rms"].detach().cpu()),
        "target/error_max": float(error_stats["max"].detach().cpu()),
        "target/error_mean_mm": float(error_stats["mean"].detach().cpu() * 1000.0),
        "target/error_rms_mm": float(error_stats["rms"].detach().cpu() * 1000.0),
        "target/error_max_mm": float(error_stats["max"].detach().cpu() * 1000.0),
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
        "best/loss_mm2": float(best.loss),
        "inverse/lr": float(lr),
        "diagnostic/effective_min_delta_rel": float(effective_min_delta_rel),
        "diagnostic/effective_min_delta_abs": float(effective_min_delta_abs),
        "segment/index": float(segment),
        "time/forward_s": forward_elapsed,
        "time/backward_s": backward_elapsed,
        "time/step_s": time.perf_counter() - step_start,
        **forward_metrics,
        **adjoint_metrics,
    }


def log_step_metrics(case: InverseCase, step: int, row: dict[str, Any]) -> None:
    logged_metrics = {
        f"{case.stem}/loss": row["loss/total"],
        f"{case.stem}/loss_mm2": row["loss/mm2"],
        f"{case.stem}/loss_m2": row["loss/m2"],
        f"{case.stem}/error_rms": row["target/error_rms"],
        f"{case.stem}/error_rms_mm": row["target/error_rms_mm"],
        f"{case.stem}/error_max": row["target/error_max"],
        f"{case.stem}/error_max_mm": row["target/error_max_mm"],
        f"{case.stem}/activation_inv_rms": row["activation_inv/rms"],
        f"{case.stem}/grad_norm": row["grad/norm"],
        f"{case.stem}/lr": row["inverse/lr"],
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
            "%s step %03d lr %.6g loss_mm2 %.6g loss_m2 %.6g "
            "rms_mm %.6g grad %.6g "
            "best %.6g "
            "forward %s/%s adjoint %s/%s rel %.6g"
        ),
        case.stem,
        step,
        row["inverse/lr"],
        row["loss/mm2"],
        row["loss/m2"],
        row["target/error_rms_mm"],
        row["grad/norm"],
        row["best/loss"],
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
            "inverse/loss_mm2": row["loss/mm2"],
            "inverse/loss_m2": row["loss/m2"],
            "inverse/loss_scale": row["loss/scale"],
            "inverse/error_rms": row["target/error_rms"],
            "inverse/error_rms_mm": row["target/error_rms_mm"],
        },
    )
    writer.append(make_history_mesh(step_mesh), time=float(step))
    return time.perf_counter() - history_start


def write_live_loss_plot(
    rows: list[dict[str, Any]], case: InverseCase, output: Path
) -> None:
    steps = np.asarray([row["step"] for row in rows], dtype=np.float64)
    losses = np.asarray([row["loss/total"] for row in rows], dtype=np.float64)
    valid = np.isfinite(losses) & (losses > 0.0)
    if not np.any(valid):
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    ax.plot(
        steps[valid],
        losses[valid],
        marker="o",
        linewidth=1.4,
        markersize=2.5,
        label="loss_mm2",
    )
    best_i = np.argmin(losses[valid])
    valid_steps = steps[valid]
    valid_losses = losses[valid]
    ax.scatter(
        [valid_steps[best_i]],
        [valid_losses[best_i]],
        color="tab:red",
        zorder=3,
        label="best",
    )
    ax.set_yscale("log")
    ax.set_xlabel("inverse step")
    ax.set_ylabel("loss_mm2")
    current_lr = float(rows[-1]["inverse/lr"])
    best_loss = float(np.nanmin(valid_losses))
    ax.set_title(f"{case.stem}\nlr={current_lr:g} best={best_loss:.6g} mm^2")
    ax.grid(visible=True, which="both", linewidth=0.4, alpha=0.35)
    ax.legend()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def json_ready(value: Any) -> Any:
    if isinstance(value, bool | int | float | str) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def append_live_trace(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {key: json_ready(value) for key, value in row.items()}
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def snapshot_path_for(
    live_plot_path: Path, case: InverseCase, step: int, interval: int
) -> Path | None:
    if interval <= 0 or step % interval != 0:
        return None
    return live_plot_path.parent / "snapshots" / f"{case.stem}-step{step:04d}.png"


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def reset_adam_moments(optimizer: torch.optim.Optimizer) -> None:
    optimizer.state.clear()


def restore_best_state(runtime: CaseRuntime, best: BestState) -> None:
    if best.activation_inv is None:
        return
    restored = best.activation_inv[runtime.active_ids]
    with torch.no_grad():
        runtime.activation_parameter.copy_(
            torch.as_tensor(
                restored,
                dtype=runtime.activation_parameter.dtype,
                device=runtime.activation_parameter.device,
            )
        )


def recent_log_slope(rows: list[dict[str, Any]]) -> float:
    losses = np.asarray(
        [
            row["loss/total"]
            for row in rows
            if math.isfinite(float(row["loss/total"]))
            and float(row["loss/total"]) > 0.0
        ],
        dtype=np.float64,
    )
    if losses.size < 2:
        return math.nan
    x = np.arange(losses.size, dtype=np.float64)
    y = np.log(losses)
    slope, _ = np.polyfit(x, y, deg=1)
    return float(slope)


def relative_improvement(before: float, after: float) -> float:
    if not math.isfinite(before) or not math.isfinite(after):
        return math.nan
    denom = max(abs(before), np.finfo(np.float64).tiny)
    return max(0.0, float((before - after) / denom))


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def optimize_case(  # noqa: C901, PLR0912, PLR0915
    *,
    case: InverseCase,
    cfg: InverseConfig,
    runtime: CaseRuntime,
    target: np.ndarray,
    loss_mask: np.ndarray,
    history_path: Path,
    trace_path: Path,
) -> OptimizationResult:
    best = BestState()
    trace: list[dict[str, Any]] = []
    segment_history: list[dict[str, Any]] = []
    lr_history: list[dict[str, Any]] = []
    min_delta_history: list[dict[str, Any]] = []
    history_frames = 0
    stop_reason = "step_limit"
    current_lr = float(cfg.inverse_lr)
    live_plot_path = cfg.live_plot_dir / f"{case.stem}-live-log-loss.png"
    effective_min_delta_rel = float(cfg.diagnostic_min_delta_rel)
    forward_fail_count = 0
    adjoint_fail_count = 0
    consecutive_solver_failures = 0
    aggressive_lr_tried = False
    deadline = time.perf_counter() + cfg.time_budget_hours * 3600.0
    reserve_s = cfg.reserve_minutes * 60.0
    set_optimizer_lr(runtime.optimizer, current_lr)
    lr_history.append(
        {
            "step": 0,
            "lr": current_lr,
            "reason": "initial_weak_guess",
        }
    )

    with melon.io.VTKHDFTemporalUnstructuredGridWriter(history_path) as writer:
        for step in range(cfg.inverse_max_steps + 1):
            in_baseline = step <= cfg.mandatory_baseline_steps
            remaining_s = deadline - time.perf_counter()
            if step > cfg.mandatory_baseline_steps and remaining_s < (
                reserve_s + cfg.step_time_budget_s
            ):
                stop_reason = (
                    "time_budget_smooth_decrease"
                    if segment_history
                    and bool(segment_history[-1].get("diagnostic/smooth_decrease"))
                    else "time_budget"
                )
                break
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
            loss_m2 = residual.square().mean()
            loss_mm2 = loss_m2 * cfg.loss_scale

            backward_start = time.perf_counter()
            loss_mm2.backward()
            backward_elapsed = time.perf_counter() - backward_start
            grad = runtime.activation_parameter.grad
            if grad is None:
                msg = "differentiable forward did not produce activation gradients"
                raise RuntimeError(msg)
            if not torch.isfinite(grad).all():
                nonfinite = int((~torch.isfinite(grad)).sum().detach().cpu())
                logger.warning(
                    "%s step %03d non-finite gradient entries: %d",
                    case.stem,
                    step,
                    nonfinite,
                )
                restore_best_state(runtime, best)
                current_lr = max(current_lr * cfg.lr_shrink_factor, cfg.min_lr)
                set_optimizer_lr(runtime.optimizer, current_lr)
                reset_adam_moments(runtime.optimizer)
                lr_history.append(
                    {
                        "step": step,
                        "lr": current_lr,
                        "reason": "nonfinite_gradient_backtrack",
                    }
                )
                stop_reason = "nonfinite_gradient_backtracked"
                break

            displacement = to_numpy(output)[runtime.global_ids]
            activation_inv = to_numpy(
                full_activation_inv_from_active(
                    runtime.activation_parameter.detach(),
                    runtime.active_ids_t,
                    runtime.mesh.n_cells,
                )
            )
            loss_value = float(loss_mm2.detach().cpu())
            raw_loss_value = float(loss_m2.detach().cpu())
            effective_min_delta_abs = (
                0.0
                if not math.isfinite(best.loss)
                else max(best.loss, np.finfo(np.float64).tiny) * effective_min_delta_rel
            )

            row = make_step_row(
                step=step,
                step_start=step_start,
                loss_value=loss_value,
                raw_loss_value=raw_loss_value,
                loss_scale=float(cfg.loss_scale),
                error_stats=point_error_stats(residual.detach()),
                runtime=runtime,
                grad=grad,
                best=best,
                forward_elapsed=forward_elapsed,
                backward_elapsed=backward_elapsed,
                lr=current_lr,
                effective_min_delta_rel=effective_min_delta_rel,
                effective_min_delta_abs=effective_min_delta_abs,
                segment=len(segment_history),
            )
            row["run/phase"] = (
                "mandatory_baseline" if in_baseline else "adaptive_continuation"
            )
            stable_forward = bool(row["forward/success"])
            stable_value = math.isfinite(loss_value)
            new_best = False
            if stable_value and stable_forward:
                new_best = best.update(
                    step=step,
                    loss=loss_value,
                    displacement=displacement,
                    activation_inv=activation_inv,
                )
                row["best/step"] = float(best.step)
                row["best/loss"] = float(best.loss)
                row["best/loss_mm2"] = float(best.loss)
            row["best/new"] = bool(new_best)
            row["best/accepted"] = bool(stable_value and stable_forward)
            skip_optimizer_step = False
            deteriorated_now = (
                stable_value
                and stable_forward
                and not new_best
                and step > 0
                and not in_baseline
                and math.isfinite(best.loss)
                and loss_value > best.loss * (1.0 + cfg.loss_deterioration_rel)
            )
            if deteriorated_now:
                restore_best_state(runtime, best)
                current_lr = max(current_lr * cfg.lr_shrink_factor, cfg.min_lr)
                set_optimizer_lr(runtime.optimizer, current_lr)
                reset_adam_moments(runtime.optimizer)
                skip_optimizer_step = True
                row["adaptive/immediate_decision"] = (
                    "loss_deterioration_backtrack_reduce_lr"
                )
                lr_history.append(
                    {
                        "step": step,
                        "lr": current_lr,
                        "reason": "loss_deterioration_backtrack_reduce_lr",
                    }
                )
            else:
                row["adaptive/immediate_decision"] = "none"
            row["inverse/lr_next"] = float(current_lr)
            if step == cfg.mandatory_baseline_steps and cfg.inverse_max_steps > step:
                restore_best_state(runtime, best)
                reset_adam_moments(runtime.optimizer)
                skip_optimizer_step = True
                row["adaptive/immediate_decision"] = (
                    "mandatory_baseline_complete_restore_best"
                )
                row["inverse/lr_next"] = float(current_lr)
                lr_history.append(
                    {
                        "step": step,
                        "lr": current_lr,
                        "reason": "mandatory_baseline_complete_restore_best",
                    }
                )
            trace.append(row)
            cherries.set_step(len(trace) - 1)
            log_step_metrics(case, step, row)
            if not row["forward/success"]:
                forward_fail_count += 1
            if not row["adjoint/success"]:
                adjoint_fail_count += 1
            if not row["forward/success"] or not row["adjoint/success"]:
                consecutive_solver_failures += 1
            else:
                consecutive_solver_failures = 0
            if cfg.require_solver_success and consecutive_solver_failures >= 3:
                if in_baseline:
                    restore_best_state(runtime, best)
                    current_lr = max(current_lr * cfg.lr_shrink_factor, cfg.min_lr)
                    set_optimizer_lr(runtime.optimizer, current_lr)
                    reset_adam_moments(runtime.optimizer)
                    skip_optimizer_step = True
                    consecutive_solver_failures = 0
                    row["adaptive/immediate_decision"] = (
                        "baseline_solver_cluster_backtrack_reduce_lr"
                    )
                    row["inverse/lr_next"] = float(current_lr)
                    lr_history.append(
                        {
                            "step": step,
                            "lr": current_lr,
                            "reason": ("baseline_solver_cluster_backtrack_reduce_lr"),
                        }
                    )
                else:
                    msg = (
                        f"{case.stem} solver failures clustered at step {step}: "
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
            plot_start = time.perf_counter()
            row["plot/live_loss_path"] = str(live_plot_path)
            row["trace/path"] = str(trace_path)
            try:
                append_live_trace(trace_path, row)
                write_live_loss_plot(trace, case, live_plot_path)
                snapshot_path = snapshot_path_for(
                    live_plot_path, case, step, cfg.live_snapshot_interval
                )
                if snapshot_path is not None:
                    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(live_plot_path, snapshot_path)
                    row["plot/live_snapshot_path"] = str(snapshot_path)
            except Exception:
                logger.exception(
                    "%s step %03d live artifact update failed", case.stem, step
                )
            row["time/live_plot_s"] = time.perf_counter() - plot_start

            if step == cfg.inverse_max_steps:
                break
            if (step + 1) % cfg.segment_steps == 0:
                segment_rows = trace[-cfg.segment_steps :]
                segment_best_before = (
                    math.inf
                    if len(trace) == len(segment_rows)
                    else min(
                        float(row["loss/total"]) for row in trace[: -cfg.segment_steps]
                    )
                )
                segment_best_after = best.loss
                log_slope = recent_log_slope(segment_rows)
                rel_improvement = relative_improvement(
                    segment_best_before, segment_best_after
                )
                segment_last_loss = float(segment_rows[-1]["loss/total"])
                deterioration = (
                    math.isfinite(segment_best_before)
                    and math.isfinite(segment_last_loss)
                    and segment_last_loss
                    > segment_best_before * (1.0 + cfg.loss_deterioration_rel)
                )
                smooth_decrease = (
                    math.isfinite(log_slope) and log_slope < -cfg.flat_log_slope_tol
                )
                flat_curve = (
                    math.isfinite(log_slope)
                    and abs(log_slope) <= cfg.flat_log_slope_tol
                )
                decision = "continue"
                if int(segment_rows[-1]["step"]) < cfg.mandatory_baseline_steps:
                    decision = "mandatory_baseline_fixed_lr_continue"
                elif consecutive_solver_failures >= 2:
                    restore_best_state(runtime, best)
                    current_lr = max(current_lr * cfg.lr_shrink_factor, cfg.min_lr)
                    set_optimizer_lr(runtime.optimizer, current_lr)
                    reset_adam_moments(runtime.optimizer)
                    decision = "solver_cluster_backtrack_reduce_lr"
                    skip_optimizer_step = True
                    lr_history.append(
                        {"step": step, "lr": current_lr, "reason": decision}
                    )
                elif deterioration:
                    restore_best_state(runtime, best)
                    current_lr = max(current_lr * cfg.lr_shrink_factor, cfg.min_lr)
                    set_optimizer_lr(runtime.optimizer, current_lr)
                    reset_adam_moments(runtime.optimizer)
                    decision = "loss_deterioration_backtrack_reduce_lr"
                    skip_optimizer_step = True
                    lr_history.append(
                        {"step": step, "lr": current_lr, "reason": decision}
                    )
                elif smooth_decrease:
                    if rel_improvement < effective_min_delta_rel:
                        effective_min_delta_rel *= 0.5
                        min_delta_history.append(
                            {
                                "step": step,
                                "effective_min_delta_rel": effective_min_delta_rel,
                                "reason": "smooth_log_decrease_reduce_diagnostic_delta",
                            }
                        )
                    if (
                        rel_improvement < cfg.diagnostic_min_delta_rel
                        and current_lr < cfg.max_lr
                    ):
                        current_lr = min(current_lr * cfg.slow_lr_factor, cfg.max_lr)
                        set_optimizer_lr(runtime.optimizer, current_lr)
                        reset_adam_moments(runtime.optimizer)
                        decision = "smooth_slow_decrease_increase_lr"
                        lr_history.append(
                            {"step": step, "lr": current_lr, "reason": decision}
                        )
                    else:
                        decision = "smooth_decrease_continue"
                elif flat_curve:
                    if not aggressive_lr_tried and current_lr < cfg.max_lr:
                        aggressive_lr_tried = True
                        current_lr = min(
                            current_lr * cfg.aggressive_lr_factor, cfg.max_lr
                        )
                        set_optimizer_lr(runtime.optimizer, current_lr)
                        reset_adam_moments(runtime.optimizer)
                        decision = "flat_curve_aggressive_lr_probe"
                        lr_history.append(
                            {"step": step, "lr": current_lr, "reason": decision}
                        )
                    else:
                        decision = "flat_after_aggressive_or_at_max_lr"
                        stop_reason = "ambiguous_plateau"
                elif current_lr < cfg.max_lr:
                    current_lr = min(current_lr * cfg.slow_lr_factor, cfg.max_lr)
                    set_optimizer_lr(runtime.optimizer, current_lr)
                    reset_adam_moments(runtime.optimizer)
                    decision = "unclear_segment_increase_lr"
                    lr_history.append(
                        {"step": step, "lr": current_lr, "reason": decision}
                    )
                segment_history.append(
                    {
                        "segment": len(segment_history),
                        "step_start": int(segment_rows[0]["step"]),
                        "step_end": int(segment_rows[-1]["step"]),
                        "lr": float(current_lr),
                        "best_loss_before": finite_or_none(segment_best_before),
                        "best_loss_after": finite_or_none(segment_best_after),
                        "diagnostic/relative_best_improvement": finite_or_none(
                            rel_improvement
                        ),
                        "diagnostic/segment_last_loss": finite_or_none(
                            segment_last_loss
                        ),
                        "diagnostic/loss_deterioration": bool(deterioration),
                        "diagnostic/log_loss_slope": finite_or_none(log_slope),
                        "diagnostic/smooth_decrease": bool(smooth_decrease),
                        "diagnostic/flat_curve": bool(flat_curve),
                        "diagnostic/effective_min_delta_rel": float(
                            effective_min_delta_rel
                        ),
                        "decision": decision,
                        "forward_fail_count_total": int(forward_fail_count),
                        "adjoint_fail_count_total": int(adjoint_fail_count),
                    }
                )
                logger.info(
                    (
                        "%s segment %d steps %d-%d slope %.6g rel_improve %.6g "
                        "decision %s lr %.6g"
                    ),
                    case.stem,
                    len(segment_history) - 1,
                    int(segment_rows[0]["step"]),
                    int(segment_rows[-1]["step"]),
                    log_slope,
                    rel_improvement,
                    decision,
                    current_lr,
                )
                if stop_reason == "ambiguous_plateau":
                    break
            if not skip_optimizer_step:
                runtime.optimizer.step()

    if stop_reason == "step_limit" and segment_history:
        stop_reason = (
            "step_limit_smooth_decrease"
            if bool(segment_history[-1].get("diagnostic/smooth_decrease"))
            else "ambiguous_plateau"
        )
    return OptimizationResult(
        trace=trace,
        best=best,
        history_frames=history_frames,
        stop_reason=stop_reason,
        segment_history=segment_history,
        lr_history=lr_history,
        min_delta_history=min_delta_history,
        forward_fail_count=forward_fail_count,
        adjoint_fail_count=adjoint_fail_count,
        aggressive_lr_tried=aggressive_lr_tried,
    )
