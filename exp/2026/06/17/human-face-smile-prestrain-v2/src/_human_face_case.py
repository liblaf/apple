from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
from _human_face_config import (
    ADJOINT_ATOL,
    ADJOINT_RTOL,
    APONEUROSIS_E,
    APONEUROSIS_NU,
    FAT_E,
    FAT_NU,
    FORWARD_ATOL,
    FORWARD_MAX_STEPS,
    FORWARD_RTOL,
    INVERSE_PATIENCE,
    MUSCLE_E,
    MUSCLE_NU,
    SKIN_E,
    SKIN_NU,
    SKIN_THICKNESS,
    InverseCase,
    InverseConfig,
)
from _human_face_loop import BestState, OptimizationResult, optimize_case
from _human_face_mesh import geometry_summary
from _human_face_metrics import adjoint_solution_metrics, forward_solution_metrics
from _human_face_output import bumpiness_metrics, make_result_mesh
from _human_face_runtime import CasePaths, CaseRuntime, build_case_runtime
from _human_face_targets import make_target_mesh, target_displacement_and_mask

from liblaf import cherries, melon


def require_best_state(
    case: InverseCase, best: BestState
) -> tuple[np.ndarray, np.ndarray]:
    if best.displacement is None or best.activation_inv is None:
        msg = f"{case.stem} did not evaluate any inverse state"
        raise RuntimeError(msg)
    return best.displacement, best.activation_inv


def case_summary(
    *,
    case: InverseCase,
    cfg: InverseConfig,
    runtime: CaseRuntime,
    target: np.ndarray,
    loss_mask: np.ndarray,
    target_metrics: dict[str, Any],
    result: OptimizationResult,
    elapsed_s: float,
    history_path: Path,
    trace_path: Path,
) -> dict[str, Any]:
    best_displacement, best_activation_inv = require_best_state(case, result.best)
    target_error = best_displacement[runtime.target_ids] - target[runtime.target_ids]
    target_norm = np.linalg.norm(target[runtime.target_ids])
    target_error_norm = np.linalg.norm(target_error, axis=1)
    active_activation_inv = best_activation_inv[runtime.active_ids]
    initial = result.trace[0]
    final = result.trace[-1]
    baseline_rows = [
        row
        for row in result.trace
        if int(row["step"]) <= int(cfg.mandatory_baseline_steps)
    ]
    baseline_completed = bool(baseline_rows) and int(baseline_rows[-1]["step"]) >= int(
        cfg.mandatory_baseline_steps
    )
    baseline_best = (
        min(baseline_rows, key=lambda row: float(row["loss/mm2"]))
        if baseline_rows
        else initial
    )
    baseline_lr_deviations = [
        row
        for row in result.lr_history
        if int(row["step"]) <= int(cfg.mandatory_baseline_steps)
        and row["reason"]
        not in {
            "initial_weak_guess",
            "mandatory_baseline_complete_restore_best",
        }
    ]
    converged = result.stop_reason == "converged_flat_after_aggressive"
    return {
        "case": case.stem,
        "input_mesh": str(cfg.input_mesh),
        "initial_activation_mesh": None
        if cfg.initial_activation_mesh is None
        else str(cfg.initial_activation_mesh),
        "initial_displacement/requested": bool(cfg.use_initial_displacement),
        "target/name": case.target,
        "case/label": case.label,
        "case/setup": case.setup_label,
        "n_points": int(runtime.mesh.n_points),
        "n_tets": int(runtime.mesh.n_cells),
        "n_active_tets": int(runtime.active_ids.size),
        "n_activation_parameters": int(runtime.active_ids.size),
        "n_activation_parameter_dofs": int(runtime.activation_parameter.numel()),
        "n_skin_triangles": 0 if runtime.skin is None else int(runtime.skin.n_cells),
        "n_bumpiness_edges": int(runtime.bump_edges.shape[0]),
        "elapsed_s": float(elapsed_s),
        "inverse/max_steps": int(cfg.inverse_max_steps),
        "baseline/mandatory_optimizer_steps": int(cfg.mandatory_baseline_steps),
        "baseline/evaluations_expected": int(cfg.mandatory_baseline_steps + 1),
        "baseline/evaluations": len(baseline_rows),
        "baseline/completed": bool(baseline_completed),
        "baseline/fixed_lr": float(cfg.inverse_lr),
        "baseline/lr_deviation_count": len(baseline_lr_deviations),
        "baseline/best_step": int(baseline_best["step"]),
        "baseline/best_loss_mm2": float(baseline_best["loss/mm2"]),
        "baseline/best_error_rms_mm": float(baseline_best["target/error_rms_mm"]),
        "inverse/lr_initial": float(cfg.inverse_lr),
        "inverse/lr_final": float(final["inverse/lr"]),
        "loss/scale": float(cfg.loss_scale),
        "optimizer/adam_eps": float(cfg.adam_eps),
        "inverse/patience": int(INVERSE_PATIENCE),
        "inverse/diagnostic_min_delta_rel_initial": float(cfg.diagnostic_min_delta_rel),
        "inverse/diagnostic_min_delta_rel_final": float(
            final["diagnostic/effective_min_delta_rel"]
        ),
        "inverse/effective_min_delta_abs_final": float(
            final["diagnostic/effective_min_delta_abs"]
        ),
        "inverse/stop_reason": result.stop_reason,
        "inverse/converged": bool(converged),
        "inverse/evaluations": len(result.trace),
        "inverse/segments": len(result.segment_history),
        "inverse/aggressive_lr_tried": bool(result.aggressive_lr_tried),
        "inverse/forward_fail_count": int(result.forward_fail_count),
        "inverse/adjoint_fail_count": int(result.adjoint_fail_count),
        "history/format": "VTKHDFTemporalUnstructuredGrid",
        "history/path": history_path.name,
        "history/frames": int(result.history_frames),
        "trace/path": trace_path.name,
        "plot/live_loss_path": str(
            cfg.live_plot_dir / f"{case.stem}-live-log-loss.png"
        ),
        "best_step": int(result.best.step),
        "best_loss": float(result.best.loss),
        "best_loss_mm2": float(result.best.loss),
        "best_loss_m2": float(result.best.loss / cfg.loss_scale),
        "best_error_rms_mm": float(
            np.linalg.norm(target_error) / math.sqrt(runtime.target_ids.size) * 1000.0
        ),
        "stop_reason": result.stop_reason,
        "history_frames": int(result.history_frames),
        "initial/loss": float(initial["loss/total"]),
        "initial/loss_mm2": float(initial["loss/mm2"]),
        "initial/loss_m2": float(initial["loss/m2"]),
        "initial/error_rms": float(initial["target/error_rms"]),
        "initial/error_rms_mm": float(initial["target/error_rms_mm"]),
        "best/step": int(result.best.step),
        "best/loss": float(result.best.loss),
        "best/loss_mm2": float(result.best.loss),
        "best/loss_m2": float(result.best.loss / cfg.loss_scale),
        "best/error_mean": float(target_error_norm.mean()),
        "best/error_rms": float(
            np.linalg.norm(target_error) / math.sqrt(runtime.target_ids.size)
        ),
        "best/error_rms_mm": float(
            np.linalg.norm(target_error) / math.sqrt(runtime.target_ids.size) * 1000.0
        ),
        "best/error_max": float(target_error_norm.max()),
        "best/error_max_mm": float(target_error_norm.max() * 1000.0),
        "best/error_rms_fraction_of_target": float(
            np.linalg.norm(target_error) / target_norm
        )
        if target_norm > 0.0
        else math.nan,
        "final/step": float(final["step"]),
        "final/loss": float(final["loss/total"]),
        "final/loss_mm2": float(final["loss/mm2"]),
        "final/loss_m2": float(final["loss/m2"]),
        "final/error_rms": float(final["target/error_rms"]),
        "final/error_rms_mm": float(final["target/error_rms_mm"]),
        "final/segment_index": int(final["segment/index"]),
        "activation/mode": "per-muscle-tet-6dof",
        "activation/shared": False,
        "activation/range_clamping": False,
        "activation_inv/initial_rms": float(
            np.linalg.norm(runtime.initial_activation)
            / math.sqrt(max(1, runtime.initial_activation.size))
        ),
        "activation_inv/initial_max_abs": float(
            np.abs(runtime.initial_activation).max()
        ),
        "initial_displacement/enabled": runtime.initial_displacement is not None,
        "initial_displacement/rms": math.nan
        if runtime.initial_displacement is None
        else float(
            np.linalg.norm(runtime.initial_displacement)
            / math.sqrt(max(1, runtime.initial_displacement.shape[0]))
        ),
        "initial_displacement/max": math.nan
        if runtime.initial_displacement is None
        else float(np.linalg.norm(runtime.initial_displacement, axis=1).max()),
        "activation_inv/rms": float(
            np.linalg.norm(active_activation_inv)
            / math.sqrt(max(1, active_activation_inv.size))
        ),
        "activation_inv/max_abs": float(np.abs(active_activation_inv).max()),
        "fat/E_MPa": float(FAT_E),
        "fat/nu": float(FAT_NU),
        "muscle/E_MPa": float(MUSCLE_E),
        "muscle/nu": float(MUSCLE_NU),
        "aponeurosis/E_MPa": float(APONEUROSIS_E),
        "aponeurosis/nu": float(APONEUROSIS_NU),
        "skin/enabled": bool(case.skin_enabled),
        "skin/E_MPa": float(SKIN_E),
        "skin/nu": float(SKIN_NU),
        "skin/thickness": float(SKIN_THICKNESS),
        **runtime.skin_metrics,
        "solver/forward": "PNCG",
        "solver/forward/rtol": float(FORWARD_RTOL),
        "solver/forward/atol": float(FORWARD_ATOL),
        "solver/forward/max_steps": int(FORWARD_MAX_STEPS),
        "solver/adjoint": "FallbackSolver(CupyCG,CupyMinRes)",
        "solver/adjoint/rtol": float(ADJOINT_RTOL),
        "solver/adjoint/atol": float(ADJOINT_ATOL),
        "loss/type": "point-to-point MSE in mm^2",
        "loss/optimizer": "loss_mm2",
        "trace": result.trace,
        "segment_history": result.segment_history,
        "lr_history": result.lr_history,
        "effective_min_delta_history": result.min_delta_history,
        **target_metrics,
        **geometry_summary(runtime.mesh),
        **bumpiness_metrics(
            mask=loss_mask,
            edges=runtime.bump_edges,
            displacement=best_displacement,
            target=target,
        ),
        **{
            f"last/{key}": value
            for key, value in forward_solution_metrics(
                runtime.differentiable_forward.last_forward_solution
            ).items()
        },
        **{
            f"last/{key}": value
            for key, value in adjoint_solution_metrics(
                runtime.differentiable_forward.last_adjoint_solution
            ).items()
        },
    }


def write_case_outputs(
    *,
    case: InverseCase,
    paths: CasePaths,
    runtime: CaseRuntime,
    target: np.ndarray,
    loss_mask: np.ndarray,
    summary: dict[str, Any],
    best: BestState,
) -> None:
    best_displacement, best_activation_inv = require_best_state(case, best)
    result_metrics = {
        key: value
        for key, value in summary.items()
        if isinstance(value, int | float | bool)
    }
    result = make_result_mesh(
        runtime.mesh,
        target,
        loss_mask,
        best_displacement,
        best_activation_inv,
        result_metrics,
    )
    melon.save(result, paths.result)
    paths.summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    cherries.log_output(paths.target)
    cherries.log_output(paths.result)
    cherries.log_output(paths.summary)
    cherries.log_output(paths.history)
    cherries.log_output(paths.trace)
    live_plot = Path(str(summary["plot/live_loss_path"]))
    if live_plot.exists():
        cherries.log_output(live_plot)


def solve_case(
    case: InverseCase, base_mesh: pv.UnstructuredGrid, cfg: InverseConfig
) -> dict[str, Any]:
    start = time.perf_counter()
    paths = CasePaths.from_case(cfg.output_summary.parent, case)
    paths.remove_stale()

    mesh = base_mesh.copy(deep=True)
    target, loss_mask, target_metrics = target_displacement_and_mask(mesh, case, cfg)
    melon.save(make_target_mesh(mesh, target, loss_mask), paths.target)

    runtime = build_case_runtime(
        case=case,
        cfg=cfg,
        mesh=mesh.copy(deep=True),
        target=target,
        loss_mask=loss_mask,
    )
    result = optimize_case(
        case=case,
        cfg=cfg,
        runtime=runtime,
        target=target,
        loss_mask=loss_mask,
        history_path=paths.history,
        trace_path=paths.trace,
    )
    summary = case_summary(
        case=case,
        cfg=cfg,
        runtime=runtime,
        target=target,
        loss_mask=loss_mask,
        target_metrics=target_metrics,
        result=result,
        elapsed_s=time.perf_counter() - start,
        history_path=paths.history,
        trace_path=paths.trace,
    )
    write_case_outputs(
        case=case,
        paths=paths,
        runtime=runtime,
        target=target,
        loss_mask=loss_mask,
        summary=summary,
        best=result.best,
    )
    return summary
