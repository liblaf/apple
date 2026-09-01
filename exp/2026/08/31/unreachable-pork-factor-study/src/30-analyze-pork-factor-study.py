"""Aggregate the controlled 2-D unreachable-pork factor-study artifacts.

The input runners deliberately retain failures and inversions.  This program
therefore reports every recorded frame, forward-equilibrium frames, and the
stricter subset usable by inverse physics; it never treats a missing field as
zero.
"""

from __future__ import annotations

# ruff: noqa: C901, EM102, FBT001, FBT003, PERF401, PLR0912, PLR0915, TRY003
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pydantic_settings as ps

from liblaf import cherries


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)

    output_dir: Path = cherries.output("30-analysis", mkdir=True)
    # A comma-separated list makes analysis independent of the invocation cwd.
    input_roots: str = "data/10-pork-2d"
    equilibrium_tolerance: float = 1.0e-6
    allow_partial: bool = False


EXPECTED_CASES = {
    ("2d", "baseline"),
    ("2d", "energy-linear"),
    ("2d", "height-high"),
    ("2d", "height-low"),
    ("2d", "loss-l1"),
    ("2d", "loss-linf"),
    ("2d", "mesh-dense"),
    ("2d", "mesh-medium"),
}


METRICS = {
    "target_mae": ("top_target_mae", "target/mae", "error_mae", "mae"),
    "target_rms": ("top_target_rms", "target/rms", "error_rms", "top_error_rms"),
    "target_max": ("top_target_max", "target/max", "error_max", "max_error"),
    "highpass_rms": ("top_highpass_rms", "top/highpass_rms", "highpass_rms"),
    "laplacian_rms": ("top_laplacian_rms", "top/laplacian_rms", "laplacian_rms"),
    "slope_rms": ("top_slope_rms", "top/slope_rms", "slope_rms"),
    "curvature_rms": ("top_curvature_rms", "top/curvature_rms", "curvature_rms"),
    "activation_jump_rms": (
        "activation_neighbor_jump_rms",
        "activation/neighbor_jump_rms",
        "activation/jump_rms",
    ),
    "activation_rms": ("activation_rms", "activation/rms"),
    "gradient_rms": ("gradient_rms", "gradient/rms", "forward/grad_norm", "grad_norm"),
    "equilibrium_residual_rms": ("equilibrium_residual_rms", "forward/residual_rms"),
    "min_det_f": ("min_det_f", "minimum_det_f", "detF/min", "det_f/min"),
    "min_det_g": ("min_det_g", "minimum_det_g", "detG/min", "det_g/min"),
    "min_det_ainv": ("min_det_ainv", "minimum_det_ainv", "detAinv/min", "det_ainv/min"),
}


def number(value: Any) -> float | None:
    """Return a finite scalar, retaining unavailable/non-finite data as missing."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def pick(record: dict[str, Any], names: tuple[str, ...]) -> float | None:
    for name in names:
        value = number(record.get(name))
        if value is not None:
            return value
    return None


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def discover(cfg: Config) -> list[Path]:
    candidates: set[Path] = set()
    experiment = Path(__file__).resolve().parents[1]
    roots = []
    for token in cfg.input_roots.split(","):
        if not token.strip():
            continue
        root = Path(token.strip())
        if not root.is_absolute() and not root.exists():
            root = experiment / root
        roots.append(root)
    for root in roots:
        if root.exists():
            candidates.update(root.rglob("summary.json"))
    return sorted(path for path in candidates if cfg.output_dir not in path.parents)


def dimension(summary: dict[str, Any], case: dict[str, Any]) -> str:
    geometry = case.get("geometry", summary.get("geometry", {}))
    domain = geometry.get("domain") if isinstance(geometry, dict) else None
    counts = case.get("counts", {})
    if (isinstance(domain, list) and len(domain) == 3) or (
        isinstance(counts, dict) and "tets" in counts
    ):
        return "3d"
    return "2d"


def case_fields(case: dict[str, Any], source: Path) -> dict[str, Any]:
    raw = case.get("case", case)
    raw = raw if isinstance(raw, dict) else case
    resolution = raw.get("resolution", case.get("resolution"))
    if isinstance(resolution, list):
        resolution = "x".join(str(v) for v in resolution)
    return {
        "case_name": str(raw.get("name", case.get("name", source.parent.name))),
        "energy": raw.get("energy", case.get("energy")),
        "loss": raw.get("loss", case.get("loss")),
        "resolution": resolution,
        "height": number(raw.get("height", case.get("height"))),
    }


def csv_trace(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))
    except OSError:
        return []


def trace_for(case: dict[str, Any], source: Path) -> list[dict[str, Any]]:
    embedded = case.get("trace")
    if isinstance(embedded, list):
        return [row for row in embedded if isinstance(row, dict)]
    return csv_trace(source.parent / "trace.csv")


def boolish(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes"}
    return bool(value)


def forward_equilibrium(row: dict[str, Any], cfg: Config) -> bool:
    evaluation = boolish(row.get("evaluation_success"))
    forward = boolish(row.get("forward_converged", row.get("forward/success")))
    residual = pick(row, METRICS["equilibrium_residual_rms"])
    # 3-D summaries do not currently expose a residual: solver success is the
    # strongest available evidence, not a fabricated residual threshold.
    return (
        evaluation is not False
        and forward is True
        and (residual is None or residual <= cfg.equilibrium_tolerance)
    )


def valid_inverse(row: dict[str, Any], cfg: Config) -> bool:
    """Return whether this frame has both a usable equilibrium and adjoint.

    2-D evaluations do not need to expose an adjoint solve.  In 3-D, however,
    an explicit ``adjoint/success`` column means the inverse gradient is only
    valid when that solve succeeded.  This is a reporting rule, not an extra
    optimization constraint.
    """
    if not forward_equilibrium(row, cfg):
        return False
    return "adjoint/success" not in row or boolish(row.get("adjoint/success")) is True


def valid_equilibrium(row: dict[str, Any], cfg: Config) -> bool:
    """Backward-compatible name for a valid inverse-physics evaluation."""
    return valid_inverse(row, cfg)


def metrics(record: dict[str, Any]) -> dict[str, float | None]:
    return {name: pick(record, aliases) for name, aliases in METRICS.items()}


def selected(
    rows: list[dict[str, Any]], key: str, valid_only: bool, cfg: Config
) -> dict[str, Any] | None:
    available = [row for row in rows if not valid_only or valid_inverse(row, cfg)]
    available = [row for row in available if number(row.get(key)) is not None]
    return min(available, key=lambda row: float(row[key])) if available else None


def exact_resolution(value: Any, dimensions: int) -> tuple[int, ...] | None:
    """Return a stored integral structured resolution, never a guessed one."""
    if not isinstance(value, (list, tuple)) or len(value) != dimensions:
        return None
    result = []
    for component in value:
        parsed = number(component)
        if parsed is None or not parsed.is_integer() or parsed < 1:
            return None
        result.append(int(parsed))
    return tuple(result)


def design_counts(
    summary: dict[str, Any], case: dict[str, Any], dim: str
) -> dict[str, int | float | None]:
    """Extract declared controls and target observations without mesh inference."""
    raw = case.get("case", case)
    raw = raw if isinstance(raw, dict) else case
    counts = case.get("counts") if isinstance(case.get("counts"), dict) else {}
    geometry = case.get("geometry", summary.get("geometry", {}))
    geometry = geometry if isinstance(geometry, dict) else {}
    if dim == "2d":
        cells = number(case.get("n_triangles"))
        muscle_cells = number(case.get("n_muscle_triangles"))
        activation_dofs = number(case.get("activation_dofs"))
        resolution = exact_resolution(raw.get("resolution", case.get("resolution")), 2)
        top_vertices = resolution[0] - 1 if resolution is not None else None
        top_components = 2 * top_vertices if top_vertices is not None else None
    else:
        cells = number(counts.get("tets"))
        muscle_cells = number(counts.get("muscle_tets"))
        activation_dofs = number(counts.get("activation_dofs"))
        resolution = exact_resolution(geometry.get("structured_resolution"), 3)
        top_vertices = (
            (resolution[0] - 1) * (resolution[2] - 1)
            if resolution is not None
            else None
        )
        top_components = 3 * top_vertices if top_vertices is not None else None
    ratio = (
        activation_dofs / top_components
        if activation_dofs is not None and top_components not in (None, 0)
        else None
    )
    return {
        "cells": cells,
        "muscle_cells": muscle_cells,
        "activation_dofs": activation_dofs,
        "free_top_observation_vertices": top_vertices,
        "free_top_observation_components": top_components,
        "activation_dofs_per_observed_component": ratio,
    }


def row_data(
    summary: dict[str, Any], case: dict[str, Any], source: Path, cfg: Config
) -> dict[str, Any]:
    fields = case_fields(case, source)
    dim = dimension(summary, case)
    inverse = case.get("inverse") if isinstance(case.get("inverse"), dict) else {}
    rows = trace_for(case, source)
    # The 2-D trace is sibling to each case summary, whereas a root summary
    # keeps cases one directory lower.  Recover that standard layout too.
    if not rows and fields["case_name"]:
        rows = csv_trace(source.parent / str(fields["case_name"]) / "trace.csv")
    case_dir = source.parent
    if source.parent.name != fields["case_name"]:
        case_dir = source.parent / str(fields["case_name"])
    if rows:
        if not (case_dir / "final.vtu").is_file():
            raise FileNotFoundError(case_dir / "final.vtu")
        series_path = case_dir / "history.vtu.series"
        if not series_path.is_file():
            raise FileNotFoundError(series_path)
        series = read_json(series_path).get("files")
        if not isinstance(series, list) or len(series) != len(rows):
            raise ValueError(f"{series_path} must have one frame per trace row")
        times = [
            number(item.get("time")) if isinstance(item, dict) else None
            for item in series
        ]
        if times != list(range(len(rows))):
            raise ValueError(f"{series_path} times must be exactly 0..N-1")
        for item in series:
            if (
                not isinstance(item, dict)
                or not (case_dir / str(item.get("name"))).is_file()
            ):
                raise FileNotFoundError(f"invalid series frame in {series_path}")
        declared_evaluations = number(
            case.get("evaluations", inverse.get("evaluations"))
        )
        paraview = case.get("paraview", {})
        declared_frames = (
            number(paraview.get("frames")) if isinstance(paraview, dict) else None
        )
        for label, declared in (
            ("evaluations", declared_evaluations),
            ("paraview.frames", declared_frames),
        ):
            if declared is not None and declared != len(rows):
                raise ValueError(
                    f"{source} {label}={declared} != trace frames={len(rows)}"
                )
    best = selected(rows, "objective", False, cfg) or selected(rows, "loss", False, cfg)
    final = rows[-1] if rows else None
    valid_best = selected(rows, "objective", True, cfg) or selected(
        rows, "loss", True, cfg
    )
    declared_best = case.get("best", case.get("metrics", {}).get("best", {}))
    declared_final = case.get("final", case.get("metrics", {}).get("final", {}))
    best_record = best or (declared_best if isinstance(declared_best, dict) else {})
    final_record = final or (declared_final if isinstance(declared_final, dict) else {})
    best_metrics, final_metrics = metrics(best_record), metrics(final_record)
    valid_metrics = metrics(valid_best) if valid_best else dict.fromkeys(METRICS)
    failures = (
        case.get("inverse", {}).get("failures", {})
        if isinstance(case.get("inverse"), dict)
        else {}
    )
    forward_failures = number(
        case.get("forward_failure_count", failures.get("forward"))
    )
    adjoint_failures = number(
        case.get("adjoint_failure_count", failures.get("adjoint"))
    )
    n = len(rows) or number(
        case.get("evaluations", case.get("inverse", {}).get("evaluations"))
    )
    result: dict[str, Any] = {
        "dimension": dim,
        "source_summary": str(source),
        **fields,
        **design_counts(summary, case, dim),
        "frames": len(rows) if rows else n,
        "forward_equilibrium_frames": sum(forward_equilibrium(row, cfg) for row in rows)
        if rows
        else None,
        "valid_inverse_frames": sum(valid_inverse(row, cfg) for row in rows)
        if rows
        else None,
        # Keep the original column name as an alias for existing consumers.
        "valid_equilibrium_frames": sum(valid_inverse(row, cfg) for row in rows)
        if rows
        else None,
        "artifact_or_nonconverged_frames": sum(
            not valid_inverse(row, cfg) for row in rows
        )
        if rows
        else None,
        "forward_failure_fraction": forward_failures / n
        if forward_failures is not None and n
        else None,
        "adjoint_failure_fraction": adjoint_failures / n
        if adjoint_failures is not None and n
        else None,
        "first_inversion_step": None,
        "trajectory_min_det_f": None,
        "trajectory_min_det_g": None,
        "trajectory_min_det_ainv": None,
        "tail_steps": None,
        "tail_converged_fraction": None,
        "tail_objective_change": None,
        "tail_relative_range": None,
        "tail_gradient_rms": None,
        "tail_residual_max": None,
        "tail_gate_1pct": None,
        "physical_stationarity_gate": None,
        "refinement_accepted_iterations": None,
        "refinement_trial_forward_failure_count": None,
        "inverse_failure_count": None,
        "orientation_preserving_checkpoint_available": False,
        "orientation_preserving_best_step": None,
        "orientation_preserving_best_objective": None,
    }
    for label, values in (
        ("best", best_metrics),
        ("final", final_metrics),
        ("valid_best", valid_metrics),
    ):
        result.update({f"{label}_{name}": value for name, value in values.items()})
    result["best_step"] = number(
        best_record.get(
            "step", case.get("best_step", case.get("inverse", {}).get("best_step"))
        )
    )
    for index, row in enumerate(rows):
        dets = [
            pick(row, METRICS[key])
            for key in ("min_det_f", "min_det_g", "min_det_ainv")
        ]
        if result["first_inversion_step"] is None and any(
            value is not None and value <= 0 for value in dets
        ):
            result["first_inversion_step"] = number(row.get("step", index))
    for label, metric in (
        ("trajectory_min_det_f", "min_det_f"),
        ("trajectory_min_det_g", "min_det_g"),
        ("trajectory_min_det_ainv", "min_det_ainv"),
    ):
        values = [pick(row, METRICS[metric]) for row in rows]
        values = [value for value in values if value is not None]
        result[label] = min(values, default=None)
    convergence = inverse.get("convergence", {})
    if not isinstance(convergence, dict):
        convergence = {}
    stationarity = case.get("stationarity", {})
    if not isinstance(stationarity, dict):
        stationarity = {}
    result["physical_stationarity_gate"] = boolish(
        case.get(
            "physical_stationarity_gate",
            stationarity.get(
                "passed",
                convergence.get(
                    "physical_stationarity_gate",
                    convergence.get("practical_stationarity_gate"),
                ),
            ),
        )
    )
    refinement = case.get("refinement", inverse.get("refinement", {}))
    if isinstance(refinement, dict):
        result["refinement_accepted_iterations"] = number(
            refinement.get("accepted_iterations")
        )
        result["refinement_trial_forward_failure_count"] = number(
            refinement.get("trial_forward_failures")
        )
    if isinstance(failures, dict):
        result["inverse_failure_count"] = number(
            case.get("inverse_evaluation_failure_count", failures.get("inverse"))
        )
        if result["refinement_trial_forward_failure_count"] is None:
            result["refinement_trial_forward_failure_count"] = number(
                failures.get("refinement_trial_forward")
            )
    tail = case.get("tail_convergence", inverse.get("tail", {}))
    if isinstance(tail, dict):
        first_loss, last_loss = (
            number(tail.get("objective_first", tail.get("loss_first"))),
            number(tail.get("objective_last", tail.get("loss_last"))),
        )
        converged_fraction = number(tail.get("forward_converged_fraction"))
        if converged_fraction is None and "all_forward_adjoint_converged" in tail:
            converged_fraction = float(
                boolish(tail.get("all_forward_adjoint_converged")) is True
            )
        result.update(
            {
                "tail_steps": number(tail.get("window_steps", tail.get("window"))),
                "tail_converged_fraction": converged_fraction,
                "tail_objective_change": number(tail.get("objective_change"))
                if "objective_change" in tail
                else (
                    last_loss - first_loss
                    if first_loss is not None and last_loss is not None
                    else None
                ),
                "tail_relative_range": number(
                    tail.get("objective_relative_range", tail.get("relative_range"))
                ),
                "tail_gradient_rms": number(
                    tail.get("gradient_rms_last", tail.get("grad_norm_last"))
                ),
                "tail_residual_max": number(tail.get("equilibrium_residual_rms_max")),
                "tail_gate_1pct": boolish(
                    tail.get(
                        "inverse_converged_1pct_tail_gate",
                        tail.get("converged_1pct_tail_gate"),
                    )
                ),
            }
        )
    elif rows:
        tail_rows = rows[-min(10, len(rows)) :]
        values = [number(row.get("objective", row.get("loss"))) for row in tail_rows]
        values = [value for value in values if value is not None]
        residuals = [
            pick(row, METRICS["equilibrium_residual_rms"]) for row in tail_rows
        ]
        residuals = [value for value in residuals if value is not None]
        result.update(
            {
                "tail_steps": len(tail_rows),
                "tail_converged_fraction": float(
                    np.mean([valid_inverse(row, cfg) for row in tail_rows])
                ),
                "tail_objective_change": values[-1] - values[0]
                if len(values) >= 2
                else None,
                "tail_relative_range": (max(values) - min(values))
                / max(abs(min(values)), 1.0e-30)
                if values
                else None,
                "tail_gradient_rms": pick(tail_rows[-1], METRICS["gradient_rms"]),
                "tail_residual_max": max(residuals, default=None),
            }
        )
    metric_block = case.get("metrics") if isinstance(case.get("metrics"), dict) else {}
    orientation = next(
        (
            candidate
            for candidate in (
                case.get("best_orientation_preserving"),
                case.get("orientation_preserving"),
                metric_block.get("best_orientation_preserving"),
                inverse.get("best_orientation_preserving"),
            )
            if isinstance(candidate, dict)
        ),
        None,
    )
    if isinstance(orientation, dict):
        orientation = orientation.get("best", orientation)
    if isinstance(orientation, dict):
        objective = number(orientation.get("objective", orientation.get("loss")))
        step = number(orientation.get("step", orientation.get("best_step")))
        result.update(
            {
                "orientation_preserving_checkpoint_available": objective is not None
                or step is not None,
                "orientation_preserving_best_step": step,
                "orientation_preserving_best_objective": objective,
            }
        )
    return result


def preferred_case(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Select retained reruns without hiding the other artifacts on disk.

    A passed tail gate is primary because an intentionally retained
    non-converged run must not replace an extended converged rerun.  The next
    keys reward more usable inverse frames and then more total frames.  The
    source path breaks an exact tie reproducibly.
    """

    def count(value: Any) -> float:
        parsed = number(value)
        return parsed if parsed is not None else -1.0

    return min(
        records,
        key=lambda row: (
            -(boolish(row.get("tail_gate_1pct")) is True),
            -count(row.get("valid_inverse_frames")),
            -count(row.get("frames")),
            str(row["source_summary"]),
        ),
    )


def load_cases(cfg: Config) -> list[dict[str, Any]]:
    candidates: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for path in discover(cfg):
        summary = read_json(path)
        if not summary:
            continue
        root_cases = summary.get("cases")
        if isinstance(root_cases, list):
            cases = root_cases
        # A 2-D case summary stores fields at its root; 3-D stores the case
        # descriptor below ``case``.  Both own a sibling trace.csv.
        elif "case" in summary or "name" in summary:
            cases = [summary]
        else:
            continue
        for case in cases:
            if not isinstance(case, dict):
                continue
            record = row_data(summary, case, path, cfg)
            identity = (record["dimension"], record["case_name"])
            candidates.setdefault(identity, []).append(record)

    return sorted(
        (preferred_case(records) for records in candidates.values()),
        key=lambda row: (row["dimension"], row["case_name"]),
    )


def completeness(rows: list[dict[str, Any]], cfg: Config) -> dict[str, Any]:
    """Fail visibly when a nominally final OFAT analysis is incomplete."""
    actual = {(row["dimension"], row["case_name"]) for row in rows}
    missing = sorted(EXPECTED_CASES - actual)
    unexpected = sorted(actual - EXPECTED_CASES)
    if unexpected:
        raise ValueError(f"unexpected factor-study cases: {unexpected}")
    if missing and not cfg.allow_partial:
        raise ValueError(
            f"missing required factor-study cases: {missing}; "
            "use --allow-partial true only for explicitly preliminary analysis"
        )
    return {
        "complete": not missing,
        "allow_partial": cfg.allow_partial,
        "expected_case_count": len(EXPECTED_CASES),
        "selected_case_count": len(actual),
        "missing_cases": [f"{dimension}:{name}" for dimension, name in missing],
    }


def baseline(rows: list[dict[str, Any]], dimension: str) -> dict[str, Any] | None:
    candidates = [row for row in rows if row["dimension"] == dimension]
    for row in candidates:
        if (
            row["case_name"] == "baseline"
            or "stable-l2-medium-h050" in row["case_name"]
        ):
            return row
    return None


def effects(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        base = baseline(rows, row["dimension"])
        changed = []
        if base:
            for key in ("energy", "loss", "resolution", "height"):
                if row.get(key) != base.get(key):
                    changed.append(key)
        effect = {
            "dimension": row["dimension"],
            "case_name": row["case_name"],
            "baseline_case": base["case_name"] if base else None,
            "changed_factors": ",".join(changed) if base else None,
            "is_controlled_ofat": len(changed) == 1 if base else None,
        }
        for metric in (
            "target_mae",
            "target_rms",
            "target_max",
            "highpass_rms",
            "laplacian_rms",
            "slope_rms",
            "curvature_rms",
            "activation_jump_rms",
        ):
            value, reference = (
                row.get(f"best_{metric}"),
                base.get(f"best_{metric}") if base else None,
            )
            effect[f"best_{metric}_delta"] = (
                value - reference
                if value is not None and reference is not None
                else None
            )
            value, reference = (
                row.get(f"valid_best_{metric}"),
                base.get(f"valid_best_{metric}") if base else None,
            )
            effect[f"valid_best_{metric}_delta"] = (
                value - reference
                if value is not None and reference is not None
                else None
            )
        result.append(effect)
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, Any]], out: Path, metric: str, title: str) -> None:
    values = [
        (
            f"{row['dimension']}:{row['case_name']}",
            row.get(f"best_{metric}"),
            row.get(f"final_{metric}"),
        )
        for row in rows
    ]
    values = [value for value in values if value[1] is not None or value[2] is not None]
    if not values:
        return
    labels = [value[0] for value in values]
    x = np.arange(len(values))
    width = 0.38
    fig, axis = plt.subplots(figsize=(max(7, 0.8 * len(values)), 4.5))
    for shift, index, label in ((-width / 2, 1, "best"), (width / 2, 2, "final")):
        ys = [value[index] if value[index] is not None else np.nan for value in values]
        axis.bar(x + shift, ys, width, label=label)
    axis.set(title=title, ylabel=metric.replace("_", " "))
    axis.set_xticks(x, labels, rotation=40, ha="right")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def diagnostics_plot(rows: list[dict[str, Any]], out: Path) -> None:
    """Plot tail convergence, failures, and inversion timing without imputation."""
    labels = [f"{row['dimension']}:{row['case_name']}" for row in rows]
    if not labels:
        return
    fields = (
        ("tail_converged_fraction", "tail converged fraction"),
        ("tail_objective_change", "tail objective change"),
        ("forward_failure_fraction", "forward failure fraction"),
        ("adjoint_failure_fraction", "adjoint failure fraction"),
        ("first_inversion_step", "first inversion step"),
    )
    fig, axes = plt.subplots(
        len(fields), 1, figsize=(max(7, 0.8 * len(rows)), 12), sharex=True
    )
    x = np.arange(len(rows))
    for axis, (field, title) in zip(axes, fields, strict=True):
        values = [
            row.get(field) if row.get(field) is not None else np.nan for row in rows
        ]
        axis.bar(x, values)
        axis.set_ylabel(title)
        axis.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(x, labels, rotation=40, ha="right")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def bumpiness_mechanisms_plot(rows: list[dict[str, Any]], out: Path) -> None:
    """Describe the observed activation-discontinuity/bumpiness relationship."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    styles = {
        False: {"color": "tab:green", "label": "no recorded inversion", "marker": "o"},
        True: {"color": "tab:red", "label": "recorded inversion", "marker": "^"},
    }
    label_offsets = {
        ("2d", "baseline"): (-6, -16),
        ("2d", "energy-linear"): (-6, 10),
        ("2d", "height-low"): (-5, 8),
        ("2d", "loss-linf"): (6, 16),
        ("2d", "mesh-medium"): (5, 8),
        ("3d", "linear-l2-medium-h050"): (5, 14),
        ("3d", "stable-l1-medium-h050"): (5, -16),
        ("3d", "stable-l2-low-h050"): (-5, -14),
        ("3d", "stable-l2-medium-h050"): (5, 14),
    }
    for axis, dim in zip(axes, ("2d", "3d"), strict=True):
        points = []
        for row in rows:
            if row["dimension"] != dim:
                continue
            jump, highpass = (
                number(row.get("final_activation_jump_rms")),
                number(row.get("final_highpass_rms")),
            )
            if jump is not None and highpass is not None and jump > 0 and highpass > 0:
                points.append((row, jump, highpass))
        axis.set(
            title=f"{dim.upper()}: final activation discontinuity and top bumpiness"
        )
        axis.set(
            xlabel="final activation neighbor-jump RMS",
            ylabel="final top high-pass RMS",
        )
        if not points:
            axis.text(
                0.5,
                0.5,
                "No finite final metrics",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
            continue
        for inverted in (False, True):
            selected_points = [
                point
                for point in points
                if (point[0]["first_inversion_step"] is not None) is inverted
            ]
            if not selected_points:
                continue
            style = styles[inverted]
            axis.scatter(
                [point[1] for point in selected_points],
                [point[2] for point in selected_points],
                s=52,
                **style,
            )
            offsets = ((5, 8), (5, -13), (-5, -12), (5, 8), (-5, -12))
            all_jumps = [point[1] for point in points]
            all_highpass = [point[2] for point in points]
            jump_edge = max(all_jumps) - 0.12 * (max(all_jumps) - min(all_jumps))
            highpass_edge = max(all_highpass) - 0.12 * (
                max(all_highpass) - min(all_highpass)
            )
            for index, (row, jump, highpass) in enumerate(selected_points):
                dx, dy = label_offsets.get(
                    (dim, row["case_name"]), offsets[index % len(offsets)]
                )
                if jump >= jump_edge:
                    dx = -5
                if highpass >= highpass_edge:
                    dy = -12
                axis.annotate(
                    row["case_name"],
                    (jump, highpass),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    fontsize=8,
                    ha="left" if dx > 0 else "right",
                    va="bottom" if dy > 0 else "top",
                )
        for values, setter in (
            ([point[1] for point in points], axis.set_xscale),
            ([point[2] for point in points], axis.set_yscale),
        ):
            if max(values) / min(values) >= 10:
                setter("log")
        axis.grid(alpha=0.25, which="both")
        axis.legend(loc="upper left", fontsize=8)
    fig.suptitle("Descriptive diagnostic only: no fitted relationship or causal claim")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.80, wspace=0.30)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def markdown(
    rows: list[dict[str, Any]],
    effects_rows: list[dict[str, Any]],
    completion: dict[str, Any],
) -> str:
    status = (
        "Complete: all 16 required OFAT cases are present."
        if completion["complete"]
        else "PRELIMINARY: missing " + ", ".join(completion["missing_cases"]) + "."
    )

    def text(value: Any) -> str:
        return (
            "NA"
            if value is None
            else f"{value:.4g}"
            if isinstance(value, float)
            else str(value)
        )

    lines = [
        "# Unreachable pork factor-study analysis",
        "",
        status,
        "",
        "`Unreachable` is the deliberately demanding benchmark label, not a mathematical infeasibility certificate. These finite trajectories do not estimate a global or orientation-preserving reachability lower bound.",
        "",
        "Rows labelled `valid_best` are selected only from usable inverse evaluations. Physical stationarity gates and refinement/trial failures are reported separately; nonstationary cases are deliberately retained, and the legacy 1% tail gate is not treated as convergence. Inversions are observations, not optimization constraints. Blank CSV cells mean the runner did not provide that metric.",
        "",
        "| Dimension | Case | Physical stationarity | Valid/every frame | Final target RMS | Refinement iters | Trial failures |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        valid, total = row.get("valid_equilibrium_frames"), row.get("frames")
        lines.append(
            f"| {row['dimension']} | {row['case_name']} | {text(row.get('physical_stationarity_gate'))} | {text(valid)}/{text(total)} | {text(row.get('final_target_rms'))} | {text(row.get('refinement_accepted_iterations'))} | {text(row.get('refinement_trial_forward_failure_count'))} |"
        )
    lines.extend(
        [
            "",
            "## Trajectory determinant minima",
            "",
            "These are finite minima across every recorded trace frame, rather than determinants at only the best or final checkpoint.",
            "",
            "| Dimension | Case | Minimum detF | Minimum detG | Minimum detAinv |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['dimension']} | {row['case_name']} | {text(row.get('trajectory_min_det_f'))} | {text(row.get('trajectory_min_det_g'))} | {text(row.get('trajectory_min_det_ainv'))} |"
        )
    lines.extend(
        [
            "",
            "## OFAT comparison",
            "",
            "`is_controlled_ofat` is true only when exactly one of energy, loss, resolution, or height differs from the shared same-dimension baseline.",
            "",
            "| Dimension | Case | Changed factor | Controlled OFAT | Delta best target RMS | Delta best highpass |",
            "| --- | --- | --- | --- | ---: | ---: |",
        ]
    )
    for row in effects_rows:
        lines.append(
            f"| {row['dimension']} | {row['case_name']} | {row['changed_factors'] or 'NA'} | {row['is_controlled_ofat'] if row['is_controlled_ofat'] is not None else 'NA'} | {row['best_target_rms_delta'] if row['best_target_rms_delta'] is not None else 'NA'} | {row['best_highpass_rms_delta'] if row['best_highpass_rms_delta'] is not None else 'NA'} |"
        )
    lines.extend(
        [
            "",
            "## Control and observation counts",
            "",
            "These post-hoc counts compare raw activation DoFs with the vector displacement components observed on free top vertices. They quantify the count-based aspect of potential underdetermination, but do not prove artifact causality or target reachability.",
            "",
            "| Dimension | Case | Cells | Muscle cells | Activation DoFs | Free top vertices | Observed components | DoFs / component |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['dimension']} | {row['case_name']} | {text(row.get('cells'))} | {text(row.get('muscle_cells'))} | {text(row.get('activation_dofs'))} | {text(row.get('free_top_observation_vertices'))} | {text(row.get('free_top_observation_components'))} | {text(row.get('activation_dofs_per_observed_component'))} |"
        )
    lines.extend(
        [
            "",
            "## Bumpiness diagnostic",
            "",
            "`bumpiness-mechanisms.png` plots final activation neighbor-jump RMS against final top high-pass RMS in separate 2-D and 3-D panels. Circles have no recorded inversion; triangles have a recorded inversion. It is descriptive only and does not fit or assert a causal relationship.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(cfg: Config) -> None:
    if cfg.output_dir.exists() and any(cfg.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {cfg.output_dir}")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_cases(cfg)
    completion = completeness(rows, cfg)
    effects_rows = effects(rows)
    (cfg.output_dir / "completeness.json").write_text(
        json.dumps(completion, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_csv(cfg.output_dir / "cases.csv", rows)
    write_csv(cfg.output_dir / "factor-effects.csv", effects_rows)
    convergence = {
        "equilibrium_tolerance": cfg.equilibrium_tolerance,
        "cases": [
            {
                key: row.get(key)
                for key in (
                    "dimension",
                    "case_name",
                    "frames",
                    "forward_equilibrium_frames",
                    "valid_inverse_frames",
                    "valid_equilibrium_frames",
                    "artifact_or_nonconverged_frames",
                    "forward_failure_fraction",
                    "adjoint_failure_fraction",
                    "first_inversion_step",
                    "tail_steps",
                    "tail_converged_fraction",
                    "tail_objective_change",
                    "tail_relative_range",
                    "tail_gradient_rms",
                    "tail_residual_max",
                    "tail_gate_1pct",
                    "physical_stationarity_gate",
                    "refinement_accepted_iterations",
                    "refinement_trial_forward_failure_count",
                    "inverse_failure_count",
                    "orientation_preserving_checkpoint_available",
                    "orientation_preserving_best_step",
                    "orientation_preserving_best_objective",
                )
            }
            for row in rows
        ],
    }
    (cfg.output_dir / "convergence.json").write_text(
        json.dumps(convergence, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (cfg.output_dir / "results.md").write_text(
        markdown(rows, effects_rows, completion), encoding="utf-8"
    )
    for metric, title in (
        ("target_mae", "Target MAE"),
        ("target_rms", "Target RMS"),
        ("target_max", "Target maximum"),
        ("highpass_rms", "High-pass bumpiness"),
        ("laplacian_rms", "Laplacian bumpiness"),
        ("slope_rms", "Slope bumpiness"),
        ("curvature_rms", "Curvature bumpiness"),
        ("activation_jump_rms", "Activation jumps"),
    ):
        plot(rows, cfg.output_dir / f"{metric}.png", metric, title)
    diagnostics_plot(rows, cfg.output_dir / "convergence-diagnostics.png")
    bumpiness_mechanisms_plot(rows, cfg.output_dir / "bumpiness-mechanisms.png")
    cherries.log_metrics(
        {
            "analysis/cases": len(rows),
            "analysis/dimensions": len({row["dimension"] for row in rows}),
        }
    )


if __name__ == "__main__":
    cherries.main(main)
