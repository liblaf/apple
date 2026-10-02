# ruff: noqa: C901, EM101, EM102, PLR0912, PLR0915, TRY003
"""Recompute archived benchmarks and compare saved activation-study states on CPU."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Literal

import matplotlib as mpl
import numpy as np
import pydantic_settings as ps
import pyvista as pv
from experiment_profile import ProfileCometNoCommit
from study_metrics import (
    StudyMetrics,
)

from liblaf import cherries

mpl.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[6]
GROUP = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "exp/2026/09/07/face-actuation-diagnosis/data/12-historical-fixture"
CAMERAS = ROOT / "exp/2026/09/08/physical-volume-closeups/data/20-regions/summary.json"
RAW6_OFF = ROOT / "exp/2026/09/08/local-skin-prestrain/data/30-refit-no-skin"
PSD_OFF = ROOT / "exp/2026/09/07/tensor-active-stress/data/21-psd"
PSD_SMOOTH = ROOT / "exp/2026/09/07/tensor-active-stress/data/22-psd-smooth"
PSD_1024 = ROOT / "exp/2026/09/07/tensor-active-stress/data/102-fit1024"
MU_MPA = 0.010067114093959731
SMOOTHNESS_LENGTH_M = 0.005
FIT_TOLERANCE_MM = 0.05
MOTION_TOLERANCE_MM = 0.05
PRIMARY_EFFECT_FRACTION = 0.10
LOW_FREQUENCY_RETENTION = 0.90
BACKGROUND = "#242c36"

FRESH_CASES = ("axis-off", "axis-on", "raw6-on")
ALL_CASES = ("axis-off", "axis-on", "raw6-off", "raw6-on")
LABELS = {
    "axis-off": "Learned axis, S(C) off",
    "axis-on": "Learned axis, S(C) on",
    "raw6-off": "Corrected Raw6, S(C) off (archived)",
    "raw6-on": "Corrected Raw6, S(C) on",
    "psd-off-64": "Historical PSD, smoothness off, step 64",
    "psd-on-64": "Historical PSD, smoothness on, step 64",
    "psd-off-1024": "Historical PSD, smoothness off, step 1024",
    "target": "Target smile",
}
COLORS = {
    "axis-off": "#43845b",
    "axis-on": "#8aa53c",
    "raw6-off": "#687787",
    "raw6-on": "#256f9c",
    "psd-off-64": "#d68a31",
    "psd-on-64": "#8b3f78",
    "psd-off-1024": "#b64c35",
    "target": "#202020",
}
PRIMARY_RESIDUAL = "primary_union_normal_residual_highpass_5mm_rms_mm"
PRIMARY_DISPLACEMENT = "primary_union_normal_displacement_highpass_5mm_rms_mm"
PRIMARY_LOW_FREQUENCY = "primary_union_low_frequency_normal_target_projection"


class Config(cherries.BaseConfig):
    """Inputs for either archived reuse or the completed comparison."""

    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    phase: Literal["reuse", "full"] = "full"
    axis_off_dir: Path = GROUP / "data/24-learned-axis"
    axis_on_dir: Path = GROUP / "data/25-learned-axis-smooth"
    raw6_on_dir: Path = GROUP / "data/28-raw6-smooth"
    reuse_dir: Path = GROUP / "data/40-reuse-metrics"
    output_dir: Path = GROUP / "data/40-comparison"


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _digest(path),
    }


def _snapshot_source(source: Path, destination: Path) -> dict[str, Any]:
    source = source.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source.read_bytes())
    destination.chmod(0o444)
    live = _record(source)
    snapshot = _record(destination)
    if live["sha256"] != snapshot["sha256"]:
        raise ValueError(f"source snapshot differs from live source: {source}")
    return {"snapshot": snapshot, "live_at_generation": live}


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refuse to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _parse_cell(value: str) -> Any:
    if value == "":
        return None
    if value == "True":
        return True
    if value == "False":
        return False
    try:
        return float(value)
    except ValueError:
        return value


def _read_table(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as stream:
        rows = [
            {key: _parse_cell(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows = _read_table(path)
    steps = np.asarray([int(row["step"]) for row in rows], dtype=np.int64)
    if np.any(np.diff(steps) <= 0):
        raise ValueError(f"trace steps are not strictly increasing: {path}")
    return rows


def _smoothness(metrics: StudyMetrics, field: np.ndarray) -> dict[str, float]:
    field = np.asarray(field, dtype=np.float64)
    expected = (len(metrics.active_ids), 3, 3)
    if field.shape != expected:
        raise ValueError(f"matrix field has shape {field.shape}; expected {expected}")
    if not np.isfinite(field).all():
        raise FloatingPointError("matrix field is non-finite")
    jump_squared = np.sum(
        (field[metrics.graph_i] - field[metrics.graph_j]) ** 2, axis=(1, 2)
    )
    weighted_sum = float(np.sum(metrics.graph_weight * jump_squared))
    volume = float(metrics.active_volume.sum())
    return {
        "same_muscle_jump_frobenius_rms": math.sqrt(
            weighted_sum / float(metrics.graph_weight.sum())
        ),
        "same_muscle_dirichlet_density_per_m2": weighted_sum / volume,
        "smoothness": SMOOTHNESS_LENGTH_M**2 * weighted_sum / volume,
    }


def _validate_saved_common(
    saved: Any, metrics: StudyMetrics, path: Path
) -> tuple[int, np.ndarray]:
    required = {"step", "u", "active_ids", "solver_valid"}
    missing = required - set(saved.files)
    if missing:
        raise ValueError(f"{path} lacks saved fields: {sorted(missing)}")
    if not bool(saved["solver_valid"]):
        raise ValueError(f"saved state is solver-invalid: {path}")
    if not np.array_equal(saved["active_ids"], metrics.active_ids):
        raise ValueError(f"active IDs differ from frozen fixture: {path}")
    if "rest_points" in saved.files and not np.array_equal(
        saved["rest_points"], metrics.rest
    ):
        raise ValueError(f"rest points differ from frozen fixture: {path}")
    u = np.asarray(saved["u"], dtype=np.float64)
    if u.shape != metrics.rest.shape or not np.isfinite(u).all():
        raise ValueError(f"invalid displacement in {path}")
    return int(saved["step"]), u


def _recompute_checkpoint(
    case: str, path: Path, metrics: StudyMetrics
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as saved:
        step, u = _validate_saved_common(saved, metrics, path)
        if case == "raw6-off":
            if "Ainv" not in saved.files:
                raise ValueError(f"Raw6 state lacks Ainv: {path}")
            b = np.asarray(saved["Ainv"], dtype=np.float64)
            if b.shape != (len(metrics.active_ids), 3, 3):
                raise ValueError(f"invalid Ainv shape in {path}")
            c = b - np.eye(3)
            z = b @ np.swapaxes(b, 1, 2) - np.eye(3)
        elif case.startswith("psd-"):
            if "Q" not in saved.files:
                raise ValueError(f"PSD state lacks Q: {path}")
            q_mpa = np.asarray(saved["Q"], dtype=np.float64)
            z = q_mpa / MU_MPA
            c = None
        else:
            raise ValueError(f"unsupported archived case: {case}")
    row: dict[str, Any] = {
        "case": case,
        "step": step,
        "source": str(path.resolve()),
        "source_sha256": _digest(path),
        **metrics.evaluate(u, z),
    }
    z_smooth = _smoothness(metrics, z)
    row.update({f"z_{key}": value for key, value in z_smooth.items()})
    if c is not None:
        c_smooth = _smoothness(metrics, c)
        row.update({f"c_{key}": value for key, value in c_smooth.items()})
        b_eigenvalues = np.linalg.eigvalsh(b)
        row["b_nonpositive_eigenvalue_cells"] = int(
            np.any(b_eigenvalues <= 0.0, axis=1).sum()
        )
        row["b_eigen_min"] = float(b_eigenvalues.min())
        row["b_eigen_max"] = float(b_eigenvalues.max())
    return row


def _checkpoint_files(directory: Path) -> list[Path]:
    files = sorted(directory.glob("step-*.npz"))
    if not files:
        raise FileNotFoundError(f"no full step checkpoints in {directory}")
    return files


def _archived_full_metrics(metrics: StudyMetrics) -> list[dict[str, Any]]:
    sources = {
        "raw6-off": RAW6_OFF,
        "psd-off-64": PSD_OFF,
        "psd-on-64": PSD_SMOOTH,
        "psd-off-1024": PSD_1024,
    }
    rows = []
    for case, directory in sources.items():
        rows.extend(
            _recompute_checkpoint(case, path, metrics)
            for path in _checkpoint_files(directory)
        )
    return rows


def _archived_raw6_trace(metrics: StudyMetrics) -> list[dict[str, Any]]:
    original = _read_rows(RAW6_OFF / "trace.csv")
    full_rows = {
        int(row["step"]): row
        for row in (
            _recompute_checkpoint("raw6-off", path, metrics)
            for path in _checkpoint_files(RAW6_OFF)
        )
    }
    rows: list[dict[str, Any]] = []
    for row in original:
        step = int(row["step"])
        surface_path = RAW6_OFF / f"surface-{step:04d}.npz"
        with np.load(surface_path, allow_pickle=False) as saved:
            if int(saved["step"]) != step:
                raise ValueError(f"surface step mismatch: {surface_path}")
            if not np.array_equal(saved["point_ids"], metrics.skin_ids):
                raise ValueError(f"surface point IDs differ: {surface_path}")
            surface = metrics.evaluate_surface(saved["u"])
        combined = {
            **row,
            **surface,
            "case": "raw6-off",
            "surface_source": str(surface_path.resolve()),
            "surface_sha256": _digest(surface_path),
            "full_checkpoint": step in full_rows,
            "B_eigenvalue_min": row.get("A_eigen_min"),
            "B_eigenvalue_max": row.get("A_eigen_max"),
            "B_nonpositive_eigenvalue_cells": row.get("non_spd_active_cells"),
        }
        if step in full_rows:
            combined["smoothness_C"] = full_rows[step]["c_smoothness"]
            combined["smoothness_Z"] = full_rows[step]["z_smoothness"]
            combined["z_eigen_min"] = full_rows[step]["z_eigen_min"]
            combined["z_eigen_max"] = full_rows[step]["z_eigen_max"]
            combined["full_state_source"] = full_rows[step]["source"]
            combined["full_state_sha256"] = full_rows[step]["source_sha256"]
        rows.append(combined)
    return rows


def _reuse_validation(full: list[dict[str, Any]]) -> dict[str, Any]:
    directories = {
        "raw6-off": RAW6_OFF,
        "psd-off-64": PSD_OFF,
        "psd-on-64": PSD_SMOOTH,
        "psd-off-1024": PSD_1024,
    }
    checks: dict[str, Any] = {}
    for case, directory in directories.items():
        original = {
            int(row["step"]): row for row in _read_rows(directory / "trace.csv")
        }
        rows = [row for row in full if row["case"] == case]
        fit_differences = [
            abs(
                float(row["fit_rms_mm"])
                - float(original[int(row["step"])]["fit_rms_mm"])
            )
            for row in rows
        ]
        if max(fit_differences) > 1e-12:
            raise ValueError(f"recomputed fit differs from archived trace: {case}")
        checks[case] = {
            "full_checkpoint_count": len(rows),
            "recomputed_fit_rms_max_abs_difference_mm": max(fit_differences),
        }
        if case == "raw6-off":
            projection_differences = [
                abs(
                    float(row["target_projection"])
                    - float(original[int(row["step"])]["target_projection"])
                )
                for row in rows
            ]
            if max(projection_differences) > 1e-12:
                raise ValueError("recomputed Raw6 target projection differs")
            checks[case]["target_projection_max_abs_difference"] = max(
                projection_differences
            )
    return {
        "status": "passed",
        "checks": checks,
        "projection_semantics": (
            "New/common target_projection is unweighted over finite IsFace. "
            "Historical PSD trace target_projection is area-weighted and is "
            "therefore not expected to reproduce; saved u is rescored instead."
        ),
    }


def _prepare_reuse(output: Path) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    source_receipts = {
        "comparison": _snapshot_source(
            Path(__file__), output / "sources/40-compare.py"
        ),
        "metrics": _snapshot_source(
            GROUP / "src/study_metrics.py", output / "sources/study_metrics.py"
        ),
    }
    metrics = StudyMetrics()
    full = _archived_full_metrics(metrics)
    raw6_trace = _archived_raw6_trace(metrics)
    validation = _reuse_validation(full)
    full_path = output / "full-checkpoint-metrics.csv"
    raw6_trace_path = output / "raw6-off-trace.csv"
    _write_csv(full_path, full)
    _write_csv(raw6_trace_path, raw6_trace)

    sources = {
        "raw6-off": RAW6_OFF,
        "psd-off-64": PSD_OFF,
        "psd-on-64": PSD_SMOOTH,
        "psd-off-1024": PSD_1024,
    }
    manifest = {
        "status": "completed_cpu_only_archived_recomputation",
        "scope": (
            "No forward, adjoint, or optimizer run. Full archived states are "
            "recomputed with frozen StudyMetrics; the archived Raw6 surface "
            "trajectory is augmented with frozen surface-only metrics."
        ),
        "metric_provenance": metrics.provenance,
        "conversions": {
            "raw6-off": "B=Ainv; C=B-I; Z=B@B.T-I",
            "old-psd": (
                "Z=Q/mu using the saved 3x3 Q field and "
                f"mu={MU_MPA:.17g} MPa; saved six-coordinate q is not used"
            ),
            "S_C_vs_S_Z": (
                "S_C is available only where an actual C control field exists; "
                "historical general-rank PSD states receive S(Z) only"
            ),
        },
        "labels": {
            "raw6-off": (
                "controlled archived off arm only for Raw6-on after exact "
                "starting-state verification; contextual for other models"
            ),
            "psd-off-64": "historical equal-budget PSD off arm; contextual",
            "psd-on-64": "historical equal-budget PSD smooth arm; contextual",
            "psd-off-1024": (
                "historical long unsmoothed PSD continuation; contextual; no "
                "matched smooth 1024 arm"
            ),
        },
        "inputs": {
            case: {
                "summary": _record(directory / "summary.json"),
                "provenance": _record(directory / "provenance.json"),
                "trace": _record(directory / "trace.csv"),
                "full_checkpoints": [
                    _record(path) for path in _checkpoint_files(directory)
                ],
            }
            for case, directory in sources.items()
        },
        "outputs": {
            "full_checkpoint_metrics": _record(full_path),
            "raw6_off_trace": _record(raw6_trace_path),
        },
        "validation": validation,
        "raw6_matching_constraint": (
            "Only archived Raw6-off rows with full C checkpoints can enter the "
            "matched success decision because S_C is unavailable from a surface-only file."
        ),
        "source": source_receipts,
    }
    manifest_path = output / "reuse-manifest.json"
    _write_json(manifest_path, manifest)
    summary = {
        "status": manifest["status"],
        "full_checkpoint_rows": len(full),
        "raw6_surface_rows": len(raw6_trace),
        "manifest": _record(manifest_path),
        "source": source_receipts,
    }
    _write_json(output / "summary.json", summary)
    return summary


def _normalize_new_trace(path: Path) -> list[dict[str, Any]]:
    rows = _read_rows(path)
    required = {
        "step",
        "fit_rms_mm",
        "motion_rms_mm",
        "target_projection",
        PRIMARY_RESIDUAL,
        PRIMARY_DISPLACEMENT,
        PRIMARY_LOW_FREQUENCY,
        "smoothness_C",
        "smoothness_Z",
        "forward_success",
        "adjoint_success",
    }
    missing = required - rows[0].keys()
    if missing:
        raise ValueError(f"new trace lacks columns {sorted(missing)}: {path}")
    for row in rows:
        if row["forward_success"] is not True or row["adjoint_success"] is not True:
            raise ValueError(f"new trace contains solver-invalid state: {path}")
    return rows


def _reuse_source_drift(reuse: Path) -> dict[str, Any]:
    manifest = json.loads((reuse / "reuse-manifest.json").read_text())
    live_sources = {
        "comparison": Path(__file__),
        "metrics": GROUP / "src/study_metrics.py",
    }
    result: dict[str, Any] = {}
    for name, live_path in live_sources.items():
        receipt = manifest["source"][name]["snapshot"]
        snapshot = _record(Path(receipt["path"]))
        if snapshot["sha256"] != receipt["sha256"]:
            raise ValueError(f"reuse source snapshot hash differs: {name}")
        live = _record(live_path)
        result[name] = {
            "snapshot": snapshot,
            "live_at_comparison": live,
            "live_source_changed_since_reuse": (live["sha256"] != snapshot["sha256"]),
        }
    return result


def _merge_archived_raw6_control(reuse: Path) -> list[dict[str, Any]]:
    rows = _read_rows(reuse / "raw6-off-trace.csv")
    full = {
        int(row["step"]): row
        for row in _read_table(reuse / "full-checkpoint-metrics.csv")
        if row["case"] == "raw6-off"
    }
    for row in rows:
        row["B_eigenvalue_min"] = row.get("A_eigen_min")
        row["B_eigenvalue_max"] = row.get("A_eigen_max")
        row["B_nonpositive_eigenvalue_cells"] = row.get("non_spd_active_cells")
        checkpoint = full.get(int(row["step"]))
        if checkpoint is not None:
            row["z_eigen_min"] = checkpoint["z_eigen_min"]
            row["z_eigen_max"] = checkpoint["z_eigen_max"]
    return rows


def _pair_actual_states(
    off_case: str,
    on_case: str,
    off_rows: list[dict[str, Any]],
    on_rows: list[dict[str, Any]],
    *,
    off_full_controls_only: bool,
) -> dict[str, Any]:
    off = [
        row
        for row in off_rows
        if int(row["step"]) > 0
        and (not off_full_controls_only or row.get("full_checkpoint") is True)
        and row.get("smoothness_C") is not None
    ]
    on = [
        row
        for row in on_rows
        if int(row["step"]) > 0 and row.get("smoothness_C") is not None
    ]
    if not off or not on:
        return {
            "available": False,
            "selection": (
                "Insufficient post-update solver-valid states for an actual-state "
                "match; no state is substituted or fabricated."
            ),
            "reason": "insufficient_post_update_states",
            "eligible_state_counts": {off_case: len(off), on_case: len(on)},
            "fit_tolerance_mm": FIT_TOLERANCE_MM,
            "motion_tolerance_mm": MOTION_TOLERANCE_MM,
            "interpolation": False,
            "off_full_controls_only": off_full_controls_only,
        }
    feasible: list[tuple[tuple[float, ...], dict[str, Any], dict[str, Any]]] = []
    nearest: tuple[tuple[float, ...], dict[str, Any], dict[str, Any]] | None = None
    for left in off:
        for right in on:
            fit_delta = abs(float(left["fit_rms_mm"]) - float(right["fit_rms_mm"]))
            motion_delta = abs(
                float(left["motion_rms_mm"]) - float(right["motion_rms_mm"])
            )
            mean_fit = 0.5 * (float(left["fit_rms_mm"]) + float(right["fit_rms_mm"]))
            distance_squared = (fit_delta / FIT_TOLERANCE_MM) ** 2 + (
                motion_delta / MOTION_TOLERANCE_MM
            ) ** 2
            nearest_score = (
                distance_squared,
                mean_fit,
                int(left["step"]),
                int(right["step"]),
            )
            candidate = (nearest_score, left, right)
            if nearest is None or candidate[0] < nearest[0]:
                nearest = candidate
            if fit_delta <= FIT_TOLERANCE_MM and motion_delta <= MOTION_TOLERANCE_MM:
                score = (
                    mean_fit,
                    distance_squared,
                    int(left["step"]) + int(right["step"]),
                    int(left["step"]),
                    int(right["step"]),
                )
                feasible.append((score, left, right))
    if not feasible:
        assert nearest is not None
        _, left, right = nearest
        return {
            "available": False,
            "selection": (
                "No post-update actual saved-state pair satisfies both frozen "
                "tolerances. Nearest values are diagnostics only and are not selected."
            ),
            "fit_tolerance_mm": FIT_TOLERANCE_MM,
            "motion_tolerance_mm": MOTION_TOLERANCE_MM,
            "nearest_diagnostic": {
                "steps": {off_case: int(left["step"]), on_case: int(right["step"])},
                "fit_delta_mm": abs(
                    float(left["fit_rms_mm"]) - float(right["fit_rms_mm"])
                ),
                "motion_delta_mm": abs(
                    float(left["motion_rms_mm"]) - float(right["motion_rms_mm"])
                ),
            },
            "interpolation": False,
            "off_full_controls_only": off_full_controls_only,
        }
    _, left, right = min(feasible, key=lambda item: item[0])
    residual_off = float(left[PRIMARY_RESIDUAL])
    residual_on = float(right[PRIMARY_RESIDUAL])
    low_off = float(left[PRIMARY_LOW_FREQUENCY])
    low_on = float(right[PRIMARY_LOW_FREQUENCY])
    residual_reduction = (
        None if residual_off == 0.0 else 1.0 - residual_on / residual_off
    )
    low_frequency_ratio = None if low_off <= 0.0 else low_on / low_off
    smoothness_reduced = float(right["smoothness_C"]) < float(left["smoothness_C"])
    useful = (
        residual_reduction is not None
        and residual_reduction >= PRIMARY_EFFECT_FRACTION
        and smoothness_reduced
        and low_frequency_ratio is not None
        and low_frequency_ratio >= LOW_FREQUENCY_RETENTION
    )
    return {
        "available": True,
        "selection": (
            "Lowest mean fit among post-update actual saved-state pairs within "
            "both tolerances; then smallest normalized mismatch and earliest steps."
        ),
        "states": {off_case: left, on_case: right},
        "fit_delta_mm": abs(float(left["fit_rms_mm"]) - float(right["fit_rms_mm"])),
        "motion_delta_mm": abs(
            float(left["motion_rms_mm"]) - float(right["motion_rms_mm"])
        ),
        "fit_tolerance_mm": FIT_TOLERANCE_MM,
        "motion_tolerance_mm": MOTION_TOLERANCE_MM,
        "residual_highpass_reduction_fraction": residual_reduction,
        "smoothness_C_reduced": smoothness_reduced,
        "low_frequency_projection_ratio": low_frequency_ratio,
        "useful_surface_effect": useful,
        "effect_threshold_fraction": PRIMARY_EFFECT_FRACTION,
        "low_frequency_retention_threshold": LOW_FREQUENCY_RETENTION,
        "interpolation": False,
        "off_full_controls_only": off_full_controls_only,
    }


def _row_at(rows: list[dict[str, Any]], step: int) -> dict[str, Any]:
    matches = [row for row in rows if int(row["step"]) == step]
    if len(matches) != 1:
        raise ValueError(f"expected one trace row at step {step}, got {len(matches)}")
    return matches[0]


def _common_update_pair(
    off_case: str,
    on_case: str,
    off: list[dict[str, Any]],
    on: list[dict[str, Any]],
    statuses: dict[str, str],
) -> dict[str, Any]:
    off_step = int(off[-1]["step"])
    on_step = int(on[-1]["step"])
    common_step = min(off_step, on_step)
    payload: dict[str, Any] = {
        "available": common_step > 0,
        "selection": (
            "Largest common evaluated update count from solver-valid trace rows; "
            "longer partial endpoints are reported separately."
        ),
        "common_step": common_step,
        "endpoint_steps": {off_case: off_step, on_case: on_step},
        "experiment_statuses": statuses,
        "comparison_status": (
            "equal_endpoints"
            if off_step == on_step
            else "common_step_from_unequal_endpoints"
        ),
        "longer_partial_endpoints": {
            case: rows[-1]
            for case, rows in ((off_case, off), (on_case, on))
            if int(rows[-1]["step"]) > common_step
        },
        "interpolation": False,
    }
    if common_step <= 0:
        payload.update(
            reason="no_positive_common_update",
            selection=(
                "Only step zero is common, so no primary post-update equal-count "
                "comparison is selected and no state is substituted or fabricated."
            ),
        )
        return payload
    payload["states"] = {
        off_case: _row_at(off, common_step),
        on_case: _row_at(on, common_step),
    }
    left = payload["states"][off_case]
    right = payload["states"][on_case]
    fit_delta = abs(float(right["fit_rms_mm"]) - float(left["fit_rms_mm"]))
    motion_delta = abs(float(right["motion_rms_mm"]) - float(left["motion_rms_mm"]))
    residual_off = float(left[PRIMARY_RESIDUAL])
    smoothness_c_off = float(left["smoothness_C"])
    smoothness_z_off = float(left["smoothness_Z"])
    low_frequency_off = float(left[PRIMARY_LOW_FREQUENCY])
    payload.update(
        fit_delta_mm=fit_delta,
        motion_delta_mm=motion_delta,
        within_matched_fit_motion_tolerances=(
            fit_delta <= FIT_TOLERANCE_MM and motion_delta <= MOTION_TOLERANCE_MM
        ),
        descriptive_on_minus_off={
            "primary_residual_highpass_mm": (
                float(right[PRIMARY_RESIDUAL]) - residual_off
            ),
            "primary_residual_highpass_reduction_fraction": (
                None
                if residual_off == 0.0
                else 1.0 - float(right[PRIMARY_RESIDUAL]) / residual_off
            ),
            "primary_displacement_highpass_mm": (
                float(right[PRIMARY_DISPLACEMENT]) - float(left[PRIMARY_DISPLACEMENT])
            ),
            "smoothness_C_reduction_fraction": (
                None
                if smoothness_c_off == 0.0
                else 1.0 - float(right["smoothness_C"]) / smoothness_c_off
            ),
            "smoothness_Z_reduction_fraction": (
                None
                if smoothness_z_off == 0.0
                else 1.0 - float(right["smoothness_Z"]) / smoothness_z_off
            ),
            "low_frequency_projection_ratio": (
                None
                if low_frequency_off <= 0.0
                else float(right[PRIMARY_LOW_FREQUENCY]) / low_frequency_off
            ),
        },
    )
    return payload


def _load_surface(
    directory: Path, step: int, point_ids: np.ndarray
) -> tuple[np.ndarray, Path]:
    path = directory / f"surface-{step:04d}.npz"
    with np.load(path, allow_pickle=False) as saved:
        if int(saved["step"]) != step:
            raise ValueError(f"surface step receipt mismatch: {path}")
        if not np.array_equal(saved["point_ids"], point_ids):
            raise ValueError(f"surface point IDs differ from fixture: {path}")
        u = np.asarray(saved["u"], dtype=np.float64)
    if u.shape != (len(point_ids), 3) or not np.isfinite(u).all():
        raise ValueError(f"invalid surface displacement: {path}")
    return u, path


def _add_lights(plotter: pv.Plotter, camera: dict[str, Any]) -> None:
    focus = np.asarray(camera["focal_point"], dtype=np.float64)
    backward = np.asarray(camera["position"], dtype=np.float64) - focus
    backward /= np.linalg.norm(backward)
    right = np.cross(np.asarray(camera["view_up"], dtype=np.float64), backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    key = focus + 0.3 * (0.72 * right + 0.35 * up + 0.60 * backward)
    fill = focus + 0.3 * backward
    for position, intensity in ((key, 0.85), (fill, 0.20)):
        plotter.add_light(
            pv.Light(
                position=position,
                focal_point=focus,
                intensity=intensity,
                light_type="scene light",
                positional=False,
            ),
            only_active=True,
        )


def _set_camera(plotter: pv.Plotter, camera: dict[str, Any]) -> None:
    plotter.enable_parallel_projection()
    plotter.camera.position = camera["position"]
    plotter.camera.focal_point = camera["focal_point"]
    plotter.camera.up = camera["view_up"]
    plotter.camera.parallel_scale = camera["parallel_scale"]
    plotter.set_background(BACKGROUND)
    plotter.reset_camera_clipping_range()


def _render_surface(
    base: pv.PolyData,
    displacement: np.ndarray,
    camera: dict[str, Any],
    path: Path,
) -> None:
    mesh = base.copy(deep=True)
    mesh.points = np.asarray(base.points, dtype=np.float64) + displacement
    plotter = pv.Plotter(off_screen=True, window_size=(900, 900), lighting="none")
    actor = plotter.add_mesh(
        mesh,
        color="#eeeeea",
        smooth_shading=False,
        ambient=0.20,
        diffuse=0.80,
        specular=0.0,
    )
    if actor.GetProperty().GetInterpolation() != 0:
        raise AssertionError("comparison geometry must use flat shading")
    _add_lights(plotter, camera)
    _set_camera(plotter, camera)
    path.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(path)
    plotter.close()


def _surface_highpass_fields(
    metrics: StudyMetrics, u_skin: np.ndarray
) -> dict[str, np.ndarray]:
    normal = np.einsum("ij,ij->i", u_skin, metrics.normals)
    low = metrics._lowpass(normal)  # noqa: SLF001  # Frozen recorded operator.
    high_displacement = normal - low
    high_target = metrics.target_normal - metrics.target_normal_lowpass
    return {
        "residual": 1000.0 * (high_displacement - high_target),
        "displacement": 1000.0 * high_displacement,
    }


def _render_scalar_surface(
    base: pv.PolyData,
    displacement: np.ndarray,
    scalar_mm: np.ndarray,
    camera: dict[str, Any],
    path: Path,
    *,
    limit_mm: float,
    title: str,
) -> None:
    mesh = base.copy(deep=True)
    mesh.points = np.asarray(base.points, dtype=np.float64) + displacement
    mesh.point_data["value_mm"] = scalar_mm
    plotter = pv.Plotter(off_screen=True, window_size=(1000, 900), lighting="none")
    plotter.add_mesh(
        mesh,
        scalars="value_mm",
        cmap="coolwarm",
        clim=(-limit_mm, limit_mm),
        smooth_shading=False,
        ambient=0.30,
        diffuse=0.70,
        scalar_bar_args={
            "title": title,
            "vertical": True,
            "color": "white",
            "position_x": 0.82,
            "position_y": 0.12,
            "width": 0.10,
            "height": 0.68,
            "title_font_size": 16,
            "label_font_size": 14,
            "fmt": "%.3g",
        },
    )
    _add_lights(plotter, camera)
    _set_camera(plotter, camera)
    path.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(path)
    plotter.close()


def _render_comparison_plate(
    images: list[Path], titles: list[str], title: str, path: Path
) -> None:
    if len(images) != 3 or len(titles) != 3:
        raise ValueError("comparison plate requires target, off, and on panels")
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), constrained_layout=True)
    figure.patch.set_facecolor(BACKGROUND)
    for axis, image, panel_title in zip(axes, images, titles, strict=True):
        axis.imshow(plt.imread(image))
        axis.set_title(panel_title, color="white", fontsize=10)
        axis.set_axis_off()
    figure.suptitle(title, color="white", fontsize=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def _triangle_segments(
    points: np.ndarray, triangles: np.ndarray, y: float
) -> np.ndarray:
    """Return exact affine triangle-plane intersections without joining."""
    segments: list[np.ndarray] = []
    for triangle in points[triangles]:
        signed = triangle[:, 1] - y
        hits: list[np.ndarray] = []
        for left, right in ((0, 1), (1, 2), (2, 0)):
            a, b = triangle[left], triangle[right]
            sa, sb = signed[left], signed[right]
            if sa == 0.0 and sb == 0.0:
                continue
            if sa == 0.0:
                hits.append(a)
            elif sb == 0.0:
                hits.append(b)
            elif (sa < 0.0) != (sb < 0.0):
                hits.append(a + (-sa / (sb - sa)) * (b - a))
        unique = [
            hit
            for index, hit in enumerate(hits)
            if not any(
                np.allclose(hit, prior, rtol=0.0, atol=1e-13) for prior in hits[:index]
            )
        ]
        if len(unique) == 2:
            segment = np.stack(unique)
            if (
                segment[:, 0].max() >= 1.414
                and segment[:, 0].min() <= 1.460
                and segment[:, 2].max() >= 0.040
            ):
                segments.append(segment)
    return np.asarray(segments, dtype=np.float64)


def _render_nlf_sections(
    base: pv.PolyData,
    states: list[tuple[str, np.ndarray, str]],
    path: Path,
) -> None:
    planes = (2.170, 2.180, 2.190)
    triangles = np.asarray(base.faces, dtype=np.int64).reshape(-1, 4)[:, 1:]
    figure, axes = plt.subplots(3, 1, figsize=(8, 9), constrained_layout=True)
    for axis, y in zip(axes, planes, strict=True):
        for label, u, color in states:
            segments = _triangle_segments(
                np.asarray(base.points, dtype=np.float64) + u, triangles, y
            )
            for index, segment in enumerate(segments):
                axis.plot(
                    1000.0 * segment[:, 0],
                    1000.0 * segment[:, 2],
                    color=color,
                    linewidth=0.8,
                    label=label if index == 0 else None,
                )
        axis.set(
            title=f"Exact skin-triangle section at y = {1000.0 * y:.0f} mm",
            xlim=(1414, 1460),
            ylim=(40, 115),
            ylabel="z (mm)",
        )
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("x (mm)")
    axes[0].legend(loc="upper left", frameon=False, fontsize=7)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def _plot_new_trajectories(
    traces: dict[str, list[dict[str, Any]]], output: Path
) -> list[str]:
    paths: list[str] = []
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for case in ALL_CASES:
        rows = traces[case]
        step = [row["step"] for row in rows]
        axes[0, 0].plot(
            step,
            [row["fit_rms_mm"] for row in rows],
            color=COLORS[case],
            label=LABELS[case],
        )
        axes[0, 1].plot(
            step,
            [row["motion_rms_mm"] for row in rows],
            color=COLORS[case],
            label=LABELS[case],
        )
        axes[1, 0].plot(
            step,
            [row[PRIMARY_RESIDUAL] for row in rows],
            color=COLORS[case],
            label=LABELS[case],
        )
        axes[1, 1].plot(
            step,
            [row[PRIMARY_LOW_FREQUENCY] for row in rows],
            color=COLORS[case],
            label=LABELS[case],
        )
    settings = (
        (axes[0, 0], "Fit RMS (mm)", "Target fit"),
        (axes[0, 1], "Motion RMS (mm)", "Expression motion"),
        (axes[1, 0], "Union residual HP RMS (mm)", "Primary surface score"),
        (
            axes[1, 1],
            "Union low-frequency target projection",
            "Low-frequency target-motion retention",
        ),
    )
    for axis, ylabel, title in settings:
        axis.set(xlabel="Adam updates", ylabel=ylabel, title=title)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    path = output / "new-pairs-trajectories.png"
    figure.savefig(path, dpi=210)
    plt.close(figure)
    paths.append(path.name)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for case in ALL_CASES:
        rows = traces[case]
        for axis, key, ylabel in (
            (axes[0], "smoothness_C", "S(C)"),
            (axes[1], "smoothness_Z", "S(Z)"),
        ):
            available = [row for row in rows if row.get(key) is not None]
            if available:
                axis.plot(
                    [row["step"] for row in available],
                    [row[key] for row in available],
                    color=COLORS[case],
                    label=LABELS[case],
                )
                axis.set_ylabel(ylabel)
    for axis, title in zip(
        axes,
        ("Optimized C variation", "Effective activation Z variation"),
        strict=True,
    ):
        positive = [
            float(line_value)
            for line in axis.lines
            for line_value in line.get_ydata()
            if float(line_value) > 0.0
        ]
        if positive and max(positive) / min(positive) >= 100.0:
            axis.set_yscale("symlog", linthresh=0.5 * min(positive))
        axis.set(xlabel="Adam updates", title=title)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    path = output / "new-pairs-activation-variation.png"
    figure.savefig(path, dpi=210)
    plt.close(figure)
    paths.append(path.name)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for case in ALL_CASES:
        rows = traces[case]
        axes[0].plot(
            [row["fit_rms_mm"] for row in rows],
            [row[PRIMARY_RESIDUAL] for row in rows],
            color=COLORS[case],
            label=LABELS[case],
        )
        axes[1].plot(
            [row["motion_rms_mm"] for row in rows],
            [row[PRIMARY_RESIDUAL] for row in rows],
            color=COLORS[case],
            label=LABELS[case],
        )
    axes[0].set(
        xlabel="Fit RMS (mm)",
        ylabel="Union residual HP RMS (mm)",
        title="Surface error versus fit",
    )
    axes[1].set(
        xlabel="Motion RMS (mm)",
        ylabel="Union residual HP RMS (mm)",
        title="Surface error versus motion",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    path = output / "new-pairs-fit-motion-surface-tradeoff.png"
    figure.savefig(path, dpi=210)
    plt.close(figure)
    paths.append(path.name)
    return paths


def _compact_row(selection: str, case: str, row: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "step",
        "fit_rms_mm",
        "motion_rms_mm",
        "target_projection",
        PRIMARY_RESIDUAL,
        PRIMARY_DISPLACEMENT,
        PRIMARY_LOW_FREQUENCY,
        "smoothness_C",
        "smoothness_Z",
        "roi_right_nose_to_mouth_fit_vector_rms_mm",
        "detF_min",
        "detF_max",
        "inverted_all_cells",
        "inverted_active_cells",
        "shortening_fraction_p50",
        "shortening_fraction_p90",
        "shortening_fraction_p99",
        "shortening_fraction_p100",
        "z_eigen_min",
        "z_eigen_max",
        "B_eigenvalue_min",
        "B_eigenvalue_max",
        "B_nonpositive_eigenvalue_cells",
    )
    return {
        "selection": selection,
        "case": case,
        **{key: row.get(key) for key in keys},
    }


def _run_outcome(
    case: str, summary: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, Any]:
    last_step = int(rows[-1]["step"])
    if int(summary["last_evaluated_step"]) != last_step:
        raise ValueError(f"summary/trace endpoint mismatch: {case}")
    best_step = summary.get("best_step")
    failure = summary.get("failure")
    return {
        "status": str(summary["status"]),
        "last_evaluated_step": last_step,
        "last_state": _compact_row("last-evaluated", case, rows[-1]),
        "best_fit_step": best_step,
        "best_fit_state": (
            None
            if best_step is None
            else _compact_row("best-fit", case, _row_at(rows, int(best_step)))
        ),
        "failure": (
            None
            if failure is None
            else {
                key: failure.get(key)
                for key in ("type", "message", "step")
                if key in failure
            }
        ),
    }


def _full_comparison(cfg: Config) -> dict[str, Any]:
    output = cfg.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    reuse_source_drift = _reuse_source_drift(cfg.reuse_dir)
    fresh_dirs = {
        "axis-off": cfg.axis_off_dir,
        "axis-on": cfg.axis_on_dir,
        "raw6-on": cfg.raw6_on_dir,
    }
    traces = {
        case: _normalize_new_trace(directory / "trace.csv")
        for case, directory in fresh_dirs.items()
    }
    traces["raw6-off"] = _merge_archived_raw6_control(cfg.reuse_dir)
    summaries = {
        case: json.loads((directory / "summary.json").read_text())
        for case, directory in fresh_dirs.items()
    }
    raw6_off_summary = json.loads((RAW6_OFF / "summary.json").read_text())
    run_outcomes = {
        case: _run_outcome(case, summary, traces[case])
        for case, summary in summaries.items()
    }
    run_outcomes["raw6-off"] = _run_outcome(
        "raw6-off", raw6_off_summary, traces["raw6-off"]
    )
    pair_statuses = {
        "learned-axis": {
            case: str(summaries[case]["status"]) for case in ("axis-off", "axis-on")
        },
        "raw6": {
            "raw6-off": str(raw6_off_summary["status"]),
            "raw6-on": str(summaries["raw6-on"]["status"]),
        },
    }
    axis_common = _common_update_pair(
        "axis-off",
        "axis-on",
        traces["axis-off"],
        traces["axis-on"],
        pair_statuses["learned-axis"],
    )
    raw6_common = _common_update_pair(
        "raw6-off",
        "raw6-on",
        traces["raw6-off"],
        traces["raw6-on"],
        pair_statuses["raw6"],
    )
    matches = {
        "learned-axis": _pair_actual_states(
            "axis-off",
            "axis-on",
            traces["axis-off"],
            traces["axis-on"],
            off_full_controls_only=False,
        ),
        "raw6": _pair_actual_states(
            "raw6-off",
            "raw6-on",
            traces["raw6-off"],
            traces["raw6-on"],
            off_full_controls_only=True,
        ),
    }
    for name, match in matches.items():
        statuses = pair_statuses[name]
        match["experiment_statuses"] = statuses
        match["trajectory_scope"] = (
            "completed_declared_budgets"
            if all(status.startswith("completed_") for status in statuses.values())
            else "solver_valid_prefix_of_partial_or_failed_run"
        )
    selections: dict[str, dict[str, dict[str, Any]]] = {}
    for name, common in (
        ("learned-axis", axis_common),
        ("raw6", raw6_common),
    ):
        if common["available"]:
            selections[f"{name}-common-update"] = common["states"]
    for name, match in matches.items():
        if match["available"]:
            selections[f"{name}-matched"] = match["states"]

    selected_rows = [
        _compact_row(selection, case, row)
        for selection, by_case in selections.items()
        for case, row in by_case.items()
    ]
    selected_path = output / "selected-states.csv" if selected_rows else None
    if selected_path is not None:
        _write_csv(selected_path, selected_rows)
    plots = _plot_new_trajectories(traces, output)

    metrics = StudyMetrics()
    volume = pv.read(FIXTURE / "volume.vtu")
    skin = pv.read(FIXTURE / "skin.vtp")
    point_ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    target_u = np.asarray(volume.point_data["Smile"], dtype=np.float64)[point_ids]
    camera_payload = json.loads(CAMERAS.read_text())
    views = camera_payload["views"]
    if len(views) != 5:
        raise ValueError("frozen camera receipt must contain five views")
    directories = {**fresh_dirs, "raw6-off": RAW6_OFF}
    assets: dict[str, Any] = {}
    for selection, by_case in selections.items():
        states: dict[str, tuple[np.ndarray, Path]] = {
            case: _load_surface(directories[case], int(row["step"]), point_ids)
            for case, row in by_case.items()
        }
        selection_assets: dict[str, Any] = {"geometry": {}, "highpass": {}, "nlf": []}
        for case, (u, source) in states.items():
            case_paths = []
            for view in views:
                path = output / "geometry" / selection / case / f"{view['id']}.png"
                _render_surface(skin, u, view["camera"], path)
                case_paths.append(_record(path))
            selection_assets["geometry"][case] = {
                "source": _record(source),
                "step": int(by_case[case]["step"]),
                "files": case_paths,
            }
        target_paths = []
        for view in views:
            path = output / "geometry" / selection / "target" / f"{view['id']}.png"
            _render_surface(skin, target_u, view["camera"], path)
            target_paths.append(_record(path))
        selection_assets["geometry"]["target"] = {"files": target_paths}
        off_case = next(case for case in states if case.endswith("-off"))
        on_case = next(case for case in states if case.endswith("-on"))
        plate_paths = []
        for view in views:
            paths = [
                output / "geometry" / selection / "target" / f"{view['id']}.png",
                output / "geometry" / selection / off_case / f"{view['id']}.png",
                output / "geometry" / selection / on_case / f"{view['id']}.png",
            ]
            titles = ["Target"] + [
                (
                    f"{LABELS[case]}\nstep {int(by_case[case]['step'])}; "
                    f"fit {float(by_case[case]['fit_rms_mm']):.4f} mm; "
                    f"motion {float(by_case[case]['motion_rms_mm']):.4f} mm"
                )
                for case in (off_case, on_case)
            ]
            plate = output / "geometry" / selection / f"{view['id']}-target-off-on.png"
            _render_comparison_plate(
                paths,
                titles,
                f"{selection.replace('-', ' ')} · {view['id']}",
                plate,
            )
            plate_paths.append(_record(plate))
        selection_assets["geometry"]["comparison_plates"] = plate_paths

        if selection.endswith("matched"):
            fields = {
                case: _surface_highpass_fields(metrics, u)
                for case, (u, _) in states.items()
            }
            primary_view = views[0]
            for field_name in ("residual", "displacement"):
                limit = max(
                    float(
                        np.max(np.abs(values[field_name][metrics.primary_surface_mask]))
                    )
                    for values in fields.values()
                )
                if limit <= 0.0:
                    raise ValueError("high-pass color range is zero")
                for case, (u, _) in states.items():
                    path = output / "highpass" / selection / f"{case}-{field_name}.png"
                    _render_scalar_surface(
                        skin,
                        u,
                        fields[case][field_name],
                        primary_view["camera"],
                        path,
                        limit_mm=limit,
                        title=f"HP {field_name} (mm)",
                    )
                    selection_assets["highpass"].setdefault(field_name, {})[case] = {
                        "file": _record(path),
                        "shared_limit_mm": limit,
                    }
        section_states = [
            (LABELS[case], u, COLORS[case]) for case, (u, _) in states.items()
        ] + [(LABELS["target"], target_u, COLORS["target"])]
        overlay = output / "nlf" / f"{selection}-overlay.png"
        _render_nlf_sections(skin, section_states, overlay)
        selection_assets["nlf"].append(_record(overlay))
        for case, (u, _) in states.items():
            path = output / "nlf" / f"{selection}-{case}.png"
            _render_nlf_sections(
                skin,
                [
                    (LABELS[case], u, COLORS[case]),
                    (LABELS["target"], target_u, COLORS["target"]),
                ],
                path,
            )
            selection_assets["nlf"].append(_record(path))
        assets[selection] = selection_assets

    reused = _read_table(cfg.reuse_dir / "full-checkpoint-metrics.csv")
    context_keys = {
        ("psd-off-64", 64),
        ("psd-on-64", 64),
        ("psd-off-1024", 1024),
    }
    context = [
        row for row in reused if (str(row["case"]), int(row["step"])) in context_keys
    ]
    if len(context) != len(context_keys):
        raise ValueError("reuse metrics lack one or more historical PSD endpoints")
    context_rows = [
        {
            "case": row["case"],
            "step": row["step"],
            "fit_rms_mm": row["fit_rms_mm"],
            "motion_rms_mm": row["motion_rms_mm"],
            PRIMARY_RESIDUAL: row[PRIMARY_RESIDUAL],
            PRIMARY_DISPLACEMENT: row[PRIMARY_DISPLACEMENT],
            PRIMARY_LOW_FREQUENCY: row[PRIMARY_LOW_FREQUENCY],
            "z_smoothness": row["z_smoothness"],
            "source": row["source"],
            "source_sha256": row["source_sha256"],
        }
        for row in context
    ]
    context_path = output / "historical-context.csv"
    _write_csv(context_path, context_rows)

    figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for case in ALL_CASES:
        row = traces[case][-1]
        axis.scatter(
            row["fit_rms_mm"],
            row[PRIMARY_RESIDUAL],
            color=COLORS[case],
            label=LABELS[case],
            s=55,
        )
    for row in context_rows:
        case = str(row["case"])
        axis.scatter(
            row["fit_rms_mm"],
            row[PRIMARY_RESIDUAL],
            color=COLORS[case],
            label=LABELS[case],
            marker="D",
            s=48,
        )
    axis.set(
        xlabel="Uniform fit RMS (mm)",
        ylabel="Union residual HP RMS (mm)",
        title="Attained fit and surface error; historical PSD is contextual",
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7)
    context_plot = output / "historical-context-tradeoff.png"
    figure.savefig(context_plot, dpi=220)
    plt.close(figure)
    plots.append(context_plot.name)

    all_fresh_completed = all(
        str(summary["status"]).startswith("completed_")
        for summary in summaries.values()
    )
    summary = {
        "status": (
            "completed_saved_state_comparison"
            if all_fresh_completed
            else "completed_saved_state_comparison_with_partial_or_failed_runs"
        ),
        "primary_run_completion": (
            "all_fresh_runs_completed_declared_budgets"
            if all_fresh_completed
            else "one_or_more_fresh_runs_partial_or_failed"
        ),
        "scope": (
            "CPU-only comparison of evaluated trace rows and exact saved states; "
            "no physics solve, optimizer update, interpolation, deformation "
            "exaggeration, geometry smoothing, or joined section reconstruction."
        ),
        "primary_metric": {
            "name": PRIMARY_RESIDUAL,
            "definition": metrics.provenance["frequency_split"][
                "primary_surface_region"
            ],
            "effect_threshold_fraction": PRIMARY_EFFECT_FRACTION,
            "fit_tolerance_mm": FIT_TOLERANCE_MM,
            "motion_tolerance_mm": MOTION_TOLERANCE_MM,
            "low_frequency_retention": LOW_FREQUENCY_RETENTION,
        },
        "comparisons": {
            "learned-axis": ("one specified seed; no initialization-sensitivity claim"),
            "raw6": (
                "archived off arm is paired only after starting-state verification; "
                "off matching candidates require full C checkpoints"
            ),
            "historical_psd": (
                "contextual protocols only; not equal-start or equal-budget model arms"
            ),
        },
        "matches": matches,
        "run_outcomes": run_outcomes,
        "equal_update": {
            "learned-axis": axis_common,
            "raw6": raw6_common,
        },
        "inputs": {
            **{
                case: {
                    "trace": _record(directory / "trace.csv"),
                    "summary": _record(directory / "summary.json"),
                    "status": summaries[case]["status"],
                }
                for case, directory in fresh_dirs.items()
            },
            "raw6-off-reuse": _record(cfg.reuse_dir / "raw6-off-trace.csv"),
            "reuse-manifest": _record(cfg.reuse_dir / "reuse-manifest.json"),
            "reuse-source-drift": reuse_source_drift,
        },
        "outputs": {
            "selected_states": (
                None if selected_path is None else _record(selected_path)
            ),
            "historical_context": _record(context_path),
            "plots": [_record(output / path) for path in plots],
            "assets": assets,
        },
        "source": _record(Path(__file__)),
    }
    _write_json(output / "summary.json", summary)
    return summary


def main(cfg: Config) -> None:
    if cfg.phase == "reuse":
        summary = _prepare_reuse(cfg.output_dir)
        cherries.log_metric(
            "reuse/full_checkpoint_rows", summary["full_checkpoint_rows"]
        )
        cherries.log_metric("reuse/raw6_surface_rows", summary["raw6_surface_rows"])
    else:
        summary = _full_comparison(cfg)
        for name, match in summary["matches"].items():
            cherries.log_metric(f"matching/{name}/available", float(match["available"]))
            if match["available"]:
                cherries.log_metric(
                    f"matching/{name}/useful_surface_effect",
                    float(match["useful_surface_effect"]),
                )
    cherries.log_output(cfg.output_dir)


if __name__ == "__main__":
    cherries.main(
        main,
        profile=None if os.getenv("DEBUG") == "1" else ProfileCometNoCommit,
    )
