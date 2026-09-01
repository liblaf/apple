"""Strict post-hoc folding analysis for the 2^4 pork-factorial study."""

from __future__ import annotations

# ruff: noqa: C901, EM102, FBT001, PERF401, PLR0912, PLR0915, TRY003
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import pydantic_settings as ps
import pyvista as pv

from liblaf import cherries

mpl.use("Agg")
import matplotlib.pyplot as plt

FACTORS = ("geometry", "muscle_extent", "activation_sharing", "poisson")
METRICS = (
    "final_inverted_rest_measure_fraction",
    "peak_inverted_rest_measure_fraction",
    "final_detf_negative_detainv_negative_detg_positive_rest_measure_fraction",
    "peak_detf_negative_detainv_negative_detg_positive_rest_measure_fraction",
    "final_detf_negative_detainv_positive_detg_negative_rest_measure_fraction",
    "peak_detf_negative_detainv_positive_detg_negative_rest_measure_fraction",
    "trajectory_min_det_f",
    "final_target_rms",
    "final_activation_rms",
    "final_activation_jump_rms",
)
MIDLINE_METRICS = (
    "final_midline_arc_length_ratio",
    "final_midline_turning_density",
    "final_midline_x_reversal_fraction",
    "final_midline_y_range",
)
TRAJECTORY_FIELDS = (
    "objective",
    "target_rms",
    "gradient_rms",
    "gradient_inf",
    "activation_update_rms",
    "activation_rms",
    "activation_neighbor_jump_rms",
    "min_det_g",
    "min_det_ainv",
    "detf_negative_detainv_negative_detg_positive_rest_measure_fraction",
    "detf_negative_detainv_positive_detg_negative_rest_measure_fraction",
)
DET_SIGN_TOLERANCE = 1.0e-12


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    input_2d_roots: str = "data/60-pork-folding-2d"
    input_3d_roots: str = "data/70-pork-folding-3d"
    dimensions: str = "2d"
    output_dir: Path = cherries.output("80-folding-analysis", mkdir=True)
    tolerance: float = 1.0e-8
    require_stationarity: bool = False


def fail(message: str) -> None:
    raise ValueError(message)


def require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        fail(f"{context} missing required key {key!r}; present={sorted(mapping)}")
    return mapping[key]


def mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        fail(f"{context} must be an object, got {type(value).__name__}")
    return value


def finite(value: Any, context: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} must be numeric, got {value!r}") from error
    if not math.isfinite(result):
        fail(f"{context} must be finite, got {result!r}")
    return result


def roots(value: str) -> list[Path]:
    result = [Path(item).resolve() for item in value.split(",") if item.strip()]
    if not result:
        fail("at least one input root is required")
    for root in result:
        if not root.is_dir():
            raise NotADirectoryError(root)
    return result


def digest(path: Path) -> dict[str, Any]:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            h.update(block)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": h.hexdigest(),
    }


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return mapping(value, str(path))


def case_dimension(summary: dict[str, Any], path: Path) -> str:
    geometry = mapping(require(summary, "geometry", str(path)), f"{path}.geometry")
    domain = require(geometry, "domain", f"{path}.geometry")
    if not isinstance(domain, list) or len(domain) not in (2, 3):
        fail(f"{path}.geometry.domain must have length 2 or 3")
    return f"{len(domain)}d"


def factor_values(summary: dict[str, Any], path: Path) -> dict[str, str]:
    case = mapping(require(summary, "case", str(path)), f"{path}.case")
    geometry = mapping(require(summary, "geometry", str(path)), f"{path}.geometry")
    materials = mapping(require(summary, "materials", str(path)), f"{path}.materials")
    activation = mapping(
        require(summary, "activation", str(path)), f"{path}.activation"
    )
    muscle = mapping(
        require(materials, "muscle", f"{path}.materials"), f"{path}.materials.muscle"
    )
    nu_muscle = finite(
        require(muscle, "nu", f"{path}.materials.muscle"), f"{path}.materials.muscle.nu"
    )
    fat_value = require(materials, "fat", f"{path}.materials")
    if fat_value is not None:
        fat = mapping(fat_value, f"{path}.materials.fat")
        nu_fat = finite(
            require(fat, "nu", f"{path}.materials.fat"),
            f"{path}.materials.fat.nu",
        )
        if nu_muscle != nu_fat:
            fail(f"{path} has unequal fat/muscle Poisson ratios")
    return {
        "geometry": str(require(geometry, "geometry_id", f"{path}.geometry")),
        "muscle_extent": str(require(geometry, "muscle_extent_id", f"{path}.geometry")),
        "activation_sharing": str(
            require(activation, "sharing_id", f"{path}.activation")
        ),
        "poisson": f"{nu_muscle:.17g}",
        "case_name": str(require(case, "name", f"{path}.case")),
    }


def determinants(
    grid: pv.UnstructuredGrid, path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = []
    for name in ("DetF", "DetG", "DetAinv"):
        if name not in grid.cell_data:
            fail(f"{path} has no cell {name} array")
        det = np.asarray(grid.cell_data[name], dtype=float).reshape(-1)
        if det.size != grid.n_cells or not np.isfinite(det).all():
            fail(f"{path} has invalid/non-finite {name}")
        values.append(det)
    det_f, det_g, det_ainv = values
    return det_f, det_g, det_ainv


def rest_measure(grid: pv.UnstructuredGrid, dimension: str, path: Path) -> np.ndarray:
    sized = grid.compute_cell_sizes(
        length=False, area=dimension == "2d", volume=dimension == "3d"
    )
    key = "Area" if dimension == "2d" else "Volume"
    measure = np.asarray(sized.cell_data[key], dtype=float).reshape(-1)
    if not np.isfinite(measure).all() or np.any(measure <= 0):
        fail(f"{path} has non-positive rest {key.lower()}")
    return measure


def fold_metrics(path: Path, measure: np.ndarray) -> dict[str, float]:
    det_f, det_g, det_ainv = determinants(pv.read(path), path)
    if det_f.size != measure.size:
        fail(f"{path} has {det_f.size} cells, expected {measure.size}")
    if not np.allclose(det_g, det_f * det_ainv, rtol=1.0e-8, atol=DET_SIGN_TOLERANCE):
        fail(f"{path} determinant arrays violate DetG = DetF * DetAinv")
    inverted = det_f < 0.0
    negative = det_f < -DET_SIGN_TOLERANCE
    ainv_negative = det_ainv < -DET_SIGN_TOLERANCE
    ainv_positive = det_ainv > DET_SIGN_TOLERANCE
    g_negative = det_g < -DET_SIGN_TOLERANCE
    g_positive = det_g > DET_SIGN_TOLERANCE
    paired = negative & ainv_negative & g_positive
    opposite = negative & ainv_positive & g_negative
    undecided = negative & ~(paired | opposite)
    if np.any(undecided) and not np.all(
        (np.abs(det_ainv[undecided]) <= DET_SIGN_TOLERANCE)
        | (np.abs(det_g[undecided]) <= DET_SIGN_TOLERANCE)
    ):
        fail(f"{path} has a nonzero DetF<0 determinant-sign partition mismatch")
    return {
        "min_det_f": float(det_f.min()),
        "min_det_g": float(det_g.min()),
        "min_det_ainv": float(det_ainv.min()),
        "inverted_cell_fraction": float(inverted.mean()),
        "inverted_rest_measure_fraction": float(
            measure[inverted].sum() / measure.sum()
        ),
        "negative_det_f_mean": float(
            np.dot(measure, np.maximum(-det_f, 0.0)) / measure.sum()
        ),
        "detf_negative_detainv_negative_detg_positive_rest_measure_fraction": float(
            measure[paired].sum() / measure.sum()
        ),
        "detf_negative_detainv_positive_detg_negative_rest_measure_fraction": float(
            measure[opposite].sum() / measure.sum()
        ),
    }


def material_midline_metrics(path: Path) -> dict[str, float]:
    """Measure visible 2-D wrinkling on the material line at mid-thickness."""
    grid = pv.read(path)
    if "Displacement" not in grid.point_data:
        fail(f"{path} lacks point-vector Displacement")
    reference = np.asarray(grid.points, dtype=float)[:, :2]
    deformed = (
        reference + np.asarray(grid.point_data["Displacement"], dtype=float)[:, :2]
    )
    if not np.isfinite(reference).all() or not np.isfinite(deformed).all():
        fail(f"{path} has non-finite reference/deformed points")
    middle_y = 0.5 * (float(reference[:, 1].min()) + float(reference[:, 1].max()))
    selected = np.isclose(reference[:, 1], middle_y, rtol=0.0, atol=1.0e-12)
    if np.count_nonzero(selected) < 3:
        fail(f"{path} has fewer than three mid-thickness material-line points")
    order = np.argsort(reference[selected, 0])
    reference_line = reference[selected][order]
    deformed_line = deformed[selected][order]
    reference_dx = np.diff(reference_line[:, 0])
    if np.any(reference_dx <= 0.0):
        fail(f"{path} mid-thickness reference x coordinates are not unique")
    reference_length = float(reference_line[-1, 0] - reference_line[0, 0])
    segments = np.diff(deformed_line, axis=0)
    segment_lengths = np.linalg.norm(segments, axis=1)
    if reference_length <= 0.0 or np.any(segment_lengths <= 0.0):
        fail(f"{path} has a degenerate mid-thickness material line")
    angles = np.unwrap(np.arctan2(segments[:, 1], segments[:, 0]))
    total_absolute_turning = float(np.abs(np.diff(angles)).sum())
    return {
        "final_midline_arc_length_ratio": float(
            segment_lengths.sum() / reference_length
        ),
        "final_midline_turning_density": total_absolute_turning / reference_length,
        "final_midline_x_reversal_fraction": float(
            np.mean(segments[:, 0] < -1.0e-12 * reference_length)
        ),
        "final_midline_y_range": float(np.ptp(deformed_line[:, 1])),
    }


def step(value: Any, context: str) -> int:
    parsed = finite(value, context)
    if not parsed.is_integer() or parsed < 0:
        fail(f"{context} must be a nonnegative integer, got {value!r}")
    return int(parsed)


def series_paths(case_dir: Path) -> dict[int, Path]:
    path = case_dir / "history.vtu.series"
    if not path.is_file():
        raise FileNotFoundError(path)
    data = read_json(path)
    files = require(data, "files", str(path))
    if not isinstance(files, list) or not files:
        fail(f"{path}.files must be nonempty")
    result: dict[int, Path] = {}
    for item in files:
        entry = mapping(item, f"{path}.files")
        name = require(entry, "name", f"{path}.files")
        frame_step = step(require(entry, "time", f"{path}.files"), f"{path}.files.time")
        frame = path.parent / str(name)
        if not frame.is_file():
            raise FileNotFoundError(frame)
        if frame_step in result:
            fail(f"{path} has duplicate time/step {frame_step}")
        result[frame_step] = frame
    if set(result) != set(range(len(result))):
        fail(f"{path} times must be exactly 0..{len(result) - 1}, got {sorted(result)}")
    return result


def trace_rows(case_dir: Path) -> dict[int, dict[str, str]]:
    path = case_dir / "trace.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        fail(f"{path} has no rows")
    required = {
        "step",
        "inverted_cell_fraction",
        "inverted_rest_measure_fraction",
        "negative_det_f_mean",
    }
    if not required <= set(rows[0]):
        fail(f"{path} lacks folding fields {sorted(required - set(rows[0]))}")
    result: dict[int, dict[str, str]] = {}
    for row in rows:
        row_step = step(row["step"], f"{path}.step")
        if row_step in result:
            fail(f"{path} has duplicate step {row_step}")
        result[row_step] = row
    if set(result) != set(range(len(result))):
        fail(f"{path} steps must be exactly 0..{len(result) - 1}, got {sorted(result)}")
    return result


def trace_finite(row: dict[str, str], names: tuple[str, ...], context: str) -> float:
    """Read one required finite trace metric, accepting documented aliases."""
    for name in names:
        if name in row:
            return finite(row[name], f"{context}.{name}")
    fail(f"{context} missing required trace metric; expected one of {names}")
    message = "unreachable"
    raise AssertionError(message)


def close(a: float, b: float, tol: float, context: str) -> None:
    if not math.isclose(a, b, rel_tol=tol, abs_tol=tol):
        fail(f"{context}: recomputed={a:.17g}, recorded={b:.17g}")


def analyze_case(
    summary_path: Path, tol: float, require_stationarity: bool
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary = read_json(summary_path)
    for key in (
        "case",
        "geometry",
        "materials",
        "activation",
        "counts",
        "inverse",
        "metrics",
    ):
        mapping(require(summary, key, str(summary_path)), f"{summary_path}.{key}")
    dim = case_dimension(summary, summary_path)
    factors = factor_values(summary, summary_path)
    case_dir = summary_path.parent
    final_path = case_dir / "final.vtu"
    if not final_path.is_file():
        raise FileNotFoundError(final_path)
    measure = rest_measure(pv.read(final_path), dim, final_path)
    final = fold_metrics(final_path, measure)
    midline = material_midline_metrics(final_path) if dim == "2d" else {}
    frames, trace = series_paths(case_dir), trace_rows(case_dir)
    if set(frames) != set(trace):
        fail(
            f"{case_dir}: series times {sorted(frames)} != trace steps {sorted(trace)}"
        )
    trajectory = []
    for frame_step in sorted(frames):
        frame, recorded = frames[frame_step], trace[frame_step]
        values = fold_metrics(frame, measure)
        for key in (
            "inverted_cell_fraction",
            "inverted_rest_measure_fraction",
            "negative_det_f_mean",
        ):
            close(
                values[key],
                finite(recorded[key], f"{case_dir}/trace.csv:{key}"),
                tol,
                f"{case_dir} step {frame_step} {key}",
            )
        context = f"{case_dir}/trace.csv step {frame_step}"
        trace_min_det_g = trace_finite(recorded, ("detG/min",), context)
        trace_min_det_ainv = trace_finite(recorded, ("detAinv/min",), context)
        close(values["min_det_g"], trace_min_det_g, tol, f"{context} detG/min")
        close(
            values["min_det_ainv"],
            trace_min_det_ainv,
            tol,
            f"{context} detAinv/min",
        )
        normalized = {
            "objective": trace_finite(recorded, ("objective", "loss"), context),
            "target_rms": trace_finite(recorded, ("target/rms",), context),
            "gradient_rms": trace_finite(recorded, ("gradient_rms",), context),
            "gradient_inf": trace_finite(recorded, ("gradient_inf",), context),
            "activation_update_rms": trace_finite(
                recorded, ("activation_update_rms",), context
            ),
            "activation_rms": trace_finite(recorded, ("activation/rms",), context),
            "activation_neighbor_jump_rms": trace_finite(
                recorded, ("activation/neighbor_jump_rms",), context
            ),
            "min_det_g": trace_min_det_g,
            "min_det_ainv": trace_min_det_ainv,
        }
        trajectory.append(
            {"dimension": dim, **factors, "step": frame_step, **values, **normalized}
        )
    for key in final:
        close(final[key], trajectory[-1][key], tol, f"{case_dir} final {key}")
    metrics = mapping(summary["metrics"], f"{summary_path}.metrics")
    final_summary = mapping(
        require(metrics, "final", f"{summary_path}.metrics"),
        f"{summary_path}.metrics.final",
    )
    for metric_key, final_key in (
        ("detF/min", "min_det_f"),
        ("detG/min", "min_det_g"),
        ("detAinv/min", "min_det_ainv"),
    ):
        recorded_min = finite(
            require(final_summary, metric_key, f"{summary_path}.metrics.final"),
            f"{summary_path}.metrics.final.{metric_key}",
        )
        close(final[final_key], recorded_min, tol, f"{case_dir} final {metric_key}")
    final_target_rms = finite(
        require(final_summary, "target/rms", f"{summary_path}.metrics.final"),
        f"{summary_path}.metrics.final.target/rms",
    )
    final_activation_rms = finite(
        require(final_summary, "activation/rms", f"{summary_path}.metrics.final"),
        f"{summary_path}.metrics.final.activation/rms",
    )
    final_activation_jump_rms = finite(
        require(
            final_summary,
            "activation/neighbor_jump_rms",
            f"{summary_path}.metrics.final",
        ),
        f"{summary_path}.metrics.final.activation/neighbor_jump_rms",
    )
    for summary_value, trace_key, label in (
        (final_target_rms, "target_rms", "target/rms"),
        (final_activation_rms, "activation_rms", "activation/rms"),
        (
            final_activation_jump_rms,
            "activation_neighbor_jump_rms",
            "activation/neighbor_jump_rms",
        ),
    ):
        close(
            summary_value,
            trajectory[-1][trace_key],
            tol,
            f"{case_dir} final {label}",
        )
    inverse = mapping(summary["inverse"], f"{summary_path}.inverse")
    frame_count = len(frames)
    inverse_evaluations = step(
        require(inverse, "evaluations", f"{summary_path}.inverse"),
        f"{summary_path}.inverse.evaluations",
    )
    if inverse_evaluations != frame_count:
        fail(
            f"{summary_path}.inverse.evaluations={inverse_evaluations} "
            f"does not equal history frame count {frame_count}"
        )
    paraview = mapping(
        require(summary, "paraview", str(summary_path)), f"{summary_path}.paraview"
    )
    paraview_frames = step(
        require(paraview, "frames", f"{summary_path}.paraview"),
        f"{summary_path}.paraview.frames",
    )
    if paraview_frames != frame_count:
        fail(
            f"{summary_path}.paraview.frames={paraview_frames} "
            f"does not equal history frame count {frame_count}"
        )
    adam = mapping(
        require(inverse, "adam", f"{summary_path}.inverse"),
        f"{summary_path}.inverse.adam",
    )
    refinement = mapping(
        require(inverse, "refinement", f"{summary_path}.inverse"),
        f"{summary_path}.inverse.refinement",
    )
    adam_updates = step(
        require(adam, "updates", f"{summary_path}.inverse.adam"),
        f"{summary_path}.inverse.adam.updates",
    )
    refinement_iterations = step(
        require(
            refinement,
            "accepted_iterations",
            f"{summary_path}.inverse.refinement",
        ),
        f"{summary_path}.inverse.refinement.accepted_iterations",
    )
    inverse_updates = step(
        require(inverse, "updates", f"{summary_path}.inverse"),
        f"{summary_path}.inverse.updates",
    )
    if inverse_updates != adam_updates + refinement_iterations:
        fail(
            f"{summary_path}.inverse.updates={inverse_updates} != Adam "
            f"{adam_updates} + refinement {refinement_iterations}"
        )
    convergence = mapping(
        require(inverse, "convergence", f"{summary_path}.inverse"),
        f"{summary_path}.inverse.convergence",
    )
    stationarity_gate = require(
        convergence,
        "practical_stationarity_gate",
        f"{summary_path}.inverse.convergence",
    )
    if not isinstance(stationarity_gate, bool):
        fail(f"{summary_path}.inverse.convergence gate must be boolean")
    if require_stationarity and not stationarity_gate:
        fail(f"{summary_path} did not pass practical stationarity")
    final_gradient_inf = finite(
        require(
            convergence, "final_gradient_inf", f"{summary_path}.inverse.convergence"
        ),
        f"{summary_path}.inverse.convergence.final_gradient_inf",
    )
    final_gradient_rms = finite(
        require(
            convergence, "final_gradient_rms", f"{summary_path}.inverse.convergence"
        ),
        f"{summary_path}.inverse.convergence.final_gradient_rms",
    )
    tail = mapping(
        require(inverse, "tail", f"{summary_path}.inverse"),
        f"{summary_path}.inverse.tail",
    )
    tail_gate = require(
        tail, "inverse_converged_1pct_tail_gate", f"{summary_path}.inverse.tail"
    )
    if not isinstance(tail_gate, bool):
        fail(f"{summary_path}.inverse.tail gate must be boolean")
    failures = mapping(
        require(inverse, "failures", f"{summary_path}.inverse"),
        f"{summary_path}.inverse.failures",
    )
    failure_counts = {
        f"{name}_failure_count": step(
            require(failures, name, f"{summary_path}.inverse.failures"),
            f"{summary_path}.inverse.failures.{name}",
        )
        for name in ("forward", "inverse", "adjoint")
    }
    refinement_trial_forward_failure_count = step(
        require(
            failures,
            "refinement_trial_forward",
            f"{summary_path}.inverse.failures",
        ),
        f"{summary_path}.inverse.failures.refinement_trial_forward",
    )
    first = next((row["step"] for row in trajectory if row["min_det_f"] < 0.0), None)
    last = next(
        (row["step"] for row in reversed(trajectory) if row["min_det_f"] < 0.0),
        None,
    )
    row = {
        "dimension": dim,
        **factors,
        "source_summary": str(summary_path.resolve()),
        "final_vtu_sha256": digest(final_path)["sha256"],
        "final_target_rms": final_target_rms,
        "final_activation_rms": final_activation_rms,
        "final_activation_jump_rms": final_activation_jump_rms,
        "first_inversion_step": first,
        "last_inversion_step": last,
        "recovered_by_final": bool(
            first is not None and final["inverted_cell_fraction"] == 0.0
        ),
        "tail_gate_1pct": tail_gate,
        "practical_stationarity_gate": stationarity_gate,
        "final_gradient_inf": final_gradient_inf,
        "final_gradient_rms": final_gradient_rms,
        "refinement_trial_forward_failure_count": refinement_trial_forward_failure_count,
        **failure_counts,
        "trajectory_min_det_f": min(item["min_det_f"] for item in trajectory),
        "inverted_frame_fraction": float(
            np.mean([item["inverted_cell_fraction"] > 0 for item in trajectory])
        ),
        "peak_inverted_cell_fraction": max(
            item["inverted_cell_fraction"] for item in trajectory
        ),
        "peak_inverted_rest_measure_fraction": max(
            item["inverted_rest_measure_fraction"] for item in trajectory
        ),
        "final_detf_negative_detainv_negative_detg_positive_rest_measure_fraction": final[
            "detf_negative_detainv_negative_detg_positive_rest_measure_fraction"
        ],
        "peak_detf_negative_detainv_negative_detg_positive_rest_measure_fraction": max(
            item["detf_negative_detainv_negative_detg_positive_rest_measure_fraction"]
            for item in trajectory
        ),
        "final_detf_negative_detainv_positive_detg_negative_rest_measure_fraction": final[
            "detf_negative_detainv_positive_detg_negative_rest_measure_fraction"
        ],
        "peak_detf_negative_detainv_positive_detg_negative_rest_measure_fraction": max(
            item["detf_negative_detainv_positive_detg_negative_rest_measure_fraction"]
            for item in trajectory
        ),
        "peak_negative_det_f_mean": max(
            item["negative_det_f_mean"] for item in trajectory
        ),
        "final_min_det_f": final["min_det_f"],
        "final_inverted_cell_fraction": final["inverted_cell_fraction"],
        "final_inverted_rest_measure_fraction": final["inverted_rest_measure_fraction"],
        "final_negative_det_f_mean": final["negative_det_f_mean"],
        **midline,
    }
    return row, trajectory


def discover(case_roots: list[Path]) -> list[Path]:
    paths = sorted(
        {
            path
            for root in case_roots
            for path in root.rglob("summary.json")
            if (path.parent / "final.vtu").is_file()
        }
    )
    if not paths:
        fail(f"no case summary/final.vtu below {case_roots}")
    return paths


def validate_matrix(rows: list[dict[str, Any]], dimensions: set[str]) -> None:
    for dim in dimensions:
        selected = [row for row in rows if row["dimension"] == dim]
        if len(selected) != 16:
            fail(f"{dim} requires exactly 16 cases, found {len(selected)}")
        values = {
            factor: sorted({row[factor] for row in selected}) for factor in FACTORS
        }
        if any(len(levels) != 2 for levels in values.values()):
            fail(f"{dim} requires exactly two levels/factor, found {values}")
        combos = {tuple(row[factor] for factor in FACTORS) for row in selected}
        expected = set(itertools.product(*(values[factor] for factor in FACTORS)))
        if combos != expected:
            fail(f"{dim} factor matrix is not complete/unique")


def effects(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for dim in sorted({row["dimension"] for row in rows}):
        selected = [row for row in rows if row["dimension"] == dim]
        levels = {
            factor: sorted({row[factor] for row in selected}) for factor in FACTORS
        }
        metrics = METRICS + (
            MIDLINE_METRICS
            if all(key in row for row in selected for key in MIDLINE_METRICS)
            else ()
        )
        for factor in FACTORS:
            other = [item for item in FACTORS if item != factor]
            for combo in itertools.product(*(levels[item] for item in other)):
                left = next(
                    row
                    for row in selected
                    if all(
                        row[item] == value
                        for item, value in zip(other, combo, strict=True)
                    )
                    and row[factor] == levels[factor][0]
                )
                right = next(
                    row
                    for row in selected
                    if all(
                        row[item] == value
                        for item, value in zip(other, combo, strict=True)
                    )
                    and row[factor] == levels[factor][1]
                )
                for metric in metrics:
                    output.append(
                        {
                            "dimension": dim,
                            "kind": "paired_marginal",
                            "factor_a": factor,
                            "level_a": levels[factor][0],
                            "level_b": levels[factor][1],
                            "conditioning": json.dumps(
                                dict(zip(other, combo, strict=True)), sort_keys=True
                            ),
                            "metric": metric,
                            "effect_b_minus_a": right[metric] - left[metric],
                        }
                    )
        for a, b in itertools.combinations(FACTORS, 2):
            other = [item for item in FACTORS if item not in (a, b)]
            for combo in itertools.product(*(levels[item] for item in other)):
                lookup = {
                    tuple(row[item] for item in (a, b)): row
                    for row in selected
                    if all(
                        row[item] == value
                        for item, value in zip(other, combo, strict=True)
                    )
                }
                for metric in metrics:
                    interaction = (
                        lookup[levels[a][1], levels[b][1]][metric]
                        - lookup[levels[a][1], levels[b][0]][metric]
                        - lookup[levels[a][0], levels[b][1]][metric]
                        + lookup[levels[a][0], levels[b][0]][metric]
                    )
                    output.append(
                        {
                            "dimension": dim,
                            "kind": "interaction_difference_in_differences",
                            "factor_a": a,
                            "factor_b": b,
                            "conditioning": json.dumps(
                                dict(zip(other, combo, strict=True)), sort_keys=True
                            ),
                            "metric": metric,
                            "effect_b_minus_a": interaction,
                        }
                    )
    return output


def factorial_coefficients(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return signed nested factorial contrasts using low=-1 and high=+1 codes."""
    output = []
    formula = "2^|S| * mean(y * product(coded +/-1 factors in S))"
    for dim in sorted({row["dimension"] for row in rows}):
        selected = [row for row in rows if row["dimension"] == dim]
        levels = {
            factor: sorted({row[factor] for row in selected}) for factor in FACTORS
        }
        metrics = METRICS + (
            MIDLINE_METRICS
            if all(key in row for row in selected for key in MIDLINE_METRICS)
            else ()
        )
        for size in range(1, len(FACTORS) + 1):
            for subset in itertools.combinations(FACTORS, size):
                codes = np.asarray(
                    [
                        math.prod(
                            -1.0 if row[factor] == levels[factor][0] else 1.0
                            for factor in subset
                        )
                        for row in selected
                    ]
                )
                coding = {
                    factor: {"low": levels[factor][0], "high": levels[factor][1]}
                    for factor in subset
                }
                for metric in metrics:
                    values = np.asarray(
                        [finite(row[metric], metric) for row in selected]
                    )
                    output.append(
                        {
                            "dimension": dim,
                            "factor_subset": ":".join(subset),
                            "order": size,
                            "metric": metric,
                            "coding": json.dumps(coding, sort_keys=True),
                            "formula": formula,
                            "signed_nested_contrast": float(
                                2**size * np.mean(values * codes)
                            ),
                        }
                    )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, Any]], out: Path, y: str, title: str) -> None:
    dimensions = sorted({row["dimension"] for row in rows})
    fig, axes = plt.subplots(1, len(dimensions), figsize=(5.5 * len(dimensions), 4))
    for axis, dim in zip(np.atleast_1d(axes), dimensions, strict=True):
        values = [row for row in rows if row["dimension"] == dim]
        axis.set(title=dim.upper(), ylabel=y.replace("_", " "))
        if not values:
            continue
        x = np.arange(len(values))
        axis.scatter(
            x,
            [row[y] for row in values],
            c=[
                "tab:red" if row["peak_inverted_cell_fraction"] > 0 else "tab:green"
                for row in values
            ],
        )
        axis.set_xticks(
            x, [row["case_name"] for row in values], rotation=55, ha="right"
        )
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_fit_roughness(rows: list[dict[str, Any]], out: Path) -> None:
    """Show final fit/roughness tradeoff, colored by transient-or-final folding."""
    peaks = [row["peak_inverted_rest_measure_fraction"] for row in rows]
    norm = mpl.colors.Normalize(vmin=0.0, vmax=max(peaks, default=0.0) or 1.0)
    dimensions = sorted({row["dimension"] for row in rows})
    fig, axes = plt.subplots(
        1, len(dimensions), figsize=(5 * len(dimensions), 4), layout="constrained"
    )
    axes = np.atleast_1d(axes)
    for axis, dim in zip(axes, dimensions, strict=True):
        selected = [row for row in rows if row["dimension"] == dim]
        axis.set(
            title=dim.upper(),
            xlabel="final target RMS",
            ylabel="final activation neighbor-jump RMS",
        )
        if selected:
            axis.scatter(
                [row["final_target_rms"] for row in selected],
                [row["final_activation_jump_rms"] for row in selected],
                c=[row["peak_inverted_rest_measure_fraction"] for row in selected],
                cmap="viridis",
                norm=norm,
            )
        axis.grid(alpha=0.25)
    fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap="viridis"),
        ax=axes.tolist(),
        label="peak inverted rest-measure fraction",
    )
    fig.savefig(out, dpi=180)
    plt.close(fig)


def main(cfg: Config) -> None:
    if cfg.output_dir.exists() and any(cfg.output_dir.iterdir()):
        raise FileExistsError(cfg.output_dir)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    requested = {item.strip() for item in cfg.dimensions.split(",") if item.strip()}
    if not requested <= {"2d", "3d"} or not requested:
        fail("--dimensions must be a nonempty subset of 2d,3d")
    root_values = {"2d": cfg.input_2d_roots, "3d": cfg.input_3d_roots}
    rows: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []
    for dim in sorted(requested):
        for path in discover(roots(root_values[dim])):
            row, frames = analyze_case(path, cfg.tolerance, cfg.require_stationarity)
            if row["dimension"] != dim:
                fail(f"{path} is {row['dimension']}, expected {dim}")
            rows.append(row)
            trajectory.extend(frames)
    validate_matrix(rows, requested)
    if len({(row["dimension"], row["case_name"]) for row in rows}) != len(rows):
        fail("duplicate dimension/case names")
    effect_rows = effects(rows)
    coefficient_rows = factorial_coefficients(rows)
    write_csv(cfg.output_dir / "cases.csv", rows)
    write_csv(cfg.output_dir / "trajectory.csv", trajectory)
    write_csv(cfg.output_dir / "factor-effects.csv", effect_rows)
    write_csv(cfg.output_dir / "factorial-coefficients.csv", coefficient_rows)
    for key, title in (
        ("peak_inverted_rest_measure_fraction", "Peak rest-measure inversion fraction"),
        ("final_target_rms", "Final target RMS"),
        ("final_activation_rms", "Final activation RMS"),
        ("final_activation_jump_rms", "Final activation jump RMS"),
    ):
        plot(rows, cfg.output_dir / f"{key}.png", key, title)
    if all(key in row for row in rows for key in MIDLINE_METRICS):
        for key, title in (
            ("final_midline_arc_length_ratio", "Final midline arc-length ratio"),
            ("final_midline_turning_density", "Final midline turning density"),
            (
                "final_midline_x_reversal_fraction",
                "Final midline horizontal-reversal fraction",
            ),
            ("final_midline_y_range", "Final midline vertical range"),
        ):
            plot(rows, cfg.output_dir / f"{key}.png", key, title)
    plot_fit_roughness(rows, cfg.output_dir / "fit-vs-activation-jump.png")
    receipt = {
        "case_count": len(rows),
        "stationarity_pass_case_count": sum(
            row["practical_stationarity_gate"] for row in rows
        ),
        "stationarity_fail_case_count": sum(
            not row["practical_stationarity_gate"] for row in rows
        ),
        "require_stationarity": cfg.require_stationarity,
        "frame_count": len(trajectory),
        "factorial_coefficient_count": len(coefficient_rows),
        "midline_definition": (
            "deformed 2-D material line whose reference y is the domain "
            "mid-thickness; turning density is total absolute tangent turning "
            "divided by reference length"
        ),
        "trajectory_fields": list(TRAJECTORY_FIELDS),
        "recovered_case_count": sum(row["recovered_by_final"] for row in rows),
        "dimensions": sorted(requested),
        "cases": [
            {
                **digest(Path(row["source_summary"])),
                "case_name": row["case_name"],
                "dimension": row["dimension"],
                "tail_gate_1pct": row["tail_gate_1pct"],
                "practical_stationarity_gate": row["practical_stationarity_gate"],
                "final_gradient_inf": row["final_gradient_inf"],
                "final_gradient_rms": row["final_gradient_rms"],
                "refinement_trial_forward_failure_count": row[
                    "refinement_trial_forward_failure_count"
                ],
                "forward_failure_count": row["forward_failure_count"],
                "inverse_failure_count": row["inverse_failure_count"],
                "adjoint_failure_count": row["adjoint_failure_count"],
                "last_inversion_step": row["last_inversion_step"],
                "recovered_by_final": row["recovered_by_final"],
                "final_detf_negative_detainv_negative_detg_positive_rest_measure_fraction": row[
                    "final_detf_negative_detainv_negative_detg_positive_rest_measure_fraction"
                ],
                "peak_detf_negative_detainv_negative_detg_positive_rest_measure_fraction": row[
                    "peak_detf_negative_detainv_negative_detg_positive_rest_measure_fraction"
                ],
                "final_detf_negative_detainv_positive_detg_negative_rest_measure_fraction": row[
                    "final_detf_negative_detainv_positive_detg_negative_rest_measure_fraction"
                ],
                "peak_detf_negative_detainv_positive_detg_negative_rest_measure_fraction": row[
                    "peak_detf_negative_detainv_positive_detg_negative_rest_measure_fraction"
                ],
                **{key: row.get(key) for key in MIDLINE_METRICS},
            }
            for row in rows
        ],
    }
    (cfg.output_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Folding factorial analysis",
        "",
        f"Practical stationarity: {sum(row['practical_stationarity_gate'] for row in rows)}/{len(rows)} pass; {sum(not row['practical_stationarity_gate'] for row in rows)} fail. The failed cases are retained for explicit comparison (require_stationarity={cfg.require_stationarity}).",
        "",
        "All effects are descriptive paired differences; interaction rows are difference-in-differences, not causal estimates. Determinant-sign fractions are descriptive frame classifications using DetF<0, DetAinv, and DetG signs (with a zero tolerance). Factorial coefficients use low=-1/high=+1 and 2^|S| * mean(y * product(coded factors)).",
        "",
        "| Dimension | Case | Stationarity / tail | Final grad inf / RMS | Failures (forward/inverse/adjoint/trial) | First/last inversion | Recovered by final | Final/peak F- A- G+ rest | Final/peak F- A+ G- rest | Final inverted rest fraction | Peak inverted rest fraction | Trajectory min detF | Final target RMS | Final activation RMS | Midline arc ratio / turning density / x-reversal |",
        "| --- | --- | --- | ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    lines += [
        f"| {r['dimension']} | {r['case_name']} | {r['practical_stationarity_gate']} / {r['tail_gate_1pct']} | {r['final_gradient_inf']:.4g} / {r['final_gradient_rms']:.4g} | {r['forward_failure_count']}/{r['inverse_failure_count']}/{r['adjoint_failure_count']}/{r['refinement_trial_forward_failure_count']} | {r['first_inversion_step']}/{r['last_inversion_step']} | {r['recovered_by_final']} | {r['final_detf_negative_detainv_negative_detg_positive_rest_measure_fraction']:.4g}/{r['peak_detf_negative_detainv_negative_detg_positive_rest_measure_fraction']:.4g} | {r['final_detf_negative_detainv_positive_detg_negative_rest_measure_fraction']:.4g}/{r['peak_detf_negative_detainv_positive_detg_negative_rest_measure_fraction']:.4g} | {r['final_inverted_rest_measure_fraction']:.4g} | {r['peak_inverted_rest_measure_fraction']:.4g} | {r['trajectory_min_det_f']:.4g} | {r['final_target_rms']:.4g} | {r['final_activation_rms']:.4g} | {r.get('final_midline_arc_length_ratio', math.nan):.4g} / {r.get('final_midline_turning_density', math.nan):.4g} / {r.get('final_midline_x_reversal_fraction', math.nan):.4g} |"
        for r in sorted(rows, key=lambda x: (x["dimension"], x["case_name"]))
    ]
    (cfg.output_dir / "results.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    cherries.log_metrics(
        {"folding/cases": len(rows), "folding/frames": len(trajectory)}
    )


if __name__ == "__main__":
    cherries.main(main)
