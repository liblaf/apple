# ruff: noqa: EM101, EM102, TRY003
"""Compare saved original- and conservative-rate activation-study states on CPU."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import pydantic_settings as ps
import pyvista as pv
from experiment_profile import ProfileCometNoCommit

mpl.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from liblaf import cherries

ROOT = Path(__file__).resolve().parents[6]
GROUP = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "exp/2026/09/07/face-actuation-diagnosis/data/12-historical-fixture"
CAMERAS = ROOT / "exp/2026/09/08/physical-volume-closeups/data/20-regions/summary.json"
BACKGROUND = "#242c36"
FIT_TOLERANCE_MM = 0.05
MOTION_TOLERANCE_MM = 0.05
PRIMARY_RESIDUAL = "primary_union_normal_residual_highpass_5mm_rms_mm"
PRIMARY_LOW_FREQUENCY = "primary_union_low_frequency_normal_target_projection"
SETTINGS = GROUP / "data/47-conservative-rate-settings/settings.json"

RUNS = ("original-off", "original-on", "conservative-off", "conservative-on")
STYLES = {
    "original-off": {
        "label": "Original lr 7.858, S(C) off",
        "color": "#D55E00",
        "linestyle": "-",
        "marker": "o",
    },
    "original-on": {
        "label": "Original lr 7.858, S(C) on",
        "color": "#0072B2",
        "linestyle": "--",
        "marker": "s",
    },
    "conservative-off": {
        "label": "Conservative lr 1.964, S(C) off",
        "color": "#CC79A7",
        "linestyle": "-",
        "marker": "^",
    },
    "conservative-on": {
        "label": "Conservative lr 1.964, S(C) on",
        "color": "#009E73",
        "linestyle": "--",
        "marker": "D",
    },
}
TARGET_STYLE = {
    "label": "Target",
    "color": "#333333",
    "linestyle": ":",
    "marker": ".",
}


class Config(cherries.BaseConfig):
    """Completed saved-run directories and a fresh comparison directory."""

    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    off_dir: Path = GROUP / "data/48-axis-off-lr-quarter-64"
    on_dir: Path = GROUP / "data/49-axis-on-lr-quarter-64"
    output_dir: Path = GROUP / "data/52-conservative-rate-comparison"


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _digest(resolved),
    }


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
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


def _parse(value: str) -> Any:
    if value == "True":
        return True
    if value == "False":
        return False
    try:
        return float(value)
    except ValueError:
        return value


def _read_trace(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as stream:
        raw = list(csv.DictReader(stream))
    if not raw:
        raise ValueError(f"empty trace: {path}")
    rows = [
        {key: _parse(value) for key, value in source.items() if value not in (None, "")}
        for source in raw
    ]
    steps = [int(row["step"]) for row in rows]
    if steps != list(range(steps[-1] + 1)):
        raise ValueError(f"trace is not a contiguous solver-valid prefix: {path}")
    required = {
        "fit_rms_mm",
        "motion_rms_mm",
        "forward_steps",
        "forward_success",
        "adjoint_success",
        "detF_min",
        "inverted_all_cells",
        "smoothness_C",
        "geometry_update_face_vector_rms_mm",
        "z_update_frobenius_rms",
        PRIMARY_RESIDUAL,
        PRIMARY_LOW_FREQUENCY,
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"trace lacks required fields {sorted(missing)}: {path}")
    invalid = [
        int(row["step"])
        for row in rows
        if row["forward_success"] is not True or row["adjoint_success"] is not True
    ]
    if invalid:
        raise ValueError(
            f"trace contains solver-invalid rows at steps {invalid}: {path}"
        )
    return rows


def _compact(run: str, selection: str, row: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "step",
        "fit_rms_mm",
        "motion_rms_mm",
        "forward_steps",
        "detF_min",
        "detF_max",
        "inverted_all_cells",
        "inverted_active_cells",
        PRIMARY_RESIDUAL,
        "primary_union_normal_displacement_highpass_5mm_rms_mm",
        PRIMARY_LOW_FREQUENCY,
        "smoothness_C",
        "smoothness_Z",
        "geometry_update_face_vector_rms_mm",
        "z_update_frobenius_rms",
        "shortening_fraction_p50",
        "shortening_fraction_p90",
        "shortening_fraction_p99",
        "shortening_fraction_p100",
    )
    return {
        "run": run,
        "selection": selection,
        **{field: row.get(field) for field in fields},
    }


def _run_outcome(
    run: str,
    directory: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    expected_learning_rate: float,
) -> dict[str, Any]:
    last_step = int(rows[-1]["step"])
    if int(summary["last_evaluated_step"]) != last_step:
        raise ValueError(f"summary/trace endpoint mismatch: {run}")
    best = min(rows, key=lambda row: (float(row["fit_rms_mm"]), int(row["step"])))
    inverted = [row for row in rows if int(row["inverted_all_cells"]) > 0]
    first_inversion = inverted[0] if inverted else None
    minimum_detf = min(rows, key=lambda row: (float(row["detF_min"]), int(row["step"])))
    learning_rates = {
        float(row["learning_rate"])
        for row in rows
        if row.get("learning_rate") is not None
    }
    if learning_rates and learning_rates != {expected_learning_rate}:
        raise ValueError(f"trace learning rate differs from frozen settings: {run}")
    updates = [row for row in rows if int(row["step"]) > 0]
    if not updates:
        raise ValueError(f"run has no completed optimizer updates: {run}")
    path_summary: dict[str, dict[str, float | int]] = {}
    for field in (
        "geometry_update_face_vector_rms_mm",
        "z_update_frobenius_rms",
    ):
        values = np.asarray([float(row[field]) for row in updates], dtype=np.float64)
        if not np.isfinite(values).all() or np.any(values < 0.0):
            raise ValueError(f"invalid optimizer update geometry metric {field}: {run}")
        path_summary[field] = {
            "count": len(values),
            "maximum": float(np.max(values)),
            "median": float(np.median(values)),
            "cumulative": float(np.sum(values)),
        }
    return {
        "run": run,
        "directory": str(directory.resolve()),
        "status": str(summary["status"]),
        "learning_rate": expected_learning_rate,
        "smoothness_weight": float(rows[-1].get("smoothness_weight", 0.0)),
        "evaluated_updates": last_step,
        "trace_rows": len(rows),
        "cumulative_forward_iterations": int(
            sum(int(row["forward_steps"]) for row in rows)
        ),
        "optimizer_update_geometry_path": path_summary,
        "endpoint": _compact(run, "endpoint", rows[-1]),
        "best_fit": _compact(run, "best-fit", best),
        "first_inversion": (
            None
            if first_inversion is None
            else _compact(run, "first-inversion", first_inversion)
        ),
        "minimum_detF": _compact(run, "minimum-detF", minimum_detf),
        "reported_best_step": summary.get("best_step"),
        "failure": summary.get("failure"),
    }


def _match_actual_states(
    left_name: str,
    right_name: str,
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    left = [row for row in left_rows if int(row["step"]) > 0]
    right = [row for row in right_rows if int(row["step"]) > 0]
    if not left or not right:
        return {
            "available": False,
            "reason": "insufficient_post_update_states",
            "eligible_state_counts": {left_name: len(left), right_name: len(right)},
            "fit_tolerance_mm": FIT_TOLERANCE_MM,
            "motion_tolerance_mm": MOTION_TOLERANCE_MM,
            "interpolation": False,
        }
    feasible: list[tuple[tuple[float, ...], dict[str, Any], dict[str, Any]]] = []
    nearest: tuple[tuple[float, ...], dict[str, Any], dict[str, Any]] | None = None
    for left_row in left:
        for right_row in right:
            fit_delta = abs(
                float(left_row["fit_rms_mm"]) - float(right_row["fit_rms_mm"])
            )
            motion_delta = abs(
                float(left_row["motion_rms_mm"]) - float(right_row["motion_rms_mm"])
            )
            mean_fit = 0.5 * (
                float(left_row["fit_rms_mm"]) + float(right_row["fit_rms_mm"])
            )
            distance_squared = (fit_delta / FIT_TOLERANCE_MM) ** 2 + (
                motion_delta / MOTION_TOLERANCE_MM
            ) ** 2
            nearest_score = (
                distance_squared,
                mean_fit,
                int(left_row["step"]),
                int(right_row["step"]),
            )
            candidate = (nearest_score, left_row, right_row)
            if nearest is None or candidate[0] < nearest[0]:
                nearest = candidate
            if fit_delta <= FIT_TOLERANCE_MM and motion_delta <= MOTION_TOLERANCE_MM:
                score = (
                    mean_fit,
                    distance_squared,
                    int(left_row["step"]) + int(right_row["step"]),
                    int(left_row["step"]),
                    int(right_row["step"]),
                )
                feasible.append((score, left_row, right_row))
    if not feasible:
        assert nearest is not None
        _, left_row, right_row = nearest
        return {
            "available": False,
            "reason": "no_pair_within_both_tolerances",
            "selection": (
                "No post-update actual saved-state pair satisfies both tolerances; "
                "nearest values are diagnostics only."
            ),
            "nearest_diagnostic": {
                "states": {
                    left_name: _compact(left_name, "nearest-only", left_row),
                    right_name: _compact(right_name, "nearest-only", right_row),
                },
                "fit_delta_mm": abs(
                    float(left_row["fit_rms_mm"]) - float(right_row["fit_rms_mm"])
                ),
                "motion_delta_mm": abs(
                    float(left_row["motion_rms_mm"]) - float(right_row["motion_rms_mm"])
                ),
            },
            "fit_tolerance_mm": FIT_TOLERANCE_MM,
            "motion_tolerance_mm": MOTION_TOLERANCE_MM,
            "interpolation": False,
        }
    _, left_row, right_row = min(feasible, key=lambda item: item[0])
    residual_left = float(left_row[PRIMARY_RESIDUAL])
    low_left = float(left_row[PRIMARY_LOW_FREQUENCY])
    return {
        "available": True,
        "selection": (
            "Lowest mean fit among post-update actual saved-state pairs within both "
            "tolerances; then smallest normalized mismatch and earliest steps."
        ),
        "states": {
            left_name: _compact(left_name, "matched", left_row),
            right_name: _compact(right_name, "matched", right_row),
        },
        "fit_delta_mm": abs(
            float(left_row["fit_rms_mm"]) - float(right_row["fit_rms_mm"])
        ),
        "motion_delta_mm": abs(
            float(left_row["motion_rms_mm"]) - float(right_row["motion_rms_mm"])
        ),
        "right_minus_left": {
            "fit_rms_mm": float(right_row["fit_rms_mm"])
            - float(left_row["fit_rms_mm"]),
            "motion_rms_mm": float(right_row["motion_rms_mm"])
            - float(left_row["motion_rms_mm"]),
            "primary_residual_highpass_mm": float(right_row[PRIMARY_RESIDUAL])
            - residual_left,
            "primary_residual_reduction_fraction": (
                None
                if residual_left == 0.0
                else 1.0 - float(right_row[PRIMARY_RESIDUAL]) / residual_left
            ),
            "smoothness_C": float(right_row["smoothness_C"])
            - float(left_row["smoothness_C"]),
            "low_frequency_projection_ratio": (
                None
                if low_left <= 0.0
                else float(right_row[PRIMARY_LOW_FREQUENCY]) / low_left
            ),
        },
        "fit_tolerance_mm": FIT_TOLERANCE_MM,
        "motion_tolerance_mm": MOTION_TOLERANCE_MM,
        "interpolation": False,
    }


def _markevery(rows: list[dict[str, Any]]) -> int:
    return max(1, len(rows) // 12)


def _plot_curve(
    axis: Any,
    rows: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    run: str,
) -> None:
    style = STYLES[run]
    axis.plot(
        [float(row[x_key]) for row in rows],
        [float(row[y_key]) for row in rows],
        color=style["color"],
        linestyle=style["linestyle"],
        marker=style["marker"],
        markevery=_markevery(rows),
        markersize=4.2,
        markerfacecolor="white",
        markeredgewidth=1.0,
        linewidth=2.0,
        label=style["label"],
    )


def _plot_trajectories(traces: dict[str, list[dict[str, Any]]], output: Path) -> Path:
    figure, axes = plt.subplots(3, 2, figsize=(12, 13), constrained_layout=True)
    panels = (
        (axes[0, 0], "fit_rms_mm", "Fit RMS (mm)", "Target fit"),
        (axes[0, 1], "motion_rms_mm", "Motion RMS (mm)", "Expression motion"),
        (
            axes[1, 0],
            "inverted_all_cells",
            "Inverted tetrahedra",
            "Mechanical inversions",
        ),
        (axes[1, 1], "detF_min", "Minimum det(F)", "Minimum volume ratio"),
        (
            axes[2, 0],
            PRIMARY_RESIDUAL,
            "Union residual HP RMS (mm)",
            "Primary surface score",
        ),
        (
            axes[2, 1],
            "forward_steps",
            "Forward solver iterations",
            "Per-state physical solve effort",
        ),
    )
    for axis, key, ylabel, title in panels:
        for run in RUNS:
            _plot_curve(axis, traces[run], "step", key, run)
        axis.set(xlabel="Adam updates", ylabel=ylabel, title=title)
        if key == "inverted_all_cells":
            axis.set_yscale("symlog", linthresh=1.0)
            axis.set_ylabel("Inverted tetrahedra (symmetric log scale)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    path = output / "optimizer-trajectories.png"
    figure.savefig(path, dpi=210)
    plt.close(figure)
    return path


def _plot_attained_states(
    traces: dict[str, list[dict[str, Any]]], output: Path
) -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for run in RUNS:
        _plot_curve(axes[0, 0], traces[run], "motion_rms_mm", PRIMARY_RESIDUAL, run)
        _plot_curve(axes[0, 1], traces[run], "fit_rms_mm", PRIMARY_RESIDUAL, run)
        _plot_curve(axes[1, 0], traces[run], "motion_rms_mm", "inverted_all_cells", run)
        _plot_curve(axes[1, 1], traces[run], "fit_rms_mm", "inverted_all_cells", run)
    axes[0, 0].set(
        xlabel="Motion RMS (mm)",
        ylabel="Union residual HP RMS (mm)",
        title="Surface score versus attained motion",
    )
    axes[0, 1].set(
        xlabel="Fit RMS (mm)",
        ylabel="Union residual HP RMS (mm)",
        title="Surface score versus attained fit",
    )
    axes[1, 0].set(
        xlabel="Motion RMS (mm)",
        ylabel="Inverted tetrahedra (symmetric log scale)",
        title="Inversions versus attained motion",
    )
    axes[1, 1].set(
        xlabel="Fit RMS (mm)",
        ylabel="Inverted tetrahedra (symmetric log scale)",
        title="Inversions versus attained fit",
    )
    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    for axis in axes[1]:
        axis.set_yscale("symlog", linthresh=1.0)
    path = output / "attained-state-comparison.png"
    figure.savefig(path, dpi=210)
    plt.close(figure)
    return path


def _load_surface(
    directory: Path, step: int, point_ids: np.ndarray
) -> tuple[np.ndarray, Path]:
    path = directory / f"surface-{step:04d}.npz"
    with np.load(path, allow_pickle=False) as saved:
        if int(saved["step"]) != step:
            raise ValueError(f"surface step receipt mismatch: {path}")
        if not np.array_equal(saved["point_ids"], point_ids):
            raise ValueError(f"surface point IDs differ from frozen fixture: {path}")
        displacement = np.asarray(saved["u"], dtype=np.float64)
    if displacement.shape != (len(point_ids), 3) or not np.isfinite(displacement).all():
        raise ValueError(f"invalid surface displacement: {path}")
    return displacement, path


def _triangle_segments(
    points: np.ndarray, triangles: np.ndarray, y: float
) -> np.ndarray:
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


def _render_sections(
    skin: pv.PolyData,
    states: list[tuple[str, np.ndarray, dict[str, str]]],
    path: Path,
) -> None:
    triangles = np.asarray(skin.faces, dtype=np.int64).reshape(-1, 4)[:, 1:]
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for axis, y in zip(axes, (2.170, 2.180, 2.190), strict=True):
        for label, displacement, style in states:
            segments = _triangle_segments(
                np.asarray(skin.points, dtype=np.float64) + displacement,
                triangles,
                y,
            )
            collection = LineCollection(
                1000.0 * segments[:, :, (0, 2)],
                colors=style["color"],
                linestyles=style["linestyle"],
                linewidths=1.15,
                label=label,
            )
            axis.add_collection(collection)
            midpoints = 1000.0 * segments.mean(axis=1)[:, (0, 2)]
            stride = max(1, len(midpoints) // 20)
            sampled = midpoints[::stride]
            axis.scatter(
                sampled[:, 0],
                sampled[:, 1],
                color=style["color"],
                marker=style["marker"],
                s=7,
                linewidths=0,
                zorder=3,
            )
        axis.set(
            title=f"y = {1000.0 * y:.0f} mm",
            xlim=(1414, 1460),
            ylim=(40, 115),
            xlabel="x (mm)",
            ylabel="z (mm)",
        )
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)
    axes[0].legend(loc="upper left", frameon=False, fontsize=6.5)
    figure.suptitle(
        "Exact skin-triangle sections at selected matched states", fontsize=12
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def _matched_section_figures(
    matches: dict[str, dict[str, Any]],
    directories: dict[str, Path],
    output: Path,
) -> tuple[list[Path], list[Path]]:
    skin_path = FIXTURE / "skin.vtp"
    volume_path = FIXTURE / "volume.vtu"
    skin = pv.read(skin_path)
    volume = pv.read(volume_path)
    point_ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    target = np.asarray(volume.point_data["Smile"], dtype=np.float64)[point_ids]
    if target.shape != (len(point_ids), 3) or not np.isfinite(target).all():
        raise ValueError("frozen target field is invalid")
    figures: list[Path] = []
    sources: list[Path] = [skin_path, volume_path]
    for pair_name, match in matches.items():
        if not match["available"]:
            continue
        states: list[tuple[str, np.ndarray, dict[str, str]]] = [
            ("Target", target, TARGET_STYLE)
        ]
        for run, row in match["states"].items():
            step = int(row["step"])
            displacement, source = _load_surface(directories[run], step, point_ids)
            states.append(
                (f"{STYLES[run]['label']}, update {step}", displacement, STYLES[run])
            )
            sources.append(source)
        path = output / "sections" / f"{pair_name}.png"
        _render_sections(skin, states, path)
        figures.append(path)
    return figures, sources


def _add_lights(plotter: pv.Plotter, camera: dict[str, Any]) -> None:
    focus = np.asarray(camera["focal_point"], dtype=np.float64)
    backward = np.asarray(camera["position"], dtype=np.float64) - focus
    backward /= np.linalg.norm(backward)
    right = np.cross(np.asarray(camera["view_up"], dtype=np.float64), backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    for position, intensity in (
        (focus + 0.3 * (0.72 * right + 0.35 * up + 0.60 * backward), 0.85),
        (focus + 0.3 * backward, 0.20),
    ):
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
        raise AssertionError("saved geometry must use flat shading")
    _add_lights(plotter, camera)
    plotter.enable_parallel_projection()
    plotter.camera.position = camera["position"]
    plotter.camera.focal_point = camera["focal_point"]
    plotter.camera.up = camera["view_up"]
    plotter.camera.parallel_scale = camera["parallel_scale"]
    plotter.set_background(BACKGROUND)
    plotter.reset_camera_clipping_range()
    path.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(path)
    plotter.close()


def _compose_endpoint_plate(
    rows: list[tuple[str, list[tuple[Path, str]]]], title: str, path: Path
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    figure.patch.set_facecolor(BACKGROUND)
    for axis_row, (view_label, panels) in zip(axes, rows, strict=True):
        for axis, (image, panel_label) in zip(axis_row, panels, strict=True):
            axis.imshow(plt.imread(image))
            axis.set_title(f"{view_label}\n{panel_label}", color="white", fontsize=8.5)
            axis.set_axis_off()
    figure.suptitle(title, color="white", fontsize=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, facecolor=figure.get_facecolor())
    plt.close(figure)


def _endpoint_geometry_figures(
    traces: dict[str, list[dict[str, Any]]],
    directories: dict[str, Path],
    output: Path,
) -> tuple[list[Path], list[Path]]:
    skin_path = FIXTURE / "skin.vtp"
    volume_path = FIXTURE / "volume.vtu"
    skin = pv.read(skin_path)
    volume = pv.read(volume_path)
    point_ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    target = np.asarray(volume.point_data["Smile"], dtype=np.float64)[point_ids]
    cameras_receipt = json.loads(CAMERAS.read_text())
    cameras = {view["id"]: view for view in cameras_receipt["views"]}
    view_ids = ("side-context", "region1-mouth-corner")
    if set(view_ids) - set(cameras):
        raise ValueError("frozen camera receipt lacks endpoint comparison views")
    jobs = {
        "unmatched-off-endpoints": ("original-off", "conservative-off"),
        "unmatched-on-endpoints": ("original-on", "conservative-on"),
    }
    figures: list[Path] = []
    sources: list[Path] = [skin_path, volume_path, CAMERAS]
    for job, (original, conservative) in jobs.items():
        endpoint_states: list[tuple[str, np.ndarray]] = [("Target smile", target)]
        for run in (original, conservative):
            row = traces[run][-1]
            step = int(row["step"])
            displacement, source = _load_surface(directories[run], step, point_ids)
            sources.append(source)
            endpoint_states.append(
                (
                    f"{STYLES[run]['label']} · update {step}\n"
                    f"fit {float(row['fit_rms_mm']):.3f} mm · "
                    f"motion {float(row['motion_rms_mm']):.3f} mm",
                    displacement,
                )
            )
        plate_rows: list[tuple[str, list[tuple[Path, str]]]] = []
        for view_id in view_ids:
            view = cameras[view_id]
            panels: list[tuple[Path, str]] = []
            for index, (label, displacement) in enumerate(endpoint_states):
                panel = (
                    output / "geometry/panels" / job / view_id / f"panel-{index}.png"
                )
                _render_surface(skin, displacement, view["camera"], panel)
                figures.append(panel)
                panels.append((panel, label))
            plate_rows.append((str(view["label"]), panels))
        plate = output / "geometry" / f"{job}.png"
        _compose_endpoint_plate(
            plate_rows,
            "Unmatched attained endpoints · exact saved surfaces · no exaggeration",
            plate,
        )
        figures.append(plate)
    return figures, sources


def _state_rows(
    outcomes: dict[str, dict[str, Any]], matches: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for outcome in outcomes.values():
        for key in ("endpoint", "best_fit", "first_inversion", "minimum_detF"):
            row = outcome[key]
            if row is not None:
                rows.append(row)
    for pair, match in matches.items():
        if not match["available"]:
            continue
        rows.extend({"pair": pair, **row} for row in match["states"].values())
    return rows


def _main(cfg: Config) -> dict[str, Any]:
    output = cfg.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    directories = {
        "original-off": GROUP / "data/24-learned-axis",
        "original-on": GROUP / "data/25-learned-axis-smooth",
        "conservative-off": cfg.off_dir,
        "conservative-on": cfg.on_dir,
    }
    trace_paths = {
        run: directory / "trace.csv" for run, directory in directories.items()
    }
    summary_paths = {
        run: directory / "summary.json" for run, directory in directories.items()
    }
    provenance_paths = {
        run: directory / "provenance.json" for run, directory in directories.items()
    }
    required = [
        *trace_paths.values(),
        *summary_paths.values(),
        *provenance_paths.values(),
        SETTINGS,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    traces = {run: _read_trace(path) for run, path in trace_paths.items()}
    run_summaries = {
        run: json.loads(path.read_text()) for run, path in summary_paths.items()
    }
    settings = json.loads(SETTINGS.read_text())
    original_rate = float(settings["followup"]["original_rate"])
    conservative_rate = float(settings["learning_rates"]["learned-axis"])
    expected_rates = {
        "original-off": original_rate,
        "original-on": original_rate,
        "conservative-off": conservative_rate,
        "conservative-on": conservative_rate,
    }
    expected_settings_hashes = {
        "original-off": str(settings["followup"]["original_settings"]["sha256"]),
        "original-on": str(settings["followup"]["original_settings"]["sha256"]),
        "conservative-off": _digest(SETTINGS),
        "conservative-on": _digest(SETTINGS),
    }
    provenances = {
        run: json.loads(path.read_text()) for run, path in provenance_paths.items()
    }
    for run, provenance in provenances.items():
        if float(provenance["optimizer"]["lr"]) != expected_rates[run]:
            raise ValueError(
                f"provenance optimizer rate differs from frozen rate: {run}"
            )
        if str(provenance["settings"]["sha256"]) != expected_settings_hashes[run]:
            raise ValueError(
                f"provenance settings hash differs from frozen hash: {run}"
            )
    outcomes = {
        run: _run_outcome(
            run,
            directories[run],
            traces[run],
            run_summaries[run],
            expected_rates[run],
        )
        for run in RUNS
    }
    matches = {
        "conservative-off-vs-on": _match_actual_states(
            "conservative-off",
            "conservative-on",
            traces["conservative-off"],
            traces["conservative-on"],
        ),
        "off-original-vs-conservative": _match_actual_states(
            "original-off",
            "conservative-off",
            traces["original-off"],
            traces["conservative-off"],
        ),
        "on-original-vs-conservative": _match_actual_states(
            "original-on",
            "conservative-on",
            traces["original-on"],
            traces["conservative-on"],
        ),
    }
    plots = [
        _plot_trajectories(traces, output),
        _plot_attained_states(traces, output),
    ]
    section_plots, geometry_inputs = _matched_section_figures(
        matches, directories, output
    )
    plots.extend(section_plots)
    endpoint_plots, endpoint_inputs = _endpoint_geometry_figures(
        traces, directories, output
    )
    plots.extend(endpoint_plots)
    states_path = output / "selected-states.csv"
    _write_csv(states_path, _state_rows(outcomes, matches))

    snapshot = output / "sources/52-compare-conservative-rate.py"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(__file__), snapshot)
    snapshot.chmod(0o444)
    if _digest(snapshot) != _digest(Path(__file__)):
        raise ValueError("saved comparison source differs from live source")
    input_paths = [*required, *geometry_inputs, *endpoint_inputs]
    outputs = {
        str(path.relative_to(output)): _record(path) for path in [*plots, states_path]
    }
    summary = {
        "status": "completed_saved_state_conservative_rate_comparison",
        "scope": (
            "CPU-only comparison of exact saved trace rows and surface arrays; no "
            "forward solve, adjoint solve, optimizer update, interpolation, or smoothing"
        ),
        "selection_rule": {
            "fit_tolerance_mm": FIT_TOLERANCE_MM,
            "motion_tolerance_mm": MOTION_TOLERANCE_MM,
            "post_update_only": True,
            "interpolation": False,
            "ranking": (
                "lowest mean fit, then smallest normalized fit-motion mismatch, "
                "then earliest summed and individual updates"
            ),
        },
        "styles": STYLES,
        "runs": outcomes,
        "matches": matches,
        "inputs": {str(path.resolve()): _record(path) for path in input_paths},
        "outputs": outputs,
        "source": {
            "snapshot": _record(snapshot),
            "live_at_generation": _record(Path(__file__)),
        },
    }
    _write_json(output / "summary.json", summary)
    return summary


def main(cfg: Config) -> None:
    summary = _main(cfg)
    cherries.log_metric("comparison/runs", len(summary["runs"]))
    cherries.log_metric(
        "comparison/available_matches",
        sum(bool(match["available"]) for match in summary["matches"].values()),
    )
    cherries.log_metric("comparison/figures", len(summary["outputs"]) - 1)
    cherries.log_output(cfg.output_dir)


if __name__ == "__main__":
    cherries.main(
        main,
        profile=None if os.getenv("DEBUG") == "1" else ProfileCometNoCommit,
    )
