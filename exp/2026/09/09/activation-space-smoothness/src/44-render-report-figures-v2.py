# ruff: noqa: EM101, EM102, TRY003
"""Redraw report figures from exact saved traces and surface states."""

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
RAW6_OFF_DIR = ROOT / "exp/2026/09/08/local-skin-prestrain/data/30-refit-no-skin"
OUTPUT = GROUP / "data/44-report-figures-v2"
PRIMARY_RESIDUAL = "primary_union_normal_residual_highpass_5mm_rms_mm"
PRIMARY_LOW_FREQUENCY = "primary_union_low_frequency_normal_target_projection"

CASES = ("axis-off", "axis-on", "raw6-off", "raw6-on")
LABELS = {
    "axis-off": "Learned axis, S(C) off",
    "axis-on": "Learned axis, S(C) on",
    "raw6-off": "Corrected Raw6, S(C) off (archived)",
    "raw6-on": "Corrected Raw6, S(C) on",
}
STYLES = {
    "axis-off": {"color": "#D55E00", "linestyle": "-", "marker": "o"},
    "axis-on": {"color": "#0072B2", "linestyle": "--", "marker": "s"},
    "raw6-off": {"color": "#CC79A7", "linestyle": "-", "marker": "^"},
    "raw6-on": {"color": "#009E73", "linestyle": "--", "marker": "D"},
}
HISTORICAL = {
    "psd-off-64": ("Historical PSD, smoothness off, step 64", "#E69F00", "P"),
    "psd-on-64": ("Historical PSD, smoothness on, step 64", "#6A3D9A", "X"),
    "psd-off-1024": ("Historical PSD, smoothness off, step 1024", "#8C564B", "v"),
}
TARGET_STYLE = {"color": "#333333", "linestyle": ":", "marker": "."}


class Config(cherries.BaseConfig):
    """Fixed inputs for the report-only redraw."""

    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output_dir: Path = OUTPUT


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


def _read_rows(
    path: Path, *, require_unique_increasing_steps: bool = True
) -> list[dict[str, Any]]:
    with path.open(newline="") as stream:
        raw = list(csv.DictReader(stream))
    if not raw:
        raise ValueError(f"empty trace: {path}")
    rows: list[dict[str, Any]] = []
    for source in raw:
        row: dict[str, Any] = {}
        for key, value in source.items():
            if value in (None, ""):
                continue
            try:
                row[key] = float(value)
            except ValueError:
                row[key] = value
        rows.append(row)
    steps = [int(row["step"]) for row in rows]
    if require_unique_increasing_steps and steps != sorted(set(steps)):
        raise ValueError(f"trace steps are not unique and increasing: {path}")
    return rows


def _markevery(rows: list[dict[str, Any]]) -> int:
    return max(1, len(rows) // 12)


def _plot_line(
    axis: Any,
    rows: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    case: str,
) -> None:
    style = STYLES[case]
    axis.plot(
        [row[x_key] for row in rows],
        [row[y_key] for row in rows],
        color=style["color"],
        linestyle=style["linestyle"],
        linewidth=2.0,
        marker=style["marker"],
        markersize=4.2,
        markevery=_markevery(rows),
        markerfacecolor="white",
        markeredgewidth=1.0,
        label=LABELS[case],
    )


def _plot_saved_traces(
    traces: dict[str, list[dict[str, Any]]], output: Path
) -> list[Path]:
    paths: list[Path] = []
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    panels = (
        (axes[0, 0], "fit_rms_mm", "Fit RMS (mm)", "Target fit"),
        (axes[0, 1], "motion_rms_mm", "Motion RMS (mm)", "Expression motion"),
        (
            axes[1, 0],
            PRIMARY_RESIDUAL,
            "Union residual HP RMS (mm)",
            "Primary surface score",
        ),
        (
            axes[1, 1],
            PRIMARY_LOW_FREQUENCY,
            "Union low-frequency target projection",
            "Low-frequency target-motion retention",
        ),
    )
    for axis, key, ylabel, title in panels:
        for case in CASES:
            _plot_line(axis, traces[case], "step", key, case)
        axis.set(xlabel="Adam updates", ylabel=ylabel, title=title)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    path = output / "new-pairs-trajectories.png"
    figure.savefig(path, dpi=210)
    plt.close(figure)
    paths.append(path)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for axis, key, ylabel, title in (
        (axes[0], "smoothness_C", "S(C)", "Optimized C variation"),
        (axes[1], "smoothness_Z", "S(Z)", "Effective activation Z variation"),
    ):
        for case in CASES:
            available = [row for row in traces[case] if key in row]
            if available:
                _plot_line(axis, available, "step", key, case)
        positive = [
            float(value)
            for line in axis.lines
            for value in line.get_ydata()
            if float(value) > 0.0
        ]
        if positive and max(positive) / min(positive) >= 100.0:
            axis.set_yscale("symlog", linthresh=0.5 * min(positive))
        axis.set(xlabel="Adam updates", ylabel=ylabel, title=title)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    path = output / "new-pairs-activation-variation.png"
    figure.savefig(path, dpi=210)
    plt.close(figure)
    paths.append(path)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for case in CASES:
        _plot_line(axes[0], traces[case], "fit_rms_mm", PRIMARY_RESIDUAL, case)
        _plot_line(axes[1], traces[case], "motion_rms_mm", PRIMARY_RESIDUAL, case)
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
    paths.append(path)
    return paths


def _plot_historical(
    traces: dict[str, list[dict[str, Any]]], source: Path, output: Path
) -> Path:
    rows = _read_rows(source, require_unique_increasing_steps=False)
    figure, axis = plt.subplots(figsize=(8, 6), constrained_layout=True)
    for case in CASES:
        row = traces[case][-1]
        style = STYLES[case]
        axis.scatter(
            row["fit_rms_mm"],
            row[PRIMARY_RESIDUAL],
            color=style["color"],
            edgecolor="#222222",
            linewidth=0.45,
            label=LABELS[case],
            marker=style["marker"],
            s=72,
        )
    for row in rows:
        case = str(row["case"])
        label, color, marker = HISTORICAL[case]
        axis.scatter(
            row["fit_rms_mm"],
            row[PRIMARY_RESIDUAL],
            color=color,
            edgecolor="#222222",
            linewidth=0.45,
            label=label,
            marker=marker,
            s=72,
        )
    axis.set(
        xlabel="Uniform fit RMS (mm)",
        ylabel="Union residual HP RMS (mm)",
        title="Attained fit and surface error; historical PSD is contextual",
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=7)
    path = output / "historical-context-tradeoff.png"
    figure.savefig(path, dpi=220)
    plt.close(figure)
    return path


def _raw_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _plot_calibration(source: Path, output: Path) -> Path:
    rows = _raw_csv(source)
    steps = [int(row["step"]) for row in rows]
    if steps != list(range(17)):
        raise ValueError(f"expected exact audited steps 0..16, got {steps}")
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    panels = (
        (axes[0, 0], "fit_rms_mm", "Fit RMS (mm)", "Target fit", "linear"),
        (axes[0, 1], "smoothness_C", "S(C)", "C variation", "symlog"),
        (
            axes[1, 0],
            "gradient_rms",
            "Fitting-gradient RMS",
            "Adjoint fitting gradient",
            "log",
        ),
        (
            axes[1, 1],
            "inverted_all_cells",
            "Inverted tetrahedra",
            "Mechanical diagnostic",
            "linear",
        ),
    )
    executions = {
        "calibration": ("Selected calibration pilot", "#6A3D9A", "-", "o"),
        "primary": ("Primary Axis-off", "#D55E00", "--", "s"),
    }
    for axis, key, ylabel, title, scale in panels:
        for execution, (label, color, linestyle, marker) in executions.items():
            values = [float(row[f"{execution}_{key}"]) for row in rows]
            axis.plot(
                steps,
                values,
                color=color,
                linestyle=linestyle,
                linewidth=2.0,
                marker=marker,
                markersize=4.0,
                markerfacecolor="white",
                label=label,
            )
        if scale == "symlog":
            positive = [
                float(row[f"{execution}_{key}"])
                for row in rows
                for execution in executions
                if float(row[f"{execution}_{key}"]) > 0.0
            ]
            axis.set_yscale("symlog", linthresh=0.5 * min(positive))
        elif scale == "log":
            axis.set_yscale("log")
        axis.set(xlabel="Adam updates", ylabel=ylabel, title=title, xlim=(0, 16))
        axis.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False, fontsize=9)
    figure.suptitle(
        "Same seed, initialization, settings, and shared sources; separate executions",
        fontsize=13,
    )
    path = output / "calibration-vs-primary-first16.png"
    figure.savefig(path, dpi=220)
    plt.close(figure)
    return path


def _plot_learning_rate_calibration(sources: list[Path], output: Path) -> Path:
    palette = ("#0072B2", "#009E73", "#CC79A7", "#D55E00")
    linestyles = ("-", "--", "-.", ":")
    markers = ("o", "s", "^", "D")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for source, color, linestyle, marker in zip(
        sources, palette, linestyles, markers, strict=True
    ):
        rows = _read_rows(source)
        steps = [int(row["step"]) for row in rows]
        if steps != list(range(17)):
            raise ValueError(f"expected exact calibration steps 0..16: {source}")
        rates = {float(row["learning_rate"]) for row in rows}
        if len(rates) != 1:
            raise ValueError(
                f"learning rate changed within calibration trace: {source}"
            )
        rate = rates.pop()
        for axis, key in zip(axes, ("fit_rms_mm", "inverted_all_cells"), strict=True):
            axis.plot(
                steps,
                [float(row[key]) for row in rows],
                color=color,
                linestyle=linestyle,
                linewidth=2.0,
                marker=marker,
                markersize=4.0,
                markerfacecolor="white",
                label=f"learning rate {rate:.3f}",
            )
    axes[0].set(
        xlabel="Adam updates",
        ylabel="Fit RMS (mm)",
        title="Target fit during 16-update calibration",
        xlim=(0, 16),
    )
    axes[1].set(
        xlabel="Adam updates",
        ylabel="Inverted tetrahedra",
        title="Inverted-cell count during calibration",
        xlim=(0, 16),
    )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    figure.suptitle(
        "Learning-rate calibration trajectories (separate 16-update pilots)",
        fontsize=13,
    )
    path = output / "learning-rate-calibration-trajectories.png"
    figure.savefig(path, dpi=220)
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
    base: pv.PolyData,
    states: list[tuple[str, np.ndarray, dict[str, str]]],
    path: Path,
) -> None:
    triangles = np.asarray(base.faces, dtype=np.int64).reshape(-1, 4)[:, 1:]
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 4.0), constrained_layout=True)
    for axis, y in zip(axes, (2.170, 2.180, 2.190), strict=True):
        for label, displacement, style in states:
            segments = _triangle_segments(
                np.asarray(base.points, dtype=np.float64) + displacement,
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
    axes[0].legend(loc="upper left", frameon=False, fontsize=7)
    figure.suptitle("Exact skin-triangle sections", fontsize=12)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def _render_all_sections(output: Path) -> tuple[list[Path], list[Path]]:
    skin_path = FIXTURE / "skin.vtp"
    volume_path = FIXTURE / "volume.vtu"
    skin = pv.read(skin_path)
    volume = pv.read(volume_path)
    point_ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    target = np.asarray(volume.point_data["Smile"], dtype=np.float64)[point_ids]
    if target.shape != (len(point_ids), 3) or not np.isfinite(target).all():
        raise ValueError("frozen target field is invalid")
    surfaces: dict[str, tuple[np.ndarray, Path]] = {
        "axis-off-11": _load_surface(GROUP / "data/24-learned-axis", 11, point_ids),
        "axis-on-11": _load_surface(
            GROUP / "data/25-learned-axis-smooth", 11, point_ids
        ),
        "axis-on-128": _load_surface(
            GROUP / "data/25-learned-axis-smooth", 128, point_ids
        ),
        "raw6-off-200": _load_surface(RAW6_OFF_DIR, 200, point_ids),
        "raw6-on-200": _load_surface(GROUP / "data/28-raw6-smooth", 200, point_ids),
    }
    target = np.asarray(target, dtype=np.float64)
    target_style = dict(TARGET_STYLE)
    jobs = {
        "learned-axis-matched.png": [
            ("Target", target, target_style),
            (
                "Axis-off matched update 11",
                surfaces["axis-off-11"][0],
                STYLES["axis-off"],
            ),
            ("Axis-on matched update 11", surfaces["axis-on-11"][0], STYLES["axis-on"]),
        ],
        "raw6-matched.png": [
            ("Target", target, target_style),
            (
                "Raw6-off matched update 200",
                surfaces["raw6-off-200"][0],
                STYLES["raw6-off"],
            ),
            (
                "Raw6-on matched update 200",
                surfaces["raw6-on-200"][0],
                STYLES["raw6-on"],
            ),
        ],
        "axis-on-endpoint-0128.png": [
            ("Target", target, target_style),
            (
                "Axis-on update 128 (unpaired)",
                surfaces["axis-on-128"][0],
                STYLES["axis-on"],
            ),
        ],
    }
    paths: list[Path] = []
    for name, states in jobs.items():
        path = output / "sections" / name
        _render_sections(skin, states, path)
        paths.append(path)
    inputs = [skin_path, volume_path, *(source for _, source in surfaces.values())]
    return paths, inputs


def _main(cfg: Config) -> dict[str, Any]:
    output = cfg.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    trace_paths = {
        "axis-off": GROUP / "data/24-learned-axis/trace.csv",
        "axis-on": GROUP / "data/25-learned-axis-smooth/trace.csv",
        "raw6-off": GROUP / "data/40-reuse-metrics/raw6-off-trace.csv",
        "raw6-on": GROUP / "data/28-raw6-smooth/trace.csv",
    }
    historical_path = GROUP / "data/40-comparison/historical-context.csv"
    calibration_path = (
        GROUP / "data/18-calibration-main-divergence-audit/trace-comparison.csv"
    )
    learning_rate_paths = [
        GROUP / f"data/18-learned-axis-calibration-refined/lr-{index}/trace.csv"
        for index in range(4)
    ]
    required = [
        *trace_paths.values(),
        historical_path,
        calibration_path,
        *learning_rate_paths,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    traces = {case: _read_rows(path) for case, path in trace_paths.items()}
    plots = _plot_saved_traces(traces, output)
    plots.append(_plot_historical(traces, historical_path, output))
    plots.append(_plot_calibration(calibration_path, output))
    plots.append(_plot_learning_rate_calibration(learning_rate_paths, output))
    sections, geometry_inputs = _render_all_sections(output)
    plots.extend(sections)

    snapshot = output / "sources/44-render-report-figures-v2.py"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(__file__), snapshot)
    snapshot.chmod(0o444)
    if _digest(snapshot) != _digest(Path(__file__)):
        raise ValueError("saved renderer source differs from live source")
    inputs = {
        str(path.resolve()): _record(path) for path in [*required, *geometry_inputs]
    }
    outputs = {str(path.relative_to(output)): _record(path) for path in plots}
    summary = {
        "status": "completed_saved_data_report_redraw",
        "scope": (
            "Matplotlib redraw from exact saved CSV rows and saved surface arrays; "
            "no mechanics solve, optimizer update, interpolation, smoothing, or raster recoloring"
        ),
        "palette": {case: {**STYLES[case], "label": LABELS[case]} for case in CASES},
        "inputs": inputs,
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
    cherries.log_metric("render/figures", len(summary["outputs"]))
    cherries.log_output(cfg.output_dir)


if __name__ == "__main__":
    cherries.main(
        main,
        profile=None if os.getenv("DEBUG") == "1" else ProfileCometNoCommit,
    )
