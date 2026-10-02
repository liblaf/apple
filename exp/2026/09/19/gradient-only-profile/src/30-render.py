"""Render a like-for-like 2D comparison of L2 and gradient-only fitting.

The metrics below are recomputed from each saved final displacement.  In
particular, they do not use the two objectives as comparable quantities:
gradient-only and L2 have different units and different minima.
"""

# ruff: noqa: EM102, TRY003

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import pydantic_settings as ps

from liblaf import cherries

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D

LOG = logging.getLogger(__name__)
GROUP = Path(__file__).resolve().parents[1]
BASELINE = (
    GROUP.parents[1] / "15" / "activation-direction-smoothness" / "data" / "tune-w0"
)
HEIGHTS = (0.05, 0.20)
MODES = ("unconstrained", "contraction_only", "learned_direction", "x_contraction")
LABELS = {
    "unconstrained": "Unconstrained symmetric B",
    "contraction_only": "Contraction only",
    "learned_direction": "Learned direction",
    "x_contraction": "Fixed x contraction",
}
COLORS = {
    "unconstrained": "#6a4c93",
    "contraction_only": "#27836f",
    "learned_direction": "#d27b21",
    "x_contraction": "#3b6fa1",
}
L2_COLOR = "#4d4a46"
GRADIENT_COLOR = "#b33a3a"
EDGE_COLOR = "#554f48"


class Config(cherries.BaseConfig):
    """Saved result locations and a Cherries-managed output folder."""

    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    gradient_dir: Path = cherries.input("10-gradient")
    baseline_dir: Path = BASELINE
    output: Path = cherries.output("30-figures", mkdir=True)


@dataclass(frozen=True)
class Case:
    method: str
    folder: Path
    summary: dict[str, Any]
    trace: dict[str, np.ndarray]
    points: np.ndarray
    triangles: np.ndarray
    muscle: np.ndarray
    top: np.ndarray
    top_all: np.ndarray
    u: np.ndarray
    height: float
    mode: str

    @property
    def name(self) -> str:
        return f"h{round(self.height * 1000):03d}-{self.mode}"

    @property
    def iterations(self) -> int:
        return int(self.summary["accepted_iterations"])

    @property
    def failure(self) -> Any:
        return self.summary.get("failure")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        return {key: np.asarray(saved[key]) for key in saved.files}


def _trace(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty trace: {path}")
    # The historical L2 trace predates the gradient diagnostics.  Objectives
    # remain useful as within-run histories, but the independent geometry
    # metrics below are the only cross-loss comparison.
    required = {"step", "objective", "normalized_loss"}
    absent = required - set(rows[0])
    if absent:
        raise ValueError(f"trace misses {sorted(absent)}: {path}")
    data = {
        key: np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        for key in rows[0]
    }
    if not all(np.all(np.isfinite(values)) for values in data.values()):
        raise ValueError(f"non-finite trace value: {path}")
    if np.any(np.diff(data["step"]) <= 0):
        raise ValueError(f"trace steps are not strictly increasing: {path}")
    return data


def _summary(folder: Path) -> dict[str, Any]:
    path = folder / "summary.json"
    document = json.loads(path.read_text())
    if isinstance(document, list):
        if len(document) != 1:
            raise ValueError(f"expected one summary record: {path}")
        document = document[0]
    required = {"mode", "height", "accepted_iterations", "failure", "initial", "final"}
    if not isinstance(document, dict) or required - document.keys():
        raise ValueError(f"invalid summary schema: {path}")
    return document


def _indices(values: np.ndarray, n: int, label: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    if values.ndim != 1 or len(values) < 2 or np.any(values < 0) or np.any(values >= n):
        raise ValueError(f"invalid {label} indices")
    if len(np.unique(values)) != len(values):
        raise ValueError(f"duplicate {label} indices")
    return values


def _case(method: str, folder: Path) -> Case:
    summary = _summary(folder)
    history = _load_npz(folder / "history.npz")
    checkpoint = _load_npz(folder / "checkpoint.npz")
    needed = {"points", "triangles", "muscle", "top", "u", "height", "mode"}
    absent = needed - history.keys()
    if absent:
        raise ValueError(f"history misses {sorted(absent)}: {folder}")
    points = np.asarray(history["points"], dtype=np.float64)
    triangles = np.asarray(history["triangles"], dtype=np.int64)
    muscle = np.asarray(history["muscle"], dtype=bool)
    if (
        points.ndim != 2
        or points.shape[1] != 2
        or triangles.ndim != 2
        or triangles.shape[1] != 3
    ):
        raise ValueError(f"invalid mesh arrays: {folder}")
    if (
        muscle.shape != (len(triangles),)
        or np.any(triangles < 0)
        or np.any(triangles >= len(points))
    ):
        raise ValueError(f"invalid triangle/muscle arrays: {folder}")
    top = _indices(history["top"], len(points), "top")
    # New runs save top_all explicitly.  The compatibility branch preserves the
    # historical L2 cases while still including their fixed corners.
    top_all = history.get("top_all")
    if top_all is None:
        top_all = np.flatnonzero(np.isclose(points[:, 1], points[:, 1].max()))
    top_all = _indices(top_all, len(points), "top_all")
    top_all = top_all[np.argsort(points[top_all, 0])]
    u = np.asarray(checkpoint.get("u"), dtype=np.float64)
    if u.shape != points.shape or not np.all(np.isfinite(u)):
        raise ValueError(f"checkpoint u must be finite (n,2): {folder}")
    height = float(summary["height"])
    mode = str(summary["mode"])
    if not np.isclose(float(np.asarray(history["height"]).item()), height):
        raise ValueError(f"history/summary height mismatch: {folder}")
    if str(np.asarray(history["mode"]).item()) != mode:
        raise ValueError(f"history/summary mode mismatch: {folder}")
    return Case(
        method,
        folder,
        summary,
        _trace(folder / "trace.csv"),
        points,
        triangles,
        muscle,
        top,
        top_all,
        u,
        height,
        mode,
    )


def _discover(method: str, root: Path) -> list[Case]:
    if not root.is_dir():
        raise FileNotFoundError(root)
    cases = [
        _case(method, path)
        for path in sorted(root.iterdir())
        if path.is_dir() and (path / "summary.json").is_file()
    ]
    if not cases:
        raise ValueError(f"no saved cases under {root}")
    return cases


def _lookup(cases: list[Case], height: float, mode: str) -> Case | None:
    matched = [
        case for case in cases if case.mode == mode and np.isclose(case.height, height)
    ]
    if len(matched) > 1:
        raise ValueError(f"duplicate {height:g}, {mode} cases in {matched[0].method}")
    return matched[0] if matched else None


def _target(points: np.ndarray, nodes: np.ndarray, height: float) -> np.ndarray:
    x = points[nodes, 0]
    x0, x1 = float(points[:, 0].min()), float(points[:, 0].max())
    fraction = (x - x0) / (x1 - x0)
    y = points[nodes, 1] + 4.0 * height * fraction * (1.0 - fraction)
    return np.column_stack((x, y))


def _metrics(case: Case) -> dict[str, float]:
    surface = case.points[case.top_all] + case.u[case.top_all]
    target = _target(case.points, case.top_all, case.height)
    error = surface - target
    reference = case.points[case.top_all]
    dx = np.diff(reference[:, 0])
    if np.any(dx <= 0):
        raise ValueError(f"non-increasing reference top chain: {case.folder}")
    span = float(dx.sum())
    gradient_error = np.diff(error, axis=0) / dx[:, None]
    # This is the target-relative second derivative in fixed reference x.  It
    # does not estimate geometric curvature of the deformed line.
    mid = 0.5 * (dx[:-1] + dx[1:])
    curvature_error = np.diff(gradient_error, axis=0) / mid[:, None]
    ref_tri = case.points[case.triangles]
    def_tri = (case.points + case.u)[case.triangles]
    ref_a, ref_b = ref_tri[:, 1] - ref_tri[:, 0], ref_tri[:, 2] - ref_tri[:, 0]
    def_a, def_b = def_tri[:, 1] - def_tri[:, 0], def_tri[:, 2] - def_tri[:, 0]
    twice_ref = ref_a[:, 0] * ref_b[:, 1] - ref_a[:, 1] * ref_b[:, 0]
    twice_def = def_a[:, 0] * def_b[:, 1] - def_a[:, 1] * def_b[:, 0]
    J = twice_def / twice_ref
    free_error = (
        case.points[case.top]
        + case.u[case.top]
        - _target(case.points, case.top, case.height)
    )
    return {
        "fit_rms": float(np.sqrt(np.mean(np.sum(free_error * free_error, axis=1)))),
        "fit_rms_all_top": float(np.sqrt(np.mean(np.sum(error * error, axis=1)))),
        "slope_rms": float(
            np.sqrt(np.sum(dx * np.sum(gradient_error * gradient_error, axis=1)) / span)
        ),
        "curvature_error": float(
            np.sqrt(
                np.sum(mid * np.sum(curvature_error * curvature_error, axis=1))
                / mid.sum()
            )
        ),
        "surface_y_rms": float(np.sqrt(np.mean(free_error[:, 1] ** 2))),
        "motion_rms": float(
            np.sqrt(np.mean(np.sum(case.u[case.top] * case.u[case.top], axis=1)))
        ),
        "peak_top_uy": float(case.u[case.top, 1].max()),
        "peak_ratio": float(case.u[case.top, 1].max() / case.height),
        "min_J": float(J.min()),
        "max_J": float(J.max()),
        "inverted_cells": float(np.sum(J <= 0)),
        "top_nodes_including_fixed_corners": float(len(case.top_all)),
    }


def _endpoint(case: Case, metric: dict[str, float]) -> dict[str, Any]:
    final = case.summary["final"]
    return {
        "method": case.method,
        "case": case.name,
        "height": case.height,
        "mode": case.mode,
        "accepted_iterations": case.iterations,
        "last_trace_step": int(case.trace["step"][-1]),
        "failure": case.failure,
        "trace_objective": float(case.trace["objective"][-1]),
        "trace_normalized_loss": float(case.trace["normalized_loss"][-1]),
        "saved_fit_rms": float(final.get("fit_rms", np.nan)),
        **metric,
    }


def _save(fig: plt.Figure, output: Path, stem: str) -> list[str]:
    names = []
    for suffix in ("png", "pdf"):
        name = f"{stem}.{suffix}"
        fig.savefig(output / name, dpi=240, bbox_inches="tight")
        names.append(name)
    plt.close(fig)
    return names


def _profile(
    axis: plt.Axes,
    l2: Case | None,
    gradient: Case | None,
    *,
    title: str,
    limits: tuple[float, float, float, float],
) -> None:
    reference = gradient or l2
    assert reference is not None
    target = _target(reference.points, reference.top_all, reference.height)
    axis.plot(
        target[:, 0], target[:, 1], color="black", lw=1.8, ls="--", label="target"
    )
    for case, color, label in (
        (l2, L2_COLOR, "L2 baseline"),
        (gradient, GRADIENT_COLOR, "gradient only"),
    ):
        if case is None:
            continue
        surface = case.points[case.top_all] + case.u[case.top_all]
        suffix = f" ({case.iterations} it.)" + (" failed" if case.failure else "")
        axis.plot(
            surface[:, 0], surface[:, 1], color=color, lw=1.65, label=label + suffix
        )
    axis.set_title(title, fontsize=10)
    if l2 is not None and l2.iterations < 1200:
        axis.text(
            0.02,
            0.95,
            f"L2 endpoint: {l2.iterations} it. (early stop)",
            transform=axis.transAxes,
            va="top",
            color="#9d2635",
            fontsize=7.2,
            weight="bold",
        )
    axis.set(xlim=limits[:2], ylim=limits[2:])
    axis.set_aspect("equal", adjustable="box")
    axis.grid(alpha=0.18)


def _render_profiles(l2: list[Case], gradient: list[Case], output: Path) -> list[str]:
    selected = [*l2, *gradient]
    row_limits = {
        height: _limits([case for case in selected if np.isclose(case.height, height)])
        for height in HEIGHTS
    }
    spans = [row_limits[height][3] - row_limits[height][2] for height in HEIGHTS]
    fig, axes = plt.subplots(
        2,
        4,
        figsize=(14, 4.6),
        gridspec_kw={"height_ratios": spans},
        layout="constrained",
    )
    for row, height in enumerate(HEIGHTS):
        for col, mode in enumerate(MODES):
            axis = axes[row, col]
            _profile(
                axis,
                _lookup(l2, height, mode),
                _lookup(gradient, height, mode),
                title=LABELS[mode],
                limits=row_limits[height],
            )
            if col == 0:
                axis.set_ylabel(f"h={height:g}\ny", rotation=0, ha="right", va="center")
            if row == 1:
                axis.set_xlabel("x")
    handles = [
        Line2D([0], [0], color="black", ls="--", label="target"),
        Line2D([0], [0], color=L2_COLOR, label="L2 baseline"),
        Line2D([0], [0], color=GRADIENT_COLOR, label="gradient-only fit"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle(
        "Saved top profiles: target shape versus final recorded fits", fontsize=14
    )
    return _save(fig, output, "profiles-all-cases")


def _slope_error(case: Case) -> tuple[np.ndarray, np.ndarray]:
    """Return edge midpoints and vector slope error on the full top chain."""
    reference = case.points[case.top_all]
    surface = reference + case.u[case.top_all]
    target = _target(case.points, case.top_all, case.height)
    dx = np.diff(reference[:, 0])
    if np.any(dx <= 0):
        raise ValueError(f"top chain must increase in x: {case.folder}")
    return 0.5 * (reference[:-1, 0] + reference[1:, 0]), np.diff(
        surface - target, axis=0
    ) / dx[:, None]


def _render_slope_errors(
    l2: list[Case], gradient: list[Case], output: Path
) -> list[str]:
    """Expose ripples directly when the profile overlays look nearly identical."""
    fig, axes = plt.subplots(2, 4, figsize=(14, 5.3), sharex=True, layout="constrained")
    for row, height in enumerate(HEIGHTS):
        for col, mode in enumerate(MODES):
            axis = axes[row, col]
            for case, vertical, horizontal, label in (
                (_lookup(l2, height, mode), L2_COLOR, "#a19c96", "L2 baseline"),
                (
                    _lookup(gradient, height, mode),
                    GRADIENT_COLOR,
                    "#de8d49",
                    "gradient only",
                ),
            ):
                if case is None:
                    continue
                x, slope = _slope_error(case)
                axis.plot(
                    x, slope[:, 1], color=vertical, lw=1.35, label=f"{label}: vertical"
                )
                axis.plot(
                    x,
                    slope[:, 0],
                    color=horizontal,
                    ls="--",
                    lw=1.0,
                    label=f"{label}: horizontal",
                )
            axis.axhline(0.0, color="0.35", lw=0.65)
            axis.set_title(LABELS[mode], fontsize=10)
            axis.grid(alpha=0.18)
            if col == 0:
                axis.set_ylabel(
                    f"h={height:g}\nerror slope", rotation=0, ha="right", va="center"
                )
            if row == 1:
                axis.set_xlabel("reference x")
    handles = [
        Line2D([0], [0], color=L2_COLOR, lw=1.5, label="L2: vertical"),
        Line2D([0], [0], color="#a19c96", lw=1.2, ls="--", label="L2: horizontal"),
        Line2D([0], [0], color=GRADIENT_COLOR, lw=1.5, label="gradient-only: vertical"),
        Line2D(
            [0],
            [0],
            color="#de8d49",
            lw=1.2,
            ls="--",
            label="gradient-only: horizontal",
        ),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle("Top-profile slope error relative to the target", fontsize=14)
    return _save(fig, output, "slope-error-profiles")


def _limits(cases: list[Case]) -> tuple[float, float, float, float]:
    clouds = [case.points for case in cases] + [case.points + case.u for case in cases]
    clouds.extend(_target(case.points, case.top_all, case.height) for case in cases)
    cloud = np.concatenate(clouds)
    span = max(np.ptp(cloud[:, 0]), np.ptp(cloud[:, 1]))
    pad = 0.035 * span
    return (
        float(cloud[:, 0].min() - pad),
        float(cloud[:, 0].max() + pad),
        float(cloud[:, 1].min() - pad),
        float(cloud[:, 1].max() + pad),
    )


def _mesh(
    axis: plt.Axes, case: Case, limits: tuple[float, float, float, float]
) -> None:
    deformed = case.points + case.u
    faces = np.where(
        case.muscle[:, None],
        np.array([[0.85, 0.51, 0.45]]),
        np.array([[0.91, 0.85, 0.75]]),
    )
    axis.add_collection(
        PolyCollection(
            deformed[case.triangles],
            facecolors=faces,
            edgecolors=EDGE_COLOR,
            linewidths=0.10,
            rasterized=True,
        )
    )
    target = _target(case.points, case.top_all, case.height)
    axis.plot(target[:, 0], target[:, 1], "k--", lw=1.0)
    m = _metrics(case)
    axis.text(
        0.5,
        -0.22,
        f"{case.iterations} it. · fit {m['fit_rms']:.3g} · slope {m['slope_rms']:.3g} · min J {m['min_J']:.3g}",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=7.6,
        clip_on=False,
    )
    axis.set(xlim=limits[:2], ylim=limits[2:])
    axis.set_aspect("equal", adjustable="box")


def _render_meshes(l2: list[Case], gradient: list[Case], output: Path) -> list[str]:
    selected = [
        case
        for height in HEIGHTS
        for case in (
            _lookup(l2, height, "learned_direction"),
            _lookup(gradient, height, "learned_direction"),
        )
        if case
    ]
    if not selected:
        return []
    row_limits = {
        height: _limits([case for case in selected if np.isclose(case.height, height)])
        for height in HEIGHTS
    }
    spans = [row_limits[height][3] - row_limits[height][2] for height in HEIGHTS]
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(10.2, 5.7),
        sharex=True,
        gridspec_kw={"height_ratios": spans},
        layout="constrained",
    )
    for row, height in enumerate(HEIGHTS):
        for col, (method, cases) in enumerate(
            (("L2 baseline", l2), ("gradient only", gradient))
        ):
            axis = axes[row, col]
            case = _lookup(cases, height, "learned_direction")
            if case is None:
                axis.set_axis_off()
                continue
            _mesh(axis, case, row_limits[height])
            if row == 0:
                axis.set_title(method)
            if col == 0:
                axis.set_ylabel(f"h={height:g}\ny", rotation=0, ha="right", va="center")
    fig.suptitle(
        "Learned-direction final deformed meshes; common physical axes", fontsize=14
    )
    fig.supxlabel("x")
    return _save(fig, output, "learned-direction-deformed-meshes")


def _render_metrics(rows: list[dict[str, Any]], output: Path) -> list[str]:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), layout="constrained")
    keys = (
        ("fit_rms", "position RMS"),
        ("slope_rms", "slope-error RMS"),
        ("curvature_error", "reference second-derivative error RMS"),
    )
    for axis, (key, label) in zip(axes, keys, strict=True):
        for height in HEIGHTS:
            subset = [row for row in rows if np.isclose(row["height"], height)]
            for method, marker, offset in (("l2", "o", -0.12), ("gradient", "s", 0.12)):
                values = [row for row in subset if row["method"] == method]
                for row in values:
                    x = (
                        MODES.index(str(row["mode"]))
                        + offset
                        + (0.025 if height == 0.20 else -0.025)
                    )
                    axis.scatter(
                        x,
                        row[key],
                        color=COLORS[str(row["mode"])],
                        marker=marker,
                        s=45,
                        zorder=3,
                    )
        axis.set(
            title=label,
            xticks=range(len(MODES)),
            xticklabels=["free", "contract", "learned", "fixed x"],
        )
        axis.grid(axis="y", alpha=0.2)
    legend = [
        Line2D([0], [0], color="0.25", marker="o", lw=0, label="L2 baseline"),
        Line2D([0], [0], color="0.25", marker="s", lw=0, label="gradient only"),
        Line2D(
            [0],
            [0],
            color="0.25",
            marker="o",
            markerfacecolor="none",
            lw=0,
            label="h=.05/.20: slight horizontal split",
        ),
    ]
    fig.legend(handles=legend, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle(
        "Independent final-shape metrics (lower is closer to target)", fontsize=14
    )
    return _save(fig, output, "final-shape-metrics")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(cfg: Config) -> None:
    gradient_dir = Path(cfg.gradient_dir)
    baseline_dir = Path(cfg.baseline_dir)
    cherries.log_input(gradient_dir)
    cherries.log_input(baseline_dir)
    gradient = _discover("gradient", gradient_dir)
    l2 = _discover("l2", baseline_dir)
    output = Path(cfg.output)
    output.mkdir(parents=True, exist_ok=False)
    rows = [_endpoint(case, _metrics(case)) for case in [*l2, *gradient]]
    rows.sort(
        key=lambda row: (row["height"], MODES.index(str(row["mode"])), row["method"])
    )
    _write_csv(output / "summary.csv", rows)
    comparisons = []
    for height in HEIGHTS:
        for mode in MODES:
            baseline, fitted = (
                _lookup(l2, height, mode),
                _lookup(gradient, height, mode),
            )
            if baseline is None or fitted is None:
                continue
            a, b = _metrics(baseline), _metrics(fitted)
            comparisons.append(
                {
                    "case": baseline.name,
                    "height": height,
                    "mode": mode,
                    "relative_gradient_minus_l2": {
                        key: (b[key] - a[key]) / a[key] if a[key] else None
                        for key in ("fit_rms", "slope_rms", "curvature_error")
                    },
                }
            )
    figures = {
        "profiles": _render_profiles(l2, gradient, output),
        "slope_errors": _render_slope_errors(l2, gradient, output),
        "learned_direction_meshes": _render_meshes(l2, gradient, output),
        "metrics": _render_metrics(rows, output),
    }
    receipt = {
        "purpose": "post-hoc comparison of saved L2 and gradient-only 2D profile fits",
        "inputs": {
            "gradient_dir": str(gradient_dir),
            "l2_baseline_dir": str(baseline_dir),
        },
        "metrics": {
            "fit_rms": "RMS full-vector error on free top nodes, matching the saved legacy diagnostic",
            "fit_rms_all_top": "RMS full-vector error on the full top chain, including fixed corners",
            "slope_rms": "reference-x-weighted RMS of the piecewise-linear slope error",
            "curvature_error": "reference-x-weighted finite difference of the target-relative slope error",
            "min_J": "recomputed triangle area ratio from the final mesh",
        },
        "limitations": [
            "Gradient-only and L2 objective values have different units and are not compared directly.",
            "All rows are saved final recorded states; finite optimization histories do not establish convergence.",
            "The curvature metric is a reference-coordinate second-derivative diagnostic, not geometric curvature.",
        ],
        "rows": rows,
        "relative_changes": comparisons,
        "figures": figures,
    }
    (output / "summary.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    )
    cherries.log_metrics(
        {
            "cases": len(rows),
            "gradient_cases": len(gradient),
            "l2_cases": len(l2),
            "figures": sum(bool(files) for files in figures.values()),
        }
    )
    LOG.info("Wrote %d endpoint rows to %s", len(rows), output)


if __name__ == "__main__":
    cherries.main(main)
