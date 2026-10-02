# Copyright (c) 2026 liblaf
"""Shared, source-faithful lineage charts for coupled inverse reviews."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def combine_lineage_rows(
    rows: Sequence[dict[str, Any]], parent_rows: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return cumulative accepted rows while retaining same-step event records."""
    combined = list(parent_rows)
    if parent_rows:
        parent_last = int(parent_rows[-1]["iteration"])
        continuation = [
            {**row, "iteration": parent_last + int(row["iteration"])} for row in rows
        ]
        if (
            continuation
            and continuation[0]["iteration"] == parent_last
            and "equilibrium_refinement" not in continuation[0]
            and "objective_change" not in continuation[0]
            and "recovery_zero_update" not in continuation[0]
        ):
            continuation = continuation[1:]
        combined.extend(continuation)
    else:
        combined = list(rows)
    assert combined
    return combined


def objective_component_series(rows: Sequence[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Extract only recorded objective terms; unsupported historic terms stay NaN."""
    assert rows
    iterations = np.asarray([int(row["iteration"]) for row in rows])
    total = np.asarray([float(row["loss"]) for row in rows])
    positional = np.full(len(rows), np.nan)
    normal = np.full(len(rows), np.nan)
    smooth = np.full(len(rows), np.nan)
    roughness = np.full(len(rows), np.nan)
    legacy_position_only = np.zeros(len(rows), dtype=bool)
    for index, row in enumerate(rows):
        components = row.get("loss_components")
        if components is None:
            # Older L2-only rows record loss itself as the normalized position term.
            positional[index] = total[index]
            legacy_position_only[index] = True
            continue
        assert isinstance(components, dict)
        positional[index] = float(components["position_loss"])
        normal[index] = float(components["normal_contribution"])
        smooth[index] = float(components["regularizer_contribution"])
        roughness[index] = float(components["activation_smoothness"])
    return {
        "iteration": iterations,
        "total_objective": total,
        "normalized_position_l2": positional,
        "weighted_normal": normal,
        "weighted_smoothness": smooth,
        "raw_activation_roughness": roughness,
        "legacy_position_only": legacy_position_only,
    }


def break_at_zero_update(
    rows: Sequence[dict[str, Any]], values: np.ndarray
) -> np.ndarray:
    """Break a plotted line at a recovery event whose state identity is unproved."""
    result = np.asarray(values, dtype=float).copy()
    assert len(result) == len(rows)
    for index, row in enumerate(rows):
        if "recovery_zero_update" in row:
            result[index] = np.nan
    return result


def _event_markers(  # noqa: C901
    axes: Sequence[plt.Axes],
    series: dict[str, np.ndarray],
    rows: Sequence[dict[str, Any]],
) -> None:
    values = [
        series["total_objective"],
        series["normalized_position_l2"],
        series["weighted_normal"],
        series["weighted_smoothness"],
        series["raw_activation_roughness"],
    ]
    iteration = series["iteration"]
    for index, row in enumerate(rows):
        x = iteration[index]
        if "equilibrium_refinement" in row:
            for axis, value in zip(axes, values, strict=True):
                if np.isfinite(value[index]):
                    axis.scatter(x, value[index], color="#6a4c93", marker="D", zorder=4)
        if "recovery_zero_update" in row:
            for axis, value in zip(axes, values, strict=True):
                axis.axvline(x, color="#555", linestyle=":", linewidth=1, alpha=0.8)
                if np.isfinite(value[index]):
                    axis.scatter(x, value[index], color="#555", marker="s", zorder=5)
            axes[0].annotate(
                "Storage recovery\nzero-update boundary",
                (x, values[0][index]),
                xytext=(8, 18),
                textcoords="offset points",
                fontsize=8,
                arrowprops={"arrowstyle": "-", "color": "#555"},
            )
        if "objective_change" in row:
            for axis in axes:
                axis.axvline(x, color="#bc6c25", linestyle="--", linewidth=1, alpha=0.8)
            for axis, value in zip(axes[:2], values[:2], strict=True):
                if np.isfinite(value[index]):
                    axis.scatter(x, value[index], color="#bc6c25", marker="X", zorder=5)
            axes[0].annotate(
                "Objective changed\noptimizer moments retained",
                (x, values[0][index]),
                xytext=(10, 20),
                textcoords="offset points",
                fontsize=8,
                arrowprops={"arrowstyle": "-", "color": "#bc6c25"},
            )


def plot_objective_components(
    output: Path,
    rows: Sequence[dict[str, Any]],
    parent_rows: Sequence[dict[str, Any]],
) -> tuple[str, dict[str, int]]:
    """Plot complete lineage objective terms without fabricating legacy components."""
    from pathlib import Path

    combined = combine_lineage_rows(rows, parent_rows)
    series = objective_component_series(combined)
    figure, axes_grid = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    axes = list(axes_grid.ravel())
    plots = (
        ("total_objective", "Full recorded objective", "Objective", "#263238"),
        (
            "normalized_position_l2",
            "Normalized positional L2",
            "Normalized loss",
            "#c75f42",
        ),
        (
            "weighted_normal",
            "Weighted normal contribution",
            "Objective contribution",
            "#087d81",
        ),
        (
            "weighted_smoothness",
            "Weighted smoothness contribution",
            "Objective contribution",
            "#6a4c93",
        ),
        (
            "raw_activation_roughness",
            "Raw activation roughness",
            "Raw roughness (separate scale)",
            "#bc6c25",
        ),
    )
    for axis, (field, title, ylabel, color) in zip(axes[:5], plots, strict=True):
        axis.plot(
            series["iteration"],
            break_at_zero_update(combined, series[field]),
            color=color,
            marker="o",
        )
        axis.set(
            title=title, xlabel="Cumulative accepted optimizer iteration", ylabel=ylabel
        )
        axis.grid(alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    note_axis = axes[-1]
    note_axis.axis("off")
    note_axis.text(
        0.05,
        0.88,
        "Lineage note",
        transform=note_axis.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
    )
    note_axis.text(
        0.05,
        0.72,
        (
            "Gaps mean a historic L2-only row did not record that component; "
            "its recorded loss is used only for positional L2."
        ),
        transform=note_axis.transAxes,
        fontsize=9,
        va="top",
        wrap=True,
    )
    _event_markers(axes[:5], series, combined)
    name = "objective-components-curves.png"
    figure.savefig(Path(output) / name, dpi=180)
    plt.close(figure)
    return name, {
        "rows": len(combined),
        "legacy_position_only_rows": int(series["legacy_position_only"].sum()),
        "rows_with_normal_component": int(np.isfinite(series["weighted_normal"]).sum()),
        "rows_with_smoothness_component": int(
            np.isfinite(series["weighted_smoothness"]).sum()
        ),
        "rows_with_raw_roughness": int(
            np.isfinite(series["raw_activation_roughness"]).sum()
        ),
    }
