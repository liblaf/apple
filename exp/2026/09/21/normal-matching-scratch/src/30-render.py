"""Render saved neutral-start L2 versus target-normal factorial histories."""

# ruff: noqa: EM101, EM102, TRY003, TRY004

from __future__ import annotations

import csv
import json
import logging
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import pydantic_settings as ps
from experiment import Profile
from matplotlib.lines import Line2D

from liblaf import cherries

mpl.use("Agg")
import matplotlib.pyplot as plt

GROUP = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)
MODES = ("unconstrained", "contraction_only", "learned_direction", "x_contraction")
SMOOTHS = ("smooth-off", "smooth-on")
VARIANTS = tuple(f"{smooth}-{loss}" for smooth in SMOOTHS for loss in ("l2", "normal"))
MODE_LABELS = {
    "unconstrained": "Unrestricted symmetric",
    "contraction_only": "PSD contraction-only",
    "learned_direction": "Learned-axis rank-one",
    "x_contraction": "Fixed-x contraction",
}
COLORS = {"l2": "#4d4a46", "normal": "#c05621"}


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    comparison_dir: Path = GROUP / "data/10-comparison"
    verification_dir: Path = GROUP / "data/20-verification"
    output_dir: Path = GROUP / "data/30-figures"


@dataclass(frozen=True)
class Cell:
    name: str
    mode: str
    shared_step: int
    unstable: bool
    variant_metrics: dict[str, dict[str, Any]]
    endpoints: dict[str, dict[str, Any]]


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def _load_cells(selection: dict[str, Any]) -> tuple[Cell, ...]:
    raw_cells = selection.get("cells")
    if not isinstance(raw_cells, list) or len(raw_cells) != 4:
        raise ValueError("selection must have four activation-mode cells")
    cells = []
    for raw in raw_cells:
        required = {"mode", "height", "shared_step", "variants", "endpoints"}
        if not isinstance(raw, dict) or required - set(raw):
            raise ValueError("invalid verification selection cell")
        metrics = raw["variants"]
        if not isinstance(metrics, dict) or set(metrics) != set(VARIANTS):
            raise ValueError(f"unexpected factorial variant set in {raw['mode']}")
        endpoints = raw["endpoints"]
        if not isinstance(endpoints, dict) or set(endpoints) != set(VARIANTS):
            raise ValueError(f"unexpected endpoint set in {raw['mode']}")
        for metric in metrics.values():
            hessian = metric["physical_hessian"]
            assert "smallest_algebraic_eigenvalue" in hessian
        unstable = any(
            float(metric["physical_hessian"]["smallest_algebraic_eigenvalue"]) < 0
            for metric in metrics.values()
        )
        cells.append(
            Cell(
                name=str(raw["mode"]),
                mode=str(raw["mode"]),
                shared_step=int(raw["shared_step"]),
                unstable=unstable,
                variant_metrics=metrics,
                endpoints=endpoints,
            )
        )
    if {cell.mode for cell in cells} != set(MODES):
        raise ValueError("selection does not cover all activation modes")
    return tuple(cells)


def _history(
    folder: Path, step: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with np.load(folder / "history.npz", allow_pickle=False) as saved:
        required = {"points", "triangles", "u", "steps", "top_all"}
        if required - set(saved.files):
            raise ValueError(f"incomplete history: {folder}")
        index = np.flatnonzero(np.asarray(saved["steps"]) == step)
        if len(index) != 1:
            raise ValueError(f"shared state {step} unavailable: {folder}")
        points = np.asarray(saved["points"], dtype=float)
        triangles = np.asarray(saved["triangles"], dtype=np.int64)
        u = np.asarray(saved["u"], dtype=float)[index[0]]
        top = np.asarray(saved["top_all"], dtype=np.int64)
    return points, triangles, u, top


def _trace(folder: Path) -> dict[str, np.ndarray]:
    with (folder / "trace.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or {"step", "objective", "projected_gradient_inf"} - set(rows[0]):
        raise ValueError(f"incomplete trace: {folder}")
    return {key: np.asarray([float(row[key]) for row in rows]) for key in rows[0]}


def _target(points: np.ndarray, top: np.ndarray) -> np.ndarray:
    reference = points[top]
    x = reference[:, 0]
    return reference + np.column_stack((np.zeros_like(x), 0.8 * x * (1 - x)))


def _axes(axis: plt.Axes) -> None:
    axis.set_aspect("equal", adjustable="box")
    axis.set(xlim=(-0.03, 1.03), ylim=(-0.15, 0.34), xlabel="deformed x", ylabel="y")
    axis.grid(alpha=0.2)


def _warning(axis: plt.Axes, cell: Cell) -> None:
    if cell.unstable:
        axis.text(
            0.98,
            0.97,
            "UNSTABLE EQUILIBRIUM\nnegative physical Hessian",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=7,
            color="#9c2c2c",
            weight="bold",
            bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "#9c2c2c"},
        )


def _condition_note(cell: Cell, variant: str) -> str:
    metric, endpoint = cell.variant_metrics[variant], cell.endpoints[variant]
    hessian = metric["physical_hessian"]["smallest_algebraic_eigenvalue"]
    if float(hessian) < 0:
        return "UNSTABLE\nnegative Hessian"
    failure = endpoint.get("failure")
    backtracking = int(metric["top_edge_backtracking_count"])
    notes = []
    if backtracking:
        notes.append(f"{backtracking} top-edge backtracks")
    if failure:
        notes.append(f"endpoint failed; last {failure['last_valid_step']}")
    return "\n".join(notes)


def _profiles(source: Path, cells: tuple[Cell, ...], output: Path) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(13.5, 11), sharex=True, sharey=True)
    for row, mode in enumerate(MODES):
        cell = next(item for item in cells if item.mode == mode)
        for col, smooth in enumerate(SMOOTHS):
            axis = axes[row, col]
            base = source / cell.name
            points, _, neutral, top = _history(base / f"{smooth}-l2", 0)
            target = _target(points, top)
            axis.plot(target[:, 0], target[:, 1], "k--", lw=1.6)
            axis.plot(
                points[top, 0],
                points[top, 1] + neutral[top, 1],
                color="#999999",
                ls=":",
                lw=1.2,
            )
            for loss in ("l2", "normal"):
                _, _, u, _ = _history(base / f"{smooth}-{loss}", cell.shared_step)
                axis.plot(
                    points[top, 0] + u[top, 0],
                    points[top, 1] + u[top, 1],
                    color=COLORS[loss],
                    lw=1.6,
                )
            axis.set_title(
                f"{MODE_LABELS[mode]} | {smooth.replace('-', ' ')} | common step {cell.shared_step}",
                fontsize=8.5,
            )
            _warning(axis, cell)
            _axes(axis)
    fig.suptitle(
        "Neutral-start profiles: L2-only versus L2 + target-normal matching",
        y=0.985,
        fontsize=14,
    )
    fig.legend(
        handles=[
            Line2D([], [], color="black", ls="--", label="Target top"),
            Line2D([], [], color="#999999", ls=":", label="Neutral"),
            Line2D([], [], color=COLORS["l2"], label="L2 only"),
            Line2D([], [], color=COLORS["normal"], label="L2 + normal β=0.05"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.958),
        ncols=4,
        frameon=False,
    )
    fig.text(
        0.5,
        0.012,
        "Every panel uses the same physical axes. Each comparison uses its mode's latest state shared by all four factorial runs.",
        ha="center",
        fontsize=8.5,
    )
    fig.subplots_adjust(
        left=0.06, right=0.985, top=0.89, bottom=0.07, wspace=0.18, hspace=0.42
    )
    fig.savefig(output / "shared-step-profiles.png", dpi=200)
    plt.close(fig)


def _meshes(source: Path, cells: tuple[Cell, ...], output: Path) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(13.5, 11), sharex=True, sharey=True)
    for row, mode in enumerate(MODES):
        cell = next(item for item in cells if item.mode == mode)
        for col, smooth in enumerate(SMOOTHS):
            axis = axes[row, col]
            base = source / cell.name
            points, triangles, _, top = _history(
                base / f"{smooth}-l2", cell.shared_step
            )
            target = _target(points, top)
            axis.plot(target[:, 0], target[:, 1], "k--", lw=1.4)
            for loss in ("l2", "normal"):
                _, _, u, _ = _history(base / f"{smooth}-{loss}", cell.shared_step)
                shape = points + u
                axis.triplot(
                    shape[:, 0],
                    shape[:, 1],
                    triangles,
                    color=COLORS[loss],
                    lw=0.25,
                    alpha=0.78,
                )
            axis.set_title(
                f"{MODE_LABELS[mode]} | {smooth.replace('-', ' ')}", fontsize=8.5
            )
            _warning(axis, cell)
            _axes(axis)
    fig.suptitle("Internal meshes at the common step", y=0.985, fontsize=14)
    fig.legend(
        handles=[
            Line2D([], [], color="black", ls="--", label="Target boundary only"),
            Line2D([], [], color=COLORS["l2"], label="L2-only mesh"),
            Line2D([], [], color=COLORS["normal"], label="L2 + normal mesh"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.958),
        ncols=3,
        frameon=False,
    )
    fig.text(
        0.5,
        0.012,
        "The target is a boundary condition; no interior target displacement field is fabricated. Equal physical axes in every panel.",
        ha="center",
        fontsize=8.5,
    )
    fig.subplots_adjust(
        left=0.06, right=0.985, top=0.89, bottom=0.07, wspace=0.18, hspace=0.42
    )
    fig.savefig(output / "shared-step-mesh-overlays.png", dpi=200)
    plt.close(fig)


def _factorial_meshes(source: Path, cells: tuple[Cell, ...], output: Path) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(16, 11), sharex=True, sharey=True)
    for row, mode in enumerate(MODES):
        cell = next(item for item in cells if item.mode == mode)
        for col, variant in enumerate(VARIANTS):
            axis = axes[row, col]
            points, triangles, u, top = _history(
                source / cell.name / variant, cell.shared_step
            )
            target = _target(points, top)
            shape = points + u
            axis.plot(target[:, 0], target[:, 1], "k--", lw=1.3)
            axis.triplot(
                shape[:, 0],
                shape[:, 1],
                triangles,
                color=COLORS["normal" if variant.endswith("normal") else "l2"],
                lw=0.25,
                alpha=0.82,
            )
            axis.set_title(
                f"{MODE_LABELS[mode]}\n{variant.replace('-', ' ')} | step {cell.shared_step}",
                fontsize=7.7,
            )
            if note := _condition_note(cell, variant):
                axis.text(
                    0.98,
                    0.96,
                    note,
                    transform=axis.transAxes,
                    ha="right",
                    va="top",
                    fontsize=6.5,
                    color="#9c2c2c",
                    weight="bold",
                    bbox={"facecolor": "white", "alpha": 0.86, "edgecolor": "#9c2c2c"},
                )
            _axes(axis)
    fig.suptitle(
        "All 16 neutral-start factorial conditions at their per-mode common steps",
        y=0.985,
        fontsize=14,
    )
    fig.legend(
        handles=[
            Line2D([], [], color="black", ls="--", label="Target boundary only"),
            Line2D([], [], color=COLORS["l2"], label="L2-only mesh"),
            Line2D([], [], color=COLORS["normal"], label="L2 + normal mesh"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.958),
        ncols=3,
        frameon=False,
    )
    fig.text(
        0.5,
        0.012,
        "Equal physical axes. Endpoint failures and all shared-step verification metrics are recorded separately; the target interior is intentionally not fabricated.",
        ha="center",
        fontsize=8,
    )
    fig.subplots_adjust(
        left=0.055, right=0.99, top=0.89, bottom=0.07, wspace=0.18, hspace=0.48
    )
    fig.savefig(output / "all-factorial-meshes.png", dpi=200)
    plt.close(fig)


def _curves(source: Path, cells: tuple[Cell, ...], output: Path) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(13.5, 11))
    for row, mode in enumerate(MODES):
        cell = next(item for item in cells if item.mode == mode)
        for col, smooth in enumerate(SMOOTHS):
            axis = axes[row, col]
            other = axis.twinx()
            for loss in ("l2", "normal"):
                trace = _trace(source / cell.name / f"{smooth}-{loss}")
                axis.plot(
                    trace["step"],
                    trace["objective"] / trace["objective"][0],
                    color=COLORS[loss],
                    lw=1.4,
                )
                other.plot(
                    trace["step"],
                    trace["projected_gradient_inf"]
                    / trace["projected_gradient_inf"][0],
                    color=COLORS[loss],
                    ls=(0, (1, 1)),
                    lw=1,
                )
            axis.axvline(cell.shared_step, color="#777", ls=":", lw=0.8)
            other.set_yscale("log")
            axis.set(
                title=f"{MODE_LABELS[mode]} | {smooth.replace('-', ' ')} | common step {cell.shared_step}",
                xlabel="Adam update",
                ylabel="Own objective / J(0)",
            )
            other.set_ylabel("Projected grad inf / initial")
            axis.grid(alpha=0.22)
            _warning(axis, cell)
    fig.suptitle("Objective and projected-gradient histories", y=0.985, fontsize=14)
    fig.legend(
        handles=[
            Line2D([], [], color=COLORS["l2"], label="L2-only objective"),
            Line2D([], [], color=COLORS["normal"], label="L2 + normal objective"),
            Line2D(
                [],
                [],
                color="#444",
                ls=(0, (1, 1)),
                label="Fine-dot: projected gradient, right axis",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.958),
        ncols=3,
        frameon=False,
        fontsize=8,
    )
    fig.text(
        0.5,
        0.012,
        "Each objective includes its own normal and smoothness terms, so own-objective heights are not a cross-variant fit ranking. Red labels mark negative physical-Hessian equilibria.",
        ha="center",
        fontsize=8,
    )
    fig.subplots_adjust(
        left=0.07, right=0.90, top=0.89, bottom=0.07, wspace=0.35, hspace=0.48
    )
    fig.savefig(
        output / "objective-and-projected-gradient-histories.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)


def _endpoints(source: Path, cells: tuple[Cell, ...], output: Path) -> None:
    rows = []
    for cell in cells:
        for variant in VARIANTS:
            summary = _json(source / cell.name / variant / "summary.json")
            rows.append(
                {
                    "cell": cell.name,
                    "variant": variant,
                    "shared_step": cell.shared_step,
                    "accepted_updates": summary["accepted_iterations"],
                    "failure": summary["failure"],
                    "optimizer_message": summary["optimizer_message"],
                    "shared_metrics": cell.variant_metrics[variant],
                }
            )
    (output / "endpoints-and-verification.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )


def main(cfg: Config) -> None:
    LOG.info(
        "Rendering saved neutral-start factorial histories; no inverse solves are run."
    )
    source, verification = (
        cherries.input(cfg.comparison_dir.resolve()),
        cherries.input(cfg.verification_dir.resolve()),
    )
    output = cherries.output(cfg.output_dir.resolve(), mkdir=True)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite: {output}")
    output.mkdir(parents=True, exist_ok=True)
    selection_path = verification / "comparison.json"
    cells = _load_cells(_json(selection_path))
    _profiles(source, cells, output)
    _meshes(source, cells, output)
    _factorial_meshes(source, cells, output)
    _curves(source, cells, output)
    _endpoints(source, cells, output)
    shutil.copy2(Path(__file__), output / "source-30-render.py")
    (output / "summary.json").write_text(
        json.dumps(
            {
                "purpose": "Read-only neutral-start factorial rendering.",
                "selection": str(selection_path),
                "equal_axes": {"x": [-0.03, 1.03], "y": [-0.15, 0.34]},
                "target": "boundary only; no interior target state",
                "python": sys.version,
            },
            indent=2,
        )
        + "\n"
    )
    for path in output.iterdir():
        cherries.log_output(path)


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
