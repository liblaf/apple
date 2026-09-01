"""Render all selected 2-D final pork shapes at one honest shared scale."""

from __future__ import annotations

# ruff: noqa: C901, EM101, EM102, RUF001, TRY003
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import pydantic_settings as ps
import pyvista as pv
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image

from liblaf import cherries

mpl.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
GROUP_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = GROUP_DIR / "data"
FAT_COLOR = "#E8C9A1"
MUSCLE_COLOR = "#B85C5C"
TARGET_COLOR = "#C2185B"
TOP_COLOR = "#17212B"


@dataclass(frozen=True)
class Case:
    name: str
    title: str
    source_dir: Path

    @property
    def final_path(self) -> Path:
        return self.source_dir / "final.vtu"

    @property
    def summary_path(self) -> Path:
        return self.source_dir / "summary.json"


CASES = (
    Case(
        "baseline",
        "Baseline · Stable NH · L2 · 100×10 · h=0.050",
        DATA_DIR / "10-pork-2d/baseline",
    ),
    Case("height-low", "Height 0.025", DATA_DIR / "10-pork-2d/height-low"),
    Case("height-high", "Height 0.100", DATA_DIR / "10-pork-2d/height-high"),
    Case("loss-l1", "Loss L1", DATA_DIR / "10-pork-2d/loss-l1"),
    Case(
        "loss-linf",
        "Loss L∞",
        DATA_DIR / "10-pork-2d/loss-linf",
    ),
    Case("mesh-medium", "Mesh 50×5", DATA_DIR / "10-pork-2d/mesh-medium"),
    Case("mesh-dense", "Mesh 200×20", DATA_DIR / "10-pork-2d/mesh-dense"),
    Case(
        "energy-linear",
        "Linear elasticity",
        DATA_DIR / "10-pork-2d/energy-linear",
    ),
)
DOWN_CASE = Case(
    "height-down",
    "Target down · Stable NH · L2 · 100×10 · h=-0.050",
    DATA_DIR / "60-pork-2d-target-down/height-down",
)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output_dir: Path = cherries.output("50-2d-final-shapes", mkdir=True)
    dpi: int = 220


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load(case: Case) -> dict[str, Any]:
    if not case.final_path.is_file() or not case.summary_path.is_file():
        raise FileNotFoundError(f"missing final state for {case.name}")
    cherries.log_input(case.final_path)
    cherries.log_input(case.summary_path)
    summary = read_json(case.summary_path)
    trace_path = case.source_dir / "trace.csv"
    series_path = case.source_dir / "history.vtu.series"
    if not trace_path.is_file() or not series_path.is_file():
        raise FileNotFoundError(f"missing trace/series for {case.name}")
    trace_count = len(trace_path.read_text(encoding="utf-8").splitlines()) - 1
    series = read_json(series_path).get("files")
    if not isinstance(series, list) or len(series) != trace_count:
        raise ValueError(f"trace/series count mismatch: {case.name}")
    if [entry.get("time") for entry in series] != list(range(trace_count)):
        raise ValueError(f"non-exact series steps: {case.name}")
    if any(
        not isinstance(entry, dict)
        or not (case.source_dir / str(entry.get("name"))).is_file()
        for entry in series
    ):
        raise FileNotFoundError(f"missing series frame: {case.name}")
    inverse = summary.get("inverse", {})
    if not isinstance(inverse, dict):
        inverse = {}
    evaluations = summary.get("evaluations", inverse.get("evaluations"))
    if evaluations is not None and int(evaluations) != trace_count:
        raise ValueError(f"evaluation/trace count mismatch: {case.name}")
    grid = pv.read(case.final_path)
    if set(grid.point_data) < {"Displacement", "TargetDisplacement"}:
        raise KeyError(f"missing displacement arrays: {case.final_path}")
    if "MuscleMask" not in grid.cell_data:
        raise KeyError(f"missing MuscleMask: {case.final_path}")
    cells = np.asarray(grid.cells).reshape(grid.n_cells, 4)
    if not np.all(cells[:, 0] == 3):
        raise ValueError(f"expected only triangles: {case.final_path}")
    reference = np.asarray(grid.points[:, :2], dtype=float)
    deformed = reference + np.asarray(grid.point_data["Displacement"][:, :2])
    target = reference + np.asarray(grid.point_data["TargetDisplacement"][:, :2])
    top = np.isclose(reference[:, 1], 0.1, atol=1.0e-12)
    order = np.argsort(reference[top, 0])
    final = summary.get("final")
    if not isinstance(final, dict):
        raise TypeError(f"missing final metrics: {case.summary_path}")
    return {
        "case": case,
        "grid": grid,
        "triangles": cells[:, 1:],
        "muscle": np.asarray(grid.cell_data["MuscleMask"], dtype=bool),
        "deformed": deformed,
        "top_deformed": deformed[top][order],
        "top_target": target[top][order],
        "summary": summary,
        "metrics": final,
    }


def metric_text(item: dict[str, Any]) -> str:
    metrics, summary = item["metrics"], item["summary"]
    inversion = summary.get("first_inversion_step")
    inversion_text = "none" if inversion is None else str(inversion)
    stationarity_receipt = summary.get("stationarity", {})
    if not isinstance(stationarity_receipt, dict):
        stationarity_receipt = {}
    convergence = summary.get("inverse", {}).get("convergence", {})
    if not isinstance(convergence, dict):
        convergence = {}
    stationarity = stationarity_receipt.get(
        "passed",
        convergence.get(
            "physical_stationarity_gate",
            convergence.get("practical_stationarity_gate"),
        ),
    )
    return (
        f"target RMS {float(metrics['top_target_rms']):.3e}  ·  "
        f"high-pass {float(metrics['top_highpass_rms']):.3e}  ·  "
        f"first inversion {inversion_text}  ·  stationarity {stationarity}"
    )


def draw_case(
    axis: Any,
    item: dict[str, Any],
    *,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    panel: str | None,
) -> None:
    triangles = item["triangles"]
    deformed = item["deformed"]
    facecolors = np.where(item["muscle"], MUSCLE_COLOR, FAT_COLOR)
    linewidth = min(0.32, max(0.045, 180.0 / len(triangles)))
    collection = PolyCollection(
        deformed[triangles],
        facecolors=facecolors,
        edgecolors="#29323A",
        linewidths=linewidth,
        alpha=0.88,
        rasterized=True,
    )
    axis.add_collection(collection)
    axis.plot(
        item["top_target"][:, 0],
        item["top_target"][:, 1],
        color=TARGET_COLOR,
        linestyle=(0, (5, 3)),
        linewidth=1.8,
        zorder=5,
    )
    axis.plot(
        item["top_deformed"][:, 0],
        item["top_deformed"][:, 1],
        color=TOP_COLOR,
        linewidth=1.35,
        zorder=6,
    )
    title = item["case"].title
    if panel is not None:
        title = f"{panel}  {title}"
    axis.set_title(title, loc="left", fontsize=11, fontweight="semibold", pad=7)
    axis.text(
        0.015,
        0.965,
        metric_text(item),
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=7.8,
        color="#263238",
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 2.0},
        zorder=10,
    )
    axis.set_xlim(x_limits)
    axis.set_ylim(y_limits)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xticks((0.0, 0.5, 1.0))
    axis.set_yticks((0.0, 0.1, 0.2))
    axis.grid(color="#CFD8DC", linewidth=0.45, alpha=0.75)
    axis.set_axisbelow(True)
    for spine in axis.spines.values():
        spine.set_color("#90A4AE")
        spine.set_linewidth(0.6)


def legend_handles() -> list[Any]:
    return [
        Patch(facecolor=FAT_COLOR, edgecolor="#29323A", label="fat"),
        Patch(facecolor=MUSCLE_COLOR, edgecolor="#29323A", label="muscle"),
        Line2D([], [], color=TOP_COLOR, linewidth=1.5, label="final top"),
        Line2D(
            [],
            [],
            color=TARGET_COLOR,
            linewidth=1.8,
            linestyle=(0, (5, 3)),
            label="target top",
        ),
    ]


def save_comparison(
    items: list[dict[str, Any]],
    output: Path,
    *,
    dpi: int,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(24, 5.0), sharex=True, sharey=True)
    for index, (axis, item) in enumerate(zip(axes.flat, items, strict=True)):
        draw_case(
            axis,
            item,
            x_limits=x_limits,
            y_limits=y_limits,
            panel=chr(ord("A") + index),
        )
        if index % 4 == 0:
            axis.set_ylabel("y")
        if index >= 4:
            axis.set_xlabel("x")
    figure.suptitle(
        "All eight 2-D OFAT final shapes",
        fontsize=18,
        fontweight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.947,
        "Final optimization states · shared physical axes · no vertical exaggeration",
        ha="center",
        va="top",
        fontsize=10,
        color="#455A64",
    )
    figure.legend(
        handles=legend_handles(),
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    figure.subplots_adjust(
        left=0.045, right=0.99, bottom=0.17, top=0.82, wspace=0.08, hspace=0.38
    )
    figure.savefig(output, dpi=dpi, facecolor="white")
    plt.close(figure)


def save_standalone(
    item: dict[str, Any],
    output: Path,
    *,
    dpi: int,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> None:
    figure, axis = plt.subplots(figsize=(12, 3.8))
    draw_case(
        axis,
        item,
        x_limits=x_limits,
        y_limits=y_limits,
        panel=None,
    )
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    figure.legend(
        handles=legend_handles(),
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.015),
    )
    figure.subplots_adjust(left=0.065, right=0.985, bottom=0.23, top=0.86)
    figure.savefig(output, dpi=dpi, facecolor="white")
    plt.close(figure)


def save_direction_comparison(
    items: list[dict[str, Any]],
    output: Path,
    *,
    dpi: int,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> None:
    if len(items) != 2:
        raise ValueError("target-direction comparison requires exactly two cases")
    figure, axes = plt.subplots(1, 2, figsize=(18, 3.8), sharex=True, sharey=True)
    for index, (axis, item) in enumerate(zip(axes, items, strict=True)):
        draw_case(
            axis,
            item,
            x_limits=x_limits,
            y_limits=y_limits,
            panel=chr(ord("A") + index),
        )
        axis.set_xlabel("x")
        if index == 0:
            axis.set_ylabel("y")
    figure.suptitle(
        "Matched target direction: h=+0.05 versus h=-0.05",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    figure.text(
        0.5,
        0.915,
        "Same model, initialization, mesh, loss, and hybrid inverse protocol · "
        "literal final states · no vertical exaggeration",
        ha="center",
        va="top",
        fontsize=9,
        color="#455A64",
    )
    figure.legend(
        handles=legend_handles(),
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.015),
    )
    figure.subplots_adjust(left=0.055, right=0.99, bottom=0.24, top=0.78, wspace=0.08)
    figure.savefig(output, dpi=dpi, facecolor="white")
    plt.close(figure)


def image_receipt(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        width, height = image.size
        mode = image.mode
    if width < 2000 or height < 700:
        raise ValueError(f"unexpectedly small visualization: {path} ({width}x{height})")
    try:
        receipt_path = str(path.relative_to(GROUP_DIR))
    except ValueError:
        receipt_path = str(path)
    return {
        "path": receipt_path,
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
        "pixels": [width, height],
        "mode": mode,
    }


def direction_case_receipt(item: dict[str, Any]) -> dict[str, Any]:
    case, summary, metrics = item["case"], item["summary"], item["metrics"]
    center = int(np.argmin(np.abs(item["top_target"][:, 0] - 0.5)))
    final_center_y = float(item["top_deformed"][center, 1])
    target_center_y = float(item["top_target"][center, 1])
    requested = float(summary["height"])
    achieved = final_center_y - 0.1
    inverse = summary.get("inverse", {})
    if not isinstance(inverse, dict):
        inverse = {}
    convergence = inverse.get("convergence", {})
    if not isinstance(convergence, dict):
        convergence = {}
    stationarity = summary.get("stationarity", {})
    if not isinstance(stationarity, dict):
        stationarity = {}
    failures = inverse.get("failures", {})
    if not isinstance(failures, dict):
        failures = {}
    refinement = summary.get("refinement", inverse.get("refinement", {}))
    if not isinstance(refinement, dict):
        refinement = {}
    return {
        "name": case.name,
        "title": case.title,
        "source": str(case.final_path.relative_to(GROUP_DIR)),
        "source_sha256": sha256(case.final_path),
        "points": int(item["grid"].n_points),
        "triangles": int(item["grid"].n_cells),
        "requested_center_displacement_y": requested,
        "achieved_center_displacement_y": achieved,
        "achieved_requested_fraction": achieved / requested,
        "final_center_y": final_center_y,
        "target_center_y": target_center_y,
        "final_target_rms": float(metrics["top_target_rms"]),
        "final_highpass_rms": float(metrics["top_highpass_rms"]),
        "final_curvature_rms": float(metrics["top_curvature_rms"]),
        "final_activation_rms": float(metrics["activation_rms"]),
        "final_activation_jump_rms": float(metrics["activation_neighbor_jump_rms"]),
        "first_inversion_step": summary.get("first_inversion_step"),
        "trajectory_min_det_f": float(summary["minimum_det_f"]),
        "trajectory_min_det_g": float(summary["minimum_det_g"]),
        "trajectory_min_det_ainv": float(summary["minimum_det_ainv"]),
        "final_min_det_f": float(metrics["min_det_f"]),
        "final_min_det_g": float(metrics["min_det_g"]),
        "final_min_det_ainv": float(metrics["min_det_ainv"]),
        "forward_failure_count": int(summary["forward_failure_count"]),
        "inverse_evaluation_failure_count": int(
            summary["inverse_evaluation_failure_count"]
        ),
        "tail_relative_range": float(
            summary["tail_convergence"]["objective_relative_range"]
        ),
        "tail_gate_1pct": bool(
            summary.get("tail_convergence", {}).get(
                "inverse_converged_1pct_tail_gate", False
            )
        ),
        "physical_stationarity_gate": stationarity.get(
            "passed",
            convergence.get(
                "physical_stationarity_gate",
                convergence.get("practical_stationarity_gate"),
            ),
        ),
        "refinement_trial_forward_failure_count": refinement.get(
            "trial_forward_failures", failures.get("refinement_trial_forward")
        ),
    }


def main(cfg: Config) -> None:
    if cfg.output_dir.exists() and any(cfg.output_dir.iterdir()):
        raise FileExistsError(f"refusing nonempty output directory: {cfg.output_dir}")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    items = [load(case) for case in CASES]
    down_item = load(DOWN_CASE)
    all_points = np.concatenate(
        [
            np.concatenate((item["deformed"], item["top_target"]), axis=0)
            for item in items
        ],
        axis=0,
    )
    x_limits = (-0.02, 1.02)
    y_span = float(np.ptp(all_points[:, 1]))
    y_limits = (
        min(-0.005, float(all_points[:, 1].min()) - 0.03 * y_span),
        float(all_points[:, 1].max()) + 0.05 * y_span,
    )
    comparison = cfg.output_dir / "all-2d-final-shapes.png"
    save_comparison(
        items,
        comparison,
        dpi=cfg.dpi,
        x_limits=x_limits,
        y_limits=y_limits,
    )
    cases_dir = cfg.output_dir / "cases"
    cases_dir.mkdir()
    standalone = []
    case_receipts = []
    for item in items:
        case = item["case"]
        output = cases_dir / f"{case.name}.png"
        save_standalone(
            item,
            output,
            dpi=cfg.dpi,
            x_limits=x_limits,
            y_limits=y_limits,
        )
        standalone.append(image_receipt(output))
        case_receipts.append(
            {
                "name": case.name,
                "title": case.title,
                "source": str(case.final_path.relative_to(GROUP_DIR)),
                "source_sha256": sha256(case.final_path),
                "points": int(item["grid"].n_points),
                "triangles": int(item["grid"].n_cells),
                "final_target_rms": float(item["metrics"]["top_target_rms"]),
                "final_highpass_rms": float(item["metrics"]["top_highpass_rms"]),
                "first_inversion_step": item["summary"].get("first_inversion_step"),
            }
        )
    direction_dir = cfg.output_dir / "target-direction"
    direction_dir.mkdir()
    direction_items = [items[0], down_item]
    direction_points = np.concatenate(
        [
            np.concatenate((item["deformed"], item["top_target"]), axis=0)
            for item in direction_items
        ],
        axis=0,
    )
    direction_y_span = float(np.ptp(direction_points[:, 1]))
    direction_y_limits = (
        min(-0.005, float(direction_points[:, 1].min()) - 0.03 * direction_y_span),
        float(direction_points[:, 1].max()) + 0.05 * direction_y_span,
    )
    direction_comparison = direction_dir / "target-up-vs-down.png"
    save_direction_comparison(
        direction_items,
        direction_comparison,
        dpi=cfg.dpi,
        x_limits=x_limits,
        y_limits=direction_y_limits,
    )
    down_standalone = direction_dir / "height-down.png"
    save_standalone(
        down_item,
        down_standalone,
        dpi=cfg.dpi,
        x_limits=x_limits,
        y_limits=direction_y_limits,
    )
    up_direction_receipt = direction_case_receipt(direction_items[0])
    down_direction_receipt = direction_case_receipt(direction_items[1])
    receipt = {
        "status": "ok",
        "selected_state": "final.vtu",
        "case_count": len(items),
        "shared_physical_axes": True,
        "vertical_exaggeration": 1.0,
        "x_limits": list(x_limits),
        "y_limits": list(y_limits),
        "comparison": image_receipt(comparison),
        "standalone": standalone,
        "cases": case_receipts,
        "target_direction": {
            "comparison": image_receipt(direction_comparison),
            "down_standalone": image_receipt(down_standalone),
            "shared_physical_axes": True,
            "vertical_exaggeration": 1.0,
            "x_limits": list(x_limits),
            "y_limits": list(direction_y_limits),
            "up_case": up_direction_receipt,
            "down_case": down_direction_receipt,
            "down_over_up_ratios": {
                key: down_direction_receipt[key] / up_direction_receipt[key]
                for key in (
                    "final_target_rms",
                    "final_highpass_rms",
                    "final_curvature_rms",
                    "final_activation_rms",
                    "final_activation_jump_rms",
                )
            },
        },
    }
    receipt_path = cfg.output_dir / "receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    cherries.log_metrics(
        {
            "visualization/cases": len(items),
            "visualization/vertical_exaggeration": 1.0,
        }
    )
    logger.info("Wrote %s and %d standalone figures", comparison, len(standalone))


if __name__ == "__main__":
    cherries.main(main)
