from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import h5py
import matplotlib as mpl
import numpy as np
import pydantic_settings as ps

from liblaf import cherries

mpl.use("Agg")
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


class PlotLossConfig(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    comparison_summary: Path = Path("data/20-unreachable-toy-skin-tetwild-summary.json")
    output_dir: Path = Path("figs/30-loss-curves")


def load_history(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h5:
        steps = np.asarray(h5["VTKHDF/FieldData/inverse_step"], dtype=np.int64)
        losses = np.asarray(h5["VTKHDF/FieldData/inverse_loss"], dtype=np.float64)
    return steps, losses


def row_label(case: dict[str, Any]) -> str:
    loss = "L2 + residual Laplacian" if case["loss/residual_laplacian_enabled"] else "L2"
    prestrain = "skin prestrain" if case["skin/prestrain_enabled"] else "no skin prestrain"
    return f"{loss}\n{prestrain}"


def activation_label(case: dict[str, Any]) -> str:
    activation = str(case["activation/mode"])
    if activation == "per-tet-smooth":
        return "per-tet + smooth"
    return activation


def case_title(case: dict[str, Any]) -> str:
    return f"{row_label(case).replace(chr(10), ', ')}\n{activation_label(case)}"


def case_filename(case: dict[str, Any]) -> str:
    return f"{case['case']}-loss-vs-step.png"


def plot_single_case(
    output_path: Path,
    case: dict[str, Any],
    steps: np.ndarray,
    losses: np.ndarray,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.2), constrained_layout=True)
    ax.plot(steps, losses, color="#2a6f97", linewidth=2.0)
    best_step = int(case["best/step"])
    best_loss = float(case["best/loss"])
    ax.scatter([best_step], [best_loss], color="#c1121f", zorder=3)
    ax.axvline(best_step, color="#c1121f", alpha=0.25, linewidth=1.0)
    ax.set_title(case_title(case))
    ax.set_xlabel("Inverse step")
    ax.set_ylabel("Loss")
    ax.grid(visible=True, alpha=0.25)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_overview(
    output_path: Path,
    cases: list[dict[str, Any]],
    histories: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    rows = [
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ]
    activations = ["per-tet", "per-tet-smooth", "shared"]
    case_by_key = {
        (
            bool(case["loss/residual_laplacian_enabled"]),
            bool(case["skin/prestrain_enabled"]),
            str(case["activation/mode"]),
        ): case
        for case in cases
    }

    fig, axes = plt.subplots(
        nrows=len(rows),
        ncols=len(activations),
        figsize=(14.0, 10.0),
        sharex=False,
        sharey=False,
        constrained_layout=True,
    )
    fig.suptitle("Toy Squash Inverse Physics: Loss vs Step", fontsize=16)

    for row_idx, row_key in enumerate(rows):
        for col_idx, activation in enumerate(activations):
            ax = axes[row_idx, col_idx]
            case = case_by_key[(*row_key, activation)]
            steps, losses = histories[str(case["case"])]
            ax.plot(steps, losses, color="#2a6f97", linewidth=1.8)
            best_step = int(case["best/step"])
            best_loss = float(case["best/loss"])
            ax.scatter([best_step], [best_loss], color="#c1121f", s=18, zorder=3)
            ax.axvline(best_step, color="#c1121f", alpha=0.2, linewidth=1.0)
            ax.grid(visible=True, alpha=0.25)
            ax.set_title(activation_label(case), fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(f"{row_label(case)}\nLoss")
            if row_idx == len(rows) - 1:
                ax.set_xlabel("Inverse step")

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main(cfg: PlotLossConfig) -> None:
    summary = json.loads(cfg.comparison_summary.read_text(encoding="utf-8"))
    cases = list(summary["cases"])
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    histories: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    output_paths: list[Path] = []
    for case in cases:
        case_name = str(case["case"])
        history_path = cfg.comparison_summary.parent / f"{case_name}-steps.vtkhdf"
        cherries.log_input(history_path)
        steps, losses = load_history(history_path)
        histories[case_name] = (steps, losses)
        output_path = cfg.output_dir / case_filename(case)
        plot_single_case(output_path, case, steps, losses)
        output_paths.append(output_path)

    overview_path = cfg.output_dir / "loss-vs-step-all-cases.png"
    plot_overview(overview_path, cases, histories)
    output_paths.insert(0, overview_path)

    for output_path in output_paths:
        cherries.log_output(output_path)
    logger.info("Wrote %d loss-curve plot(s) under %s", len(output_paths), cfg.output_dir)


if __name__ == "__main__":
    cherries.main(main)
