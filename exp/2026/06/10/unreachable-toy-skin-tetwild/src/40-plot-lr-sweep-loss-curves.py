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


class PlotLrSweepConfig(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    input_dir: Path = Path("data/20-stretch-lr001")
    output_dir: Path = Path("figs/20-stretch-lr001")


def load_history(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h5:
        steps = np.asarray(h5["VTKHDF/FieldData/inverse_step"], dtype=np.int64)
        losses = np.asarray(h5["VTKHDF/FieldData/inverse_loss"], dtype=np.float64)
    return steps, losses


def load_cases(input_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    loaded: list[tuple[Path, dict[str, Any]]] = []
    summary_paths = sorted(input_dir.glob("**/20-toy-tetwild-*-summary.json"))
    if not summary_paths and (input_dir / "summary.json").exists():
        summary_paths = [input_dir / "summary.json"]
    for summary_path in summary_paths:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        cases = list(summary.get("cases", [summary]))
        loaded.extend((summary_path, case) for case in cases)
    if not loaded:
        msg = f"no inverse summaries found under {input_dir}"
        raise FileNotFoundError(msg)
    return sorted(loaded, key=lambda item: case_sort_key(item[1]))


def case_sort_key(case: dict[str, Any]) -> tuple[int, float, str]:
    if not bool(case.get("skin/energy_enabled", False)):
        rank = 0
    elif bool(case.get("skin/prestrain_enabled", False)):
        rank = 2
    else:
        rank = 1
    return rank, float(case["inverse/lr"]), str(case["case"])


def case_label(case: dict[str, Any]) -> str:
    if not bool(case.get("skin/energy_enabled", False)):
        label = "baseline"
    elif bool(case.get("skin/prestrain_enabled", False)):
        label = "skin + prestrain"
    else:
        label = "skin"
    return f"{label}, lr={float(case['inverse/lr']):g}"


def history_path_for_case(summary_path: Path, case: dict[str, Any]) -> Path:
    recorded = case.get("history/path")
    if isinstance(recorded, str) and recorded:
        path = Path(recorded)
        if path.exists():
            return path
        candidate = summary_path.parent / path.name
        if candidate.exists():
            return candidate
    return summary_path.parent / f"{case['case']}-steps.vtkhdf"


def case_title(case: dict[str, Any]) -> str:
    mode = str(case.get("mode", "unknown")).title()
    target_y = float(case.get("target_y", 0.0))
    tetwild_lr = float(case.get("tetwild/lr", 0.0))
    return f"{mode} target y={target_y:g}, TetWild lr={tetwild_lr:g}"


def plot_overlay(
    output_path: Path,
    rows: list[tuple[dict[str, Any], np.ndarray, np.ndarray]],
) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0.15, 0.85, len(rows)))
    for color, (case, steps, losses) in zip(colors, rows, strict=True):
        ax.plot(steps, losses, linewidth=2.0, color=color, label=case_label(case))
        best_step = int(case["best/step"])
        best_loss = float(case["best/loss"])
        ax.scatter([best_step], [best_loss], color=color, edgecolor="black", zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel("Inverse step")
    ax.set_ylabel("Loss")
    ax.set_title(case_title(rows[0][0]))
    ax.grid(visible=True, which="both", alpha=0.25)
    ax.legend(title="Case")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


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
    ax.set_yscale("log")
    ax.set_xlabel("Inverse step")
    ax.set_ylabel("Loss")
    ax.set_title(f"{case_title(case)} ({case_label(case)})")
    ax.grid(visible=True, which="both", alpha=0.25)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main(cfg: PlotLrSweepConfig) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[dict[str, Any], np.ndarray, np.ndarray]] = []
    output_paths: list[Path] = []
    for summary_path, case in load_cases(cfg.input_dir):
        case_name = str(case["case"])
        history_path = history_path_for_case(summary_path, case)
        cherries.log_input(summary_path)
        cherries.log_input(history_path)
        steps, losses = load_history(history_path)
        rows.append((case, steps, losses))

        output_path = cfg.output_dir / f"{case_name}-log-loss-vs-step.png"
        plot_single_case(output_path, case, steps, losses)
        output_paths.append(output_path)

    overlay_path = cfg.output_dir / "loss-vs-step-log.png"
    plot_overlay(overlay_path, rows)
    output_paths.insert(0, overlay_path)

    for output_path in output_paths:
        cherries.log_output(output_path)
    logger.info("Wrote %d LR sweep plot(s) under %s", len(output_paths), cfg.output_dir)


if __name__ == "__main__":
    cherries.main(main)
