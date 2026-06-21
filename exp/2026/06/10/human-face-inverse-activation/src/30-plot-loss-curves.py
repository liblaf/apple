from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pydantic_settings as ps

from liblaf import cherries

logger = logging.getLogger(__name__)


class PlotConfig(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    comparison_summary: Path = cherries.input("20-inverse-summary.json")
    output_dir: Path = Path("figs/30-loss-curves")


def read_history_loss(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h5:
        steps = np.asarray(h5["VTKHDF/FieldData/inverse_step"], dtype=np.float64)
        losses = np.asarray(h5["VTKHDF/FieldData/inverse_loss"], dtype=np.float64)
    return steps, losses


def plot_loss(case: dict[str, Any], history_path: Path, output_dir: Path) -> Path:
    steps, losses = read_history_loss(history_path)
    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    ax.plot(steps, losses, marker="o", linewidth=1.4, markersize=2.5)
    best_step = float(case["best/step"])
    best_loss = float(case["best/loss"])
    ax.scatter([best_step], [best_loss], color="tab:red", zorder=3, label="best")
    ax.set_yscale("log")
    ax.set_xlabel("inverse step")
    ax.set_ylabel("point-to-point L2 loss")
    ax.set_title(str(case["case"]))
    ax.grid(visible=True, which="both", linewidth=0.4, alpha=0.35)
    ax.legend()
    output = output_dir / f"{case['case']}-log-loss.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_comparison(cases: list[dict[str, Any]], root: Path, output_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    for case in cases:
        history_path = root / str(case["history/path"])
        steps, losses = read_history_loss(history_path)
        ax.plot(
            steps,
            losses,
            marker="o",
            linewidth=1.3,
            markersize=2.3,
            label=case["target/name"],
        )
    ax.set_yscale("log")
    ax.set_xlabel("inverse step")
    ax.set_ylabel("point-to-point L2 loss")
    ax.set_title("Human face inverse activation")
    ax.grid(visible=True, which="both", linewidth=0.4, alpha=0.35)
    ax.legend()
    output = output_dir / "loss-comparison-log-y.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def main(cfg: PlotConfig) -> None:
    summary = json.loads(cfg.comparison_summary.read_text(encoding="utf-8"))
    cases = list(summary.get("cases", []))
    if not cases:
        msg = f"no cases in {cfg.comparison_summary}"
        raise ValueError(msg)
    root = cfg.comparison_summary.parent
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for case in cases:
        history_path = root / str(case["history/path"])
        if not history_path.exists():
            logger.warning(
                "Missing history file for %s: %s", case["case"], history_path
            )
            continue
        outputs.append(plot_loss(case, history_path, cfg.output_dir))
    if len(cases) > 1:
        outputs.append(plot_comparison(cases, root, cfg.output_dir))
    for output in outputs:
        cherries.log_output(output)
        logger.info("Wrote %s", output)
    if not outputs:
        msg = "no loss plots were written"
        raise FileNotFoundError(msg)


if __name__ == "__main__":
    cherries.main(main)
