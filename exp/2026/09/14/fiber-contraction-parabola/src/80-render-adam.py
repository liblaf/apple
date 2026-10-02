"""Render Adam loss and deformation histories, retaining loss increases."""

import csv
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import pydantic_settings as ps
from liblaf.cherries import core, plugins, profiles

from liblaf import cherries

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection

GROUP = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)
MODES = ("x_contraction", "unconstrained")
LABELS = {"x_contraction": "x contraction", "unconstrained": "free symmetric B"}
COLORS = {"x_contraction": "#b24b35", "unconstrained": "#246fa6"}


class ProfileAdamFigures(profiles.Profile):
    def init(self):
        run = core.run
        run.plugins.register(plugins.Comet(run=run, disabled=False))
        run.plugins.register(plugins.Git(run=run, commit=False))
        run.plugins.register(plugins.Local(run=run))
        run.plugins.register(plugins.Logging(run=run))
        return run


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    input_dir: Path = Path("70-adam-comparison")
    output: Path = Path("80-adam-figures")
    frame_stride: int = 5
    fps: int = 20


def load_case(source: Path, name: str):
    folder = source / name
    with (folder / "trace.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    history = dict(np.load(folder / "history.npz", allow_pickle=False))
    assert len(rows) == len(history["u"])
    return {
        "name": name,
        "history": history,
        "rows": rows,
        "summary": json.loads((folder / "summary.json").read_text()),
    }


def draw_shape(
    axis: Axes,
    case: dict[str, Any],
    index: int,
    limits: tuple[float, float, float, float],
):
    h = case["history"]
    points = h["points"] + h["u"][index]
    axis.add_collection(
        PolyCollection(
            points[h["triangles"]],
            facecolors=np.where(
                h["muscle"][:, None],
                np.array([[0.73, 0.36, 0.31]]),
                np.array([[0.85, 0.70, 0.51]]),
            ),
            edgecolors="#5b4b40",
            linewidths=0.1,
        )
    )
    x = np.linspace(0, 1, 401)
    height = float(h["height"])
    axis.plot(x, 0.1 + 4 * height * x * (1 - x), "k--", lw=1.2, label="target")
    axis.set(xlim=limits[:2], ylim=limits[2:], xlabel="deformed x", ylabel="absolute y")
    axis.set_aspect("equal", adjustable="box")


def shape_limits(pair: list[dict[str, Any]], height: float):
    positions = np.concatenate(
        [
            (c["history"]["points"][None] + c["history"]["u"]).reshape(-1, 2)
            for c in pair
        ]
    )
    low = min(0.0, float(positions[:, 1].min()))
    high = max(0.1 + height, float(positions[:, 1].max()))
    return (
        min(-0.02, float(positions[:, 0].min()) - 0.02),
        max(1.02, float(positions[:, 0].max()) + 0.02),
        low - 0.01,
        high + 0.01,
    )


def render_pair(pair: list[dict[str, Any]], height: float, output: Path, cfg: Config):
    suffix = f"h{round(height * 1000):03d}"
    limits = shape_limits(pair, height)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), layout="constrained")
    for axis, case in zip(axes, pair, strict=True):
        draw_shape(axis, case, -1, limits)
        mode = str(case["history"]["mode"])
        last = int(case["rows"][-1]["step"])
        axis.set_title(f"{LABELS[mode]} | last solved Adam step {last}")
        axis.legend(frameon=False)
    fig.suptitle(f"Adam with physical J = det(F) | h = {height:g}")
    fig.savefig(output / f"adam-deformation-{suffix}.png", dpi=180)
    plt.close(fig)

    final = max(len(c["history"]["u"]) - 1 for c in pair)
    indices = sorted({*range(0, final + 1, cfg.frame_stride), final})
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), layout="constrained")
    writer = animation.FFMpegWriter(
        fps=cfg.fps,
        metadata={
            "title": f"Adam h={height:g}; saved steps, stride={cfg.frame_stride}"
        },
    )
    with writer.saving(fig, str(output / f"adam-evolution-{suffix}.mp4"), dpi=140):
        for index in indices:
            for axis, case in zip(axes, pair, strict=True):
                axis.clear()
                saved = min(index, len(case["history"]["u"]) - 1)
                draw_shape(axis, case, saved, limits)
                mode = str(case["history"]["mode"])
                status = "last state held" if saved != index else "Adam step"
                axis.set_title(f"{LABELS[mode]} | {status} {saved}")
            fig.suptitle(
                f"Adam h={height:g} | requested step {index} | every {cfg.frame_stride} saved steps"
            )
            writer.grab_frame()
    plt.close(fig)
    return {"height": height, "frame_indices": indices, "fps": cfg.fps}


def main(cfg: Config):
    source = GROUP / "data" / cfg.input_dir
    cases = [
        load_case(source, f"h{round(h * 1000):03d}-{mode}")
        for h in (0.05, 0.20)
        for mode in MODES
    ]
    output = cherries.output(cfg.output)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / Path(__file__).name)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained", sharey=True)
    records = []
    for case in cases:
        h = case["history"]
        mode, height = str(h["mode"]), float(h["height"])
        loss = np.array([float(r["objective_normalized"]) for r in case["rows"]])
        steps = np.array([int(r["step"]) for r in case["rows"]])
        delta = np.diff(loss)
        increase = np.r_[False, delta > np.maximum(1e-12, 1e-10 * np.abs(loss[:-1]))]
        axis = axes[0 if height == 0.05 else 1]
        axis.plot(steps, loss, color=COLORS[mode], lw=1.4, label=LABELS[mode])
        axis.scatter(
            steps[increase],
            loss[increase],
            color=COLORS[mode],
            s=10,
            marker="x",
            zorder=3,
        )
        axis.scatter(steps[-1], loss[-1], color=COLORS[mode], s=30, zorder=4)
        axis.set_title(f"Target height h = {height:g}")
        axis.set_xlabel("Adam updates (no outer line search)")
        axis.grid(alpha=0.2)
        records.append(
            {
                "case": case["name"],
                "last_solved_step": int(steps[-1]),
                "initial_loss": float(loss[0]),
                "final_loss": float(loss[-1]),
                "best_loss": float(loss.min()),
                "best_step": int(steps[np.argmin(loss)]),
                "loss_increases": int(increase.sum()),
                "maximum_loss_increase": float(max(0, delta.max()))
                if len(delta)
                else 0.0,
                "maximum_relative_loss_increase_percent": float(
                    max(0, np.max(100 * delta / loss[:-1]))
                )
                if len(delta)
                else 0.0,
            }
        )
    axes[0].set_ylabel("Mean squared top error / h² (display normalization)")
    for axis in axes:
        axis.legend(frameon=False)
    fig.suptitle(
        "Adam loss histories | crosses mark retained loss increases", fontsize=15
    )
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"adam-loss.{suffix}", dpi=180)
    plt.close(fig)
    videos = []
    for height in (0.05, 0.20):
        pair = [c for c in cases if float(c["history"]["height"]) == height]
        videos.append(render_pair(pair, height, output, cfg))
    manifest = {
        "input": str(source),
        "loss_plot": "all solved Adam states; rises retained; no outer line search; optimizer differentiates raw unnormalized MSE",
        "videos": videos,
        "cases": records,
    }
    (output / "adam-plot-metrics.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    cherries.log_metrics(
        {r["case"] + "/loss_increases": r["loss_increases"] for r in records}
    )
    LOG.info("Wrote Adam figures: %s", json.dumps(records))


if __name__ == "__main__":
    cherries.main(main, profile=ProfileAdamFigures)
