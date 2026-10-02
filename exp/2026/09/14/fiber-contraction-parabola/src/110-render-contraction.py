"""Render the three-mode Adam contraction comparison without rerunning it."""

from __future__ import annotations

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
MODES = ("x_contraction", "contraction_only", "unconstrained")
LABELS = {
    "x_contraction": "x-fiber contraction",
    "contraction_only": "contraction only (free directions)",
    "unconstrained": "free symmetric B",
}
COLORS = {
    "x_contraction": "#b24b35",
    "contraction_only": "#547c38",
    "unconstrained": "#246fa6",
}


class ProfileContractionFigures(profiles.Profile):
    def init(self):
        run = core.run
        run.plugins.register(plugins.Comet(run=run, disabled=False))
        run.plugins.register(plugins.Git(run=run, commit=False))
        run.plugins.register(plugins.Local(run=run))
        run.plugins.register(plugins.Logging(run=run))
        return run


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    input_dir: Path = Path("100-adam-contraction")
    output: Path = Path("110-contraction-figures")
    frame_stride: int = 10
    fps: int = 15


def load_case(source: Path, name: str) -> dict[str, Any]:
    folder = source / name
    with (folder / "trace.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    history = dict(np.load(folder / "history.npz", allow_pickle=False))
    assert len(rows) == len(history["u"]) == len(history["controls"])
    assert [int(row["step"]) for row in rows] == list(range(len(rows)))
    return {
        "name": name,
        "history": history,
        "rows": rows,
        "summary": json.loads((folder / "summary.json").read_text()),
    }


def common_limits(
    cases: list[dict[str, Any]], height: float
) -> tuple[float, float, float, float]:
    positions = np.concatenate(
        [
            (case["history"]["points"][None] + case["history"]["u"]).reshape(-1, 2)
            for case in cases
        ]
    )
    return (
        min(-0.02, float(positions[:, 0].min()) - 0.015),
        max(1.02, float(positions[:, 0].max()) + 0.015),
        min(-0.01, float(positions[:, 1].min()) - 0.01),
        max(0.1 + height + 0.01, float(positions[:, 1].max()) + 0.01),
    )


def draw_shape(
    axis: Axes,
    case: dict[str, Any],
    state: int,
    limits: tuple[float, float, float, float],
) -> None:
    history = case["history"]
    points = history["points"] + history["u"][state]
    axis.add_collection(
        PolyCollection(
            points[history["triangles"]],
            facecolors=np.where(
                history["muscle"][:, None],
                np.array([[0.73, 0.36, 0.31]]),
                np.array([[0.85, 0.70, 0.51]]),
            ),
            edgecolors="#5b4b40",
            linewidths=0.08,
        )
    )
    x = np.linspace(0.0, 1.0, 401)
    axis.plot(
        x,
        0.1 + 4.0 * float(history["height"]) * x * (1.0 - x),
        "k--",
        lw=1.1,
    )
    axis.set(xlim=limits[:2], ylim=limits[2:], xlabel="deformed x", ylabel="absolute y")
    axis.set_aspect("equal", adjustable="box")


def loss_data(case: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    steps = np.array([int(row["step"]) for row in case["rows"]])
    loss = np.array([float(row["objective_normalized"]) for row in case["rows"]])
    delta = np.diff(loss)
    rises = np.r_[False, delta > np.maximum(1e-12, 1e-10 * np.abs(loss[:-1]))]
    return steps, loss, rises


def render_loss(cases: list[dict[str, Any]], output: Path) -> list[dict[str, Any]]:
    all_loss = np.concatenate([loss_data(case)[1] for case in cases])
    loss_padding = 0.04 * max(float(np.ptp(all_loss)), 1.0e-3)
    loss_limits = (
        float(all_loss.min() - loss_padding),
        float(all_loss.max() + loss_padding),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.3))
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.20, top=0.67, wspace=0.25)
    records = []
    for column, height in enumerate((0.05, 0.20)):
        axis = axes[column]
        selected = [c for c in cases if float(c["history"]["height"]) == height]
        for case in selected:
            mode = str(case["history"]["mode"])
            steps, loss, rises = loss_data(case)
            axis.plot(steps, loss, color=COLORS[mode], lw=1.4, label=LABELS[mode])
            axis.scatter(
                steps[rises], loss[rises], color=COLORS[mode], marker="x", s=14
            )
            axis.scatter(steps[-1], loss[-1], color=COLORS[mode], s=25, zorder=3)
            delta = np.diff(loss)
            records.append(
                {
                    "case": case["name"],
                    "states": len(steps),
                    "last_solved_step": int(steps[-1]),
                    "initial_loss": float(loss[0]),
                    "final_loss": float(loss[-1]),
                    "best_loss": float(loss.min()),
                    "best_step": int(steps[np.argmin(loss)]),
                    "loss_change": float(loss[-1] - loss[0]),
                    "loss_rises": int(rises.sum()),
                    "maximum_loss_rise": float(max(0.0, delta.max()))
                    if len(delta)
                    else 0.0,
                    "optimizer_message": case["summary"]["optimizer_message"],
                }
            )
        axis.set(
            title=f"Target height h = {height:g}",
            xlabel="Solved Adam update",
            ylabel="Mean squared top error / h²",
            ylim=loss_limits,
        )
        axis.grid(alpha=0.22)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(
        "Adam loss histories | x marks a retained loss rise; dots mark each endpoint",
        y=0.99,
    )
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"contraction-loss.{suffix}", dpi=180)
    plt.close(fig)
    return records


def render_final_shapes(
    cases: list[dict[str, Any]], height: float, output: Path
) -> dict[str, Any]:
    selected = [c for c in cases if float(c["history"]["height"]) == height]
    assert [str(c["history"]["mode"]) for c in selected] == list(MODES)
    limits = common_limits(selected, height)
    # Equal data aspect is maintained; the deliberately shallow h=.05 canvas
    # avoids spending most of the image on whitespace above a thin strip.
    figure_height = 2.4 if height == 0.05 else 3.4
    fig, axes = plt.subplots(1, 3, figsize=(14, figure_height), layout="constrained")
    last_steps = []
    for axis, case in zip(axes, selected, strict=True):
        draw_shape(axis, case, -1, limits)
        mode = str(case["history"]["mode"])
        last_step = int(case["rows"][-1]["step"])
        last_steps.append(last_step)
        axis.set_title(f"{LABELS[mode]}\nendpoint step {last_step}", fontsize=9)
    fig.suptitle(
        f"Final deformations, h = {height:g} | dashed black: target | common scale, equal aspect",
        fontsize=11,
    )
    suffix = f"h{round(height * 1000):03d}"
    for extension in ("png", "pdf"):
        fig.savefig(output / f"contraction-final-{suffix}.{extension}", dpi=180)
    plt.close(fig)
    return {"height": height, "limits": limits, "last_solved_steps": last_steps}


def render_history(
    cases: list[dict[str, Any]], height: float, output: Path, cfg: Config
) -> dict[str, Any]:
    selected = [c for c in cases if float(c["history"]["height"]) == height]
    limits = common_limits(selected, height)
    final = max(len(case["history"]["u"]) - 1 for case in selected)
    requested = sorted({*range(0, final + 1, cfg.frame_stride), final})
    suffix = f"h{round(height * 1000):03d}"
    figure_height = 2.4 if height == 0.05 else 3.4
    fig, axes = plt.subplots(1, 3, figsize=(14, figure_height), layout="constrained")
    writer = animation.FFMpegWriter(
        fps=cfg.fps,
        metadata={"title": f"Three Adam contraction modes, h={height:g}"},
    )
    with writer.saving(fig, str(output / f"contraction-history-{suffix}.mp4"), dpi=150):
        for step in requested:
            held_modes = []
            for axis, case in zip(axes, selected, strict=True):
                axis.clear()
                saved_step = min(step, len(case["history"]["u"]) - 1)
                draw_shape(axis, case, saved_step, limits)
                mode = str(case["history"]["mode"])
                held = saved_step != step
                if held:
                    held_modes.append(mode)
                state_label = (
                    f"held at step {saved_step}" if held else f"step {saved_step}"
                )
                axis.set_title(f"{LABELS[mode]}\n{state_label}", fontsize=9)
            held_label = (
                " | held endpoint: " + ", ".join(held_modes) if held_modes else ""
            )
            fig.suptitle(
                f"Adam deformation history, h = {height:g} | requested step {step}{held_label}",
                fontsize=11,
            )
            writer.grab_frame()
    plt.close(fig)
    return {
        "height": height,
        "requested_steps": requested,
        "last_solved_steps": [len(c["history"]["u"]) - 1 for c in selected],
        "fps": cfg.fps,
    }


def main(cfg: Config) -> None:
    source = GROUP / "data" / cfg.input_dir
    protocol = json.loads((source / "protocol.json").read_text())
    heights = tuple(map(float, protocol["config"]["heights"].split(",")))
    assert heights == (0.05, 0.20)
    assert tuple(protocol["config"]["modes"].split(",")) == MODES
    names = [
        f"h{round(height * 1000):03d}-{mode}" for height in heights for mode in MODES
    ]
    missing = [name for name in names if not (source / name / "summary.json").is_file()]
    if missing:
        message = f"All six Adam summaries are required; missing {missing}"
        raise FileNotFoundError(message)
    cases = [load_case(source, name) for name in names]
    output = cherries.output(cfg.output)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / Path(__file__).name)
    metrics = render_loss(cases, output)
    final_shapes = [render_final_shapes(cases, height, output) for height in heights]
    histories = [render_history(cases, height, output, cfg) for height in heights]
    manifest = {
        "input": str(source),
        "loss_plot": "all real solved Adam states; x marks retained rises; dots mark endpoints",
        "shape_panels": "common limits within each height and equal data aspect; no vertical exaggeration",
        "history_videos": "requested global steps; each stopped case holds its actual final solved state",
        "metrics": metrics,
        "final_shapes": final_shapes,
        "histories": histories,
    }
    (output / "contraction-figure-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    cherries.log_metrics(
        {f"{item['case']}/loss_rises": item["loss_rises"] for item in metrics}
    )
    LOG.info("Wrote three-mode contraction figures to %s", output)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileContractionFigures)
