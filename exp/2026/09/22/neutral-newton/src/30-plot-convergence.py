"""Plot the single forward's saved energy and gradient norm without re-solving."""

# ruff: noqa: E402, PLR0915, RUF001

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
JOINT = GROUP.parents[4] / "exp/2026/09/21/joint-activation-material-mandible"
sys.path.insert(0, str(JOINT / "src"))
from joint_common import ProfileJoint, sha256, write_json


class Config(cherries.BaseConfig):
    run_dir: Path = GROUP / "data/forward-002"
    output_dir: Path = GROUP / "data/convergence-plots-001"


def main(cfg: Config) -> None:
    summary_path = cfg.run_dir / "summary.json"
    protocol_path = cfg.run_dir / "protocol.json"
    trace_path = cfg.run_dir / "trace.jsonl"
    summary = json.loads(summary_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    assert sha256(protocol_path) == summary["protocol"]["sha256"]
    trace = [json.loads(line) for line in trace_path.read_text().splitlines()]
    assert len(trace) == summary["accepted_steps"] + 1
    assert all(row["accepted"] for row in trace)
    iteration = np.asarray([row["iteration"] for row in trace])
    np.testing.assert_array_equal(iteration, np.arange(len(trace)))
    energy = np.asarray([row["energy"] for row in trace])
    gradient = np.asarray([row["grad_norm"] for row in trace])
    np.testing.assert_allclose(gradient[-1], summary["final_grad_norm"], rtol=1e-12)
    np.testing.assert_allclose(energy[-1], summary["final_energy"], rtol=1e-12)
    assert np.all(np.diff(energy) < 0)
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, cfg.output_dir / "30-plot-convergence.py")
    threshold = summary["effective_grad_threshold"]
    ratio = summary["final_grad_norm"] / threshold
    resume = summary.get("resume")
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 11, "axes.titleweight": "semibold"}
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    fig.subplots_adjust(left=0.075, right=0.975, bottom=0.21, top=0.80, wspace=0.31)
    fig.suptitle(
        "Neutral face · Newton-CG continuation"
        if resume
        else "Neutral face · one cold-start Newton-CG forward",
        x=0.075,
        y=0.98,
        ha="left",
        fontsize=16,
        fontweight="semibold",
    )
    axes[0].plot(iteration, energy, color="#166b93", linewidth=2.2)
    axes[0].scatter(iteration[-1], energy[-1], color="#166b93", s=28, zorder=3)
    axes[0].axhline(0, color="#a7adb4", linewidth=0.7)
    axes[0].set(title="Energy", ylabel="Energy [MPa · m³]")
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    axes[1].semilogy(iteration, gradient, color="#7852a0", linewidth=2.2)
    axes[1].scatter(iteration[-1], gradient[-1], color="#7852a0", s=28, zorder=3)
    axes[1].axhline(
        threshold,
        color="#b65d35",
        linestyle="--",
        linewidth=1.5,
        label=f"Stopping threshold: {threshold:.2e}",
    )
    axes[1].axhline(
        protocol["config"]["atol"],
        color="#888888",
        linestyle=":",
        linewidth=1.1,
        label="Absolute tolerance: 1.00e−08",
    )
    axes[1].annotate(
        f"Final: {gradient[-1]:.2e}\n{ratio:.1f}× stopping threshold",
        xy=(iteration[-1], gradient[-1]),
        xytext=(0.53 * summary["accepted_steps"], 3e-6),
        fontsize=10,
        arrowprops={"arrowstyle": "-", "color": "#7852a0"},
        color="#52346f",
    )
    axes[1].set(
        title="Gradient norm", ylabel="‖∇ E‖₂ [MPa · m²]", ylim=(0.6e-8, 1.2e-4)
    )
    axes[1].legend(loc="upper right", fontsize=8.5, frameon=False)
    for axis in axes:
        if resume:
            axis.axvline(
                resume["start_iteration"],
                color="#aaaaaa",
                linewidth=0.9,
                linestyle="--",
            )
        axis.set_xlabel("Accepted Newton iteration")
        axis.set_xlim(0, summary["accepted_steps"])
        axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.17)
    status = "Converged" if summary["success"] else "Not converged"
    fig.text(
        0.075,
        0.88,
        f"{summary['accepted_steps']} iterations   ·   {summary['forward_seconds']:.2f} s{' cumulative' if resume else ''}   ·   {status}   ·   {summary['geometry']['inverted_tetrahedra']} inverted tetrahedra",
        fontsize=11,
        color="#9a422c" if not summary["success"] else "#087d81",
    )
    fig.text(
        0.075,
        0.075,
        "Stopping threshold = max(absolute tolerance, relative tolerance × initial gradient norm).",
        fontsize=9,
        color="#535c67",
    )
    fig.text(
        0.075,
        0.035,
        (
            "Vertical line: resumed from iteration 100. "
            if resume
            else "Saved accepted states, including iteration 0. "
        )
        + "Multiply energy by 10⁶ for J and gradient norm by 10⁶ for N.",
        fontsize=9,
        color="#535c67",
    )
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(
            cfg.output_dir / f"energy-gradient.{suffix}", dpi=200, facecolor="white"
        )
    plt.close(fig)
    receipt = {
        "schema": "neutral-forward-energy-gradient-plots-v1",
        "forward_solves": 0,
        "inputs": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in (
                ("summary", summary_path),
                ("protocol", protocol_path),
                ("trace", trace_path),
            )
        },
        "accepted_states": len(trace),
        "energy_strictly_decreasing": True,
        "final_gradient_over_threshold": ratio,
        "figures": {
            suffix: sha256(cfg.output_dir / f"energy-gradient.{suffix}")
            for suffix in ("png", "svg", "pdf")
        },
    }
    write_json(cfg.output_dir / "summary.json", receipt)
    cherries.log_output(cfg.output_dir)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
