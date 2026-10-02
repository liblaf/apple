# ruff: noqa: RUF001
"""Plot measured cold-start force residuals without inventing missing samples."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent


class Config(cherries.BaseConfig):
    source: Path = GROUP / "data/cold-forward-comparison-001"
    output: Path = GROUP / "data/cold-forward-error-001"
    site: Path = GROUP / "data/current-model-report-001"


def main(cfg: Config) -> None:  # noqa: PLR0915
    cfg.output.mkdir(parents=True, exist_ok=False)
    summary_path = cfg.source / "summary.json"
    log_path = cfg.source / "stdout.log"
    summary = json.loads(summary_path.read_text())
    newton, hybrid = summary["results"]
    assert newton["method"] == "newton_diag"
    assert hybrid["method"] == "hybrid_diag"
    assert not newton["success"]
    assert not hybrid["success"]
    initial = newton["prewarm_force_norm"]
    assert initial == hybrid["prewarm_force_norm"]
    trace = newton["forward"]["solver"]["trace"]
    assert [step["iteration"] for step in trace] == list(range(100))
    n_steps = [step["iteration"] for step in trace] + [100]
    n_force = [step["force"] for step in trace] + [
        newton["forward"]["solver"]["grad_norm"]
    ]
    logged = re.findall(
        r"Expression PNCG accepted step (\d+), force ([\deE.+-]+)", log_path.read_text()
    )
    assert [int(step) for step, _ in logged] == [500, 1000, 1500, 2000]
    h_steps = [
        0,
        *[int(step) for step, _ in logged],
        hybrid["forward"]["solver"]["optimizer_step"],
    ]
    h_force = [
        initial,
        *[float(force) for _, force in logged],
        hybrid["forward"]["solver"]["grad_norm"],
    ]
    final_tolerance = summary["protocol"]["config"]["forward_atol"]
    switch = max(
        final_tolerance,
        initial * 1e-3,
        summary["protocol"]["config"]["newton_switch_atol"],
    )
    evidence = {
        "quantity": "free-force Euclidean norm divided by common initial free-force norm; not shape error",
        "initial_force": initial,
        "thresholds_relative": {
            "nominal_1e3": 1e-3,
            "actual_hybrid_switch": switch / initial,
            "final_convergence": final_tolerance / initial,
        },
        "newton": {
            "iteration": n_steps,
            "force": n_force,
            "relative_force": (np.array(n_force) / initial).tolist(),
            "total_seconds": newton["forward_wall_seconds"],
            "sampling": "pre-step forces at iterations 0..99, then exact post-update terminal force at 100",
        },
        "hybrid": {
            "iteration": h_steps,
            "force": h_force,
            "relative_force": (np.array(h_force) / initial).tolist(),
            "total_seconds": hybrid["forward_wall_seconds"],
            "sampling": "initial force, four logged PNCG samples, then last optimizer gradient in failure receipt; no invented intermediate values",
        },
        "time_axis_limit": "Exact synchronized total durations exist, but no exact per-state elapsed times. Log events occur after updates while Newton forces refer to pre-update states. No elapsed-time curve was inferred.",
        "source_sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (summary_path, log_path, Path(__file__))
        },
    }
    (cfg.output / "curve-data.json").write_text(json.dumps(evidence, indent=2) + "\n")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.9), sharey=True)
    colors = ("#2368a0", "#c06618")
    thresholds = [
        (switch / initial, "#9b5918", "--", "Hybrid switch: 7.89 × 10⁻³"),
        (1e-3, "#777777", ":", "Nominal relative target: 10⁻³"),
        (final_tolerance / initial, "#222222", "-.", "Final tolerance: 7.89 × 10⁻⁴"),
    ]
    for ax in axes:
        ax.set_yscale("log")
        ax.set_ylim(4e-4, 9)
        ax.grid(axis="y", which="major", alpha=0.22)
        ax.axhline(1, color="#aaaaaa", linewidth=0.8, alpha=0.7)
        for value, color, style, _ in thresholds:
            ax.axhline(value, color=color, linestyle=style, linewidth=1.25)
    axes[0].plot(
        n_steps,
        np.array(n_force) / initial,
        color=colors[0],
        linewidth=2,
        marker="o",
        markersize=2.3,
    )
    axes[0].set(
        xlabel="Newton iteration",
        ylabel=r"Relative force residual  $\|g_k\|_2\,/\,\|g_0\|_2$",
        xlim=(-2, 103),
    )
    axes[0].set_title(
        "Newton-CG only\n204.12 s · 100-iteration limit",
        loc="left",
        fontweight="bold",
        pad=12,
    )
    n_last = n_force[-1] / initial
    axes[0].scatter([100], [n_last], marker="X", s=80, color=colors[0], zorder=4)
    axes[0].annotate(
        f"Stopped: {n_last:.4g}\n7.25 × convergence tolerance",
        (100, n_last),
        xytext=(43, 0.040),
        arrowprops={"arrowstyle": "-", "color": colors[0]},
        fontsize=10,
        color=colors[0],
    )
    axes[1].plot(
        h_steps,
        np.array(h_force) / initial,
        color=colors[1],
        linestyle=(0, (3, 4)),
        linewidth=1.3,
        marker="o",
        markerfacecolor="white",
        markeredgewidth=1.7,
        markersize=6,
    )
    axes[1].scatter(
        [h_steps[-1]],
        [h_force[-1] / initial],
        marker="X",
        s=80,
        color=colors[1],
        zorder=4,
    )
    axes[1].set(xlabel="Reported PNCG iteration counter", xlim=(-45, 2410))
    axes[1].set_title(
        "Hybrid · PNCG phase only\n600.12 s · time limit; Newton never started",
        loc="left",
        fontweight="bold",
        pad=12,
    )
    axes[1].annotate(
        "Last recorded: 1.311",
        (h_steps[-1], h_force[-1] / initial),
        xytext=(800, 1.05),
        arrowprops={"arrowstyle": "-", "color": colors[1]},
        fontsize=10,
        color=colors[1],
    )
    axes[1].text(
        0.04,
        0.47,
        "Only four intermediate samples were logged.\nDashed segments are visual guides,\nnot a reconstructed trajectory.",
        transform=axes[1].transAxes,
        fontsize=9.5,
        color="#555555",
    )
    legend = [
        Line2D([0], [0], color=color, linestyle=style, linewidth=1.4, label=label)
        for _, color, style, label in thresholds
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.075),
        ncol=3,
        frameon=False,
        fontsize=9.5,
    )
    fig.suptitle(
        "Cold-start Smile: force-residual convergence",
        x=0.07,
        y=0.98,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.07,
        0.915,
        "Same saved stress and neutral initial geometry · both attempts missed the convergence target",
        color="#555555",
        fontsize=11,
    )
    fig.text(
        0.07,
        0.032,
        "Logarithmic residual axes; iteration scales differ. Initial force = 1.267878 × 10⁻⁵.\nPNCG terminal marker is the last recorded optimizer gradient; no validated endpoint was produced.",
        fontsize=9,
        color="#555555",
    )
    fig.subplots_adjust(left=0.09, right=0.975, bottom=0.23, top=0.79, wspace=0.15)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(
            cfg.output / f"cold-forward-error.{extension}", dpi=180, facecolor="white"
        )
    plt.close(fig)
    for name in (
        "cold-forward-error.png",
        "cold-forward-error.svg",
        "cold-forward-error.pdf",
        "curve-data.json",
    ):
        destination = (
            "cold-forward-curve-data.json" if name == "curve-data.json" else name
        )
        shutil.copy2(cfg.output / name, cfg.site / destination)
    start, end = "<!-- cold-error-curve:start -->", "<!-- cold-error-curve:end -->"
    section = (
        start
        + '<section><h2>Measured error curves</h2><figure><a href="cold-forward-error.png"><img src="cold-forward-error.png" alt="Relative force residual against Newton iteration and logged PNCG iteration; both remain above convergence tolerance"></a><figcaption>Force residual, not shape error. PNCG samples are sparse; dashed segments only connect recorded samples. Separate iteration scales.</figcaption></figure><p><a href="cold-forward-error.svg">SVG</a> · <a href="cold-forward-error.pdf">PDF</a> · <a href="cold-forward-curve-data.json">Curve data</a></p></section>'
        + end
    )
    path = cfg.site / "cold-forward.html"
    page = path.read_text()
    if start in page:
        page = re.sub(
            re.escape(start) + ".*?" + re.escape(end),
            lambda _: section,
            page,
            flags=re.DOTALL,
        )
    else:
        page = page.replace("<h2>", section + "<h2>", 1)
    path.write_text(page)
    output_relative = cfg.output.resolve().relative_to(GROUP)
    report = f"""# Cold-start error curves

The figure plots measured relative free-force residual against solver progress, with separate Newton and PNCG iteration scales. It is not a skin-position or energy-error plot. Both arms use initial force `{initial:.17g}`. The actual hybrid switch is `{switch / initial:.8g}` relative; final convergence is `{final_tolerance / initial:.8g}` relative; the nominal `1e-3` reference is also shown.

![Cold-start force residual](../{output_relative}/cold-forward-error.png)

Newton has 100 pre-step force records plus its terminal force. PNCG has only four intermediate logged samples at counters 500, 1000, 1500 and 2000, plus the common initial force and last optimizer gradient at failure counter 2289. Dashed PNCG lines are visual guides, not interpolation claims. Missing oscillations cannot be inferred. Its terminal sample is not an independent recomputation at a saved endpoint.

The exact synchronized totals are 204.124 s and 600.117 s. The logs do not contain exact per-state elapsed times; Newton logs a pre-update force after accepting its update. An exact elapsed-time curve therefore cannot be reconstructed and is not plotted. No GPU solve was rerun.

Source receipts: `data/cold-forward-comparison-001/summary.json` and `stdout.log`; hashes and every plotted coordinate are in `{output_relative}/curve-data.json`. PNG, editable SVG and PDF are in the same directory. The temporary tailnet report includes the figure.

Reproduce from this experiment directory with a new output name:

```sh
DEBUG=1 CHERRIES_NAME=cold-forward-error-curves CHERRIES_TAGS=smile,performance,plot \\
  uv run python src/45-plot-cold-error.py --output data/NEW-PLOT
```

Cherries records local logs and artifacts; remote Comet is disabled for this local analysis. Plot generation does not modify raw numerical receipts or physics.
"""
    (GROUP / "docs/45-cold-forward-error-curves.md").write_text(report)
    cherries.log_output(cfg.output)
    cherries.log_output(GROUP / "docs/45-cold-forward-error-curves.md")


if __name__ == "__main__":
    cherries.main(main)
