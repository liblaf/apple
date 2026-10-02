"""Plot the saved cold-start Newton energy and observed energy decreases."""

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

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent


class Config(cherries.BaseConfig):
    source: Path = GROUP / "data/cold-forward-comparison-001/summary.json"
    output: Path = GROUP / "data/cold-forward-energy-001"
    site: Path = GROUP / "data/current-model-report-001"


def main(cfg: Config) -> None:
    cfg.output.mkdir(parents=True, exist_ok=False)
    summary = json.loads(cfg.source.read_text())
    row = summary["results"][0]
    assert row["method"] == "newton_diag"
    trace = row["forward"]["solver"]["trace"]
    steps = np.array([step["iteration"] for step in trace])
    energy = np.array([step["energy"] for step in trace])
    assert np.array_equal(steps, np.arange(100))
    assert np.all(np.isfinite(energy))
    decrease = -np.diff(energy)
    assert np.all(decrease > 0)
    evidence = {
        "method": "newton_diag",
        "quantity": "inner physical objective from problem.fun; solver energy units",
        "sampling": "100 pre-update states, iterations 0..99; terminal post-update energy at iteration 100 was not saved",
        "iteration": steps.tolist(),
        "energy": energy.tolist(),
        "decrease_per_transition": decrease.tolist(),
        "strictly_decreasing_recorded_transitions": int(np.count_nonzero(decrease > 0)),
        "net_recorded_decrease": float(energy[0] - energy[-1]),
        "pncg_energy_available": False,
        "terminal_force": row["forward"]["solver"]["grad_norm"],
        "force_tolerance": summary["protocol"]["config"]["forward_atol"],
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (cfg.source, Path(__file__))
        },
    }
    (cfg.output / "energy-data.json").write_text(json.dumps(evidence, indent=2) + "\n")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.8))
    axes[0].plot(
        steps, energy, color="#2368a0", linewidth=2.1, marker="o", markersize=2.7
    )
    axes[0].set(
        xlabel="Newton iteration (pre-update state)",
        ylabel="Mechanical objective (solver energy units)",
        title="Recorded energy",
        xlim=(-2, 101),
    )
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    axes[0].annotate(
        f"First: {energy[0]:.5e}",
        (steps[0], energy[0]),
        xytext=(23, -2.0e-7),
        fontsize=10,
        color="#2368a0",
        arrowprops={"arrowstyle": "-", "color": "#2368a0"},
    )
    axes[0].annotate(
        f"Last recorded: {energy[-1]:.5e}",
        (steps[-1], energy[-1]),
        xytext=(33, -5.3e-7),
        fontsize=10,
        color="#2368a0",
        arrowprops={"arrowstyle": "-", "color": "#2368a0"},
    )
    axes[1].semilogy(
        steps[1:], decrease, color="#18806c", linewidth=1.5, marker="o", markersize=3
    )
    axes[1].set(
        xlabel="Recorded transition ending at iteration k",
        ylabel=r"Energy decrease  $E_{k-1} - E_k$  (solver units)",
        title="Decrease per recorded iteration",
        xlim=(-1, 101),
    )
    axes[1].text(
        0.27,
        0.97,
        "99/99 recorded transitions decrease energy.\nForce convergence was not reached.",
        transform=axes[1].transAxes,
        va="top",
        fontsize=9.5,
        color="#555555",
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
    )
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.title.set_fontweight("bold")
    fig.suptitle(
        "Cold-start Newton-CG: mechanical energy",
        x=0.075,
        y=0.98,
        ha="left",
        fontsize=18,
        fontweight="bold",
    )
    fig.text(
        0.075,
        0.905,
        "Saved history only; no new solve. Energy decreases while the force target remains unmet.",
        color="#555555",
        fontsize=11,
    )
    fig.text(
        0.075,
        0.025,
        "Energy is the inner physical objective, not the inverse skin-fitting loss. Values retain the solver's energy offset.\nThe energy after update 100 was not saved; PNCG energy was not recorded in this test.",
        color="#555555",
        fontsize=9.3,
    )
    fig.subplots_adjust(left=0.085, right=0.975, bottom=0.19, top=0.76, wspace=0.30)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(
            cfg.output / f"cold-forward-energy.{extension}", dpi=180, facecolor="white"
        )
    plt.close(fig)
    for name in (
        "cold-forward-energy.png",
        "cold-forward-energy.svg",
        "cold-forward-energy.pdf",
        "energy-data.json",
    ):
        target = "cold-forward-energy-data.json" if name == "energy-data.json" else name
        shutil.copy2(cfg.output / name, cfg.site / target)
    start, end = "<!-- cold-energy:start -->", "<!-- cold-energy:end -->"
    section = (
        start
        + '<section><h2>Saved Newton energy curve</h2><figure><a href="cold-forward-energy.png"><img src="cold-forward-energy.png" alt="Saved Newton mechanical energy decreases; per-iteration decreases become smaller, but force convergence was not reached"></a><figcaption>100 pre-update energy samples; all 99 recorded transitions decrease. PNCG energy was not recorded. No new solve.</figcaption></figure><p><a href="cold-forward-energy.svg">SVG</a> · <a href="cold-forward-energy.pdf">PDF</a> · <a href="cold-forward-energy-data.json">Energy data</a></p></section>'
        + end
    )
    page = cfg.site / "cold-forward.html"
    content = page.read_text()
    if start in content:
        content = re.sub(
            re.escape(start) + ".*?" + re.escape(end),
            lambda _: section,
            content,
            flags=re.DOTALL,
        )
    else:
        content = content.replace("<h2>", section + "<h2>", 1)
    page.write_text(content)
    relative = cfg.output.resolve().relative_to(GROUP)
    report = f"""# Saved Newton energy curve

![Mechanical energy](../{relative}/cold-forward-energy.png)

The 100 pre-update energies at Newton iterations 0 through 99 decrease strictly across all 99 recorded transitions, from `{energy[0]:.17g}` to `{energy[-1]:.17g}` solver energy units. Total recorded decrease is `{energy[0] - energy[-1]:.17g}`. The right panel shows positive differences between successive recorded energies on a logarithmic axis.

This is the inner physical objective, not inverse skin-position loss or a force-error estimate. The absolute energy includes the model's chosen offset. The terminal post-update energy after step 100 was not saved. PNCG energies were not recorded, so no PNCG curve is inferred. The user selected plotting saved Newton energy; no GPU solve was rerun. Despite decreasing energy, the final force was `{evidence["terminal_force"]:.17g}`, above the `1e-8` convergence target.

Data and source hashes are in `{relative}/energy-data.json`; PNG, SVG and PDF are alongside it. Reproduce from the experiment directory with a new output directory:

```sh
DEBUG=1 CHERRIES_NAME=saved-newton-energy CHERRIES_TAGS=smile,performance,energy,plot \\
  uv run python src/46-plot-newton-energy.py --output data/NEW-PLOT
```

Cherries local logging is enabled; remote Comet is disabled for this CPU-only plotting task. Raw solver artifacts remain unchanged.
"""
    (GROUP / "docs/46-saved-newton-energy.md").write_text(report)
    cherries.log_output(cfg.output)
    cherries.log_output(GROUP / "docs/46-saved-newton-energy.md")


if __name__ == "__main__":
    cherries.main(main)
