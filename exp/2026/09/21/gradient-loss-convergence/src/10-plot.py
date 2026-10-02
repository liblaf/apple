"""Plot and audit saved inverse-physics loss histories without rerunning fitting."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import comet_ml
import matplotlib as mpl
import numpy as np
from liblaf.cherries import core, plugins, profiles

from liblaf import cherries

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

GROUP = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[6]
PREVIOUS = ROOT / "exp/2026/09/19"
logger = logging.getLogger(__name__)
CASES = (
    (
        "raw6-l2",
        "gradient-only-face/data/10-comparison/l2",
        "L2 control",
        "#777777",
        "--",
        0,
    ),
    (
        "raw6-gradient",
        "gradient-only-face/data/10-comparison/gradient",
        "Gradient only",
        "#1769aa",
        "-",
        0,
    ),
    (
        "axis-off",
        "learned-axis-gradient-face/data/20-comparison/off",
        "Smoothness off",
        "#1769aa",
        "-",
        1,
    ),
    (
        "axis-on",
        "learned-axis-gradient-face/data/20-comparison/on",
        "Smoothness on",
        "#c15a19",
        "-",
        1,
    ),
    (
        "mixed-off",
        "mixed-loss-learned-axis-face/data/20-comparison/mixed-off",
        "Smoothness off",
        "#1769aa",
        "-",
        2,
    ),
    (
        "mixed-on",
        "mixed-loss-learned-axis-face/data/20-comparison/mixed-on",
        "Smoothness on",
        "#c15a19",
        "-",
        2,
    ),
)
TITLES = (
    "Original 6-DoF activation\nGradient loss and L2 control",
    "Learned-axis contraction only\nGradient data loss",
    "Learned-axis contraction only\nMixed L2 + gradient data loss",
)


class Comet(plugins.Comet):
    @core.impl
    def start(self) -> None:
        experiment = comet_ml.start(
            project_name=self.run.project_name,
            experiment_config=comet_ml.ExperimentConfig(
                disabled=self.disabled,
                name=self.run.run_name,
                tags=self.run.tags,
                log_env_details=False,
                log_git_patch=False,
                auto_log_co2=False,
            ),
        )
        self.run.log_other("cherries/comet/url", experiment.url)


class ProfileNoCommit(profiles.Profile):
    def init(self) -> core.Run:
        run = core.run
        run.plugins.register(Comet(run=run, disabled=os.environ.get("DEBUG") == "1"))
        run.plugins.register(plugins.Git(run=run, commit=False))
        run.plugins.register(plugins.Logging(run=run))
        run.plugins.register(plugins.Local(run=run))
        return run


class Config(cherries.BaseConfig):
    output_dir: Path = GROUP / "data/10-curves"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_case(case: tuple, output: Path) -> dict:
    identifier, relative, label, color, style, column = case
    folder = PREVIOUS / relative
    trace_path = folder / "trace.csv"
    summary_path = folder / "summary.json"
    with trace_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    trace = {key: np.array([float(row[key]) for row in rows]) for key in rows[0]}
    assert all(np.isfinite(values).all() for values in trace.values())
    assert np.array_equal(trace["step"], np.arange(len(rows)))
    summary = json.loads(summary_path.read_text())
    assert summary["last_step"] == trace["step"][-1]
    assert summary["status"] == "completed_budget_not_convergence_certified"
    for key in ("objective", "surface_gradient_loss", "fit_rms_mm"):
        np.testing.assert_allclose(
            trace[key][-1], summary["last_metrics"][key], rtol=1e-13
        )
    inputs = []
    for source in (trace_path, summary_path, folder.parent / "protocol.json"):
        snapshot = output / "inputs" / identifier / source.name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, snapshot)
        inputs.append(
            {"path": str(source), "snapshot": str(snapshot), "sha256": digest(source)}
        )
        assert digest(snapshot) == inputs[-1]["sha256"]
        cherries.input(snapshot)
    norm_key = "gradient_rms" if column == 0 else "projected_gradient_rms"
    objective = trace["objective"]
    assert np.all(objective > 0)
    assert np.all(trace[norm_key] > 0)
    first_inverted = np.flatnonzero(trace["inverted_all_cells"] > 0)
    record = {
        "id": identifier,
        "label": label,
        "column": column,
        "status": summary["status"],
        "last_step": int(trace["step"][-1]),
        "initial_objective": float(objective[0]),
        "final_objective": float(objective[-1]),
        "objective_ratio": float(objective[-1] / objective[0]),
        "last_10_drop_percent": float(100 * (1 - objective[-1] / objective[-11])),
        "last_25_drop_percent": float(100 * (1 - objective[-1] / objective[-26])),
        "loss_increase_count": int(np.count_nonzero(np.diff(objective) > 0)),
        "optimizer_norm_key": norm_key,
        "optimizer_norm_ratio": float(trace[norm_key][-1] / trace[norm_key][0]),
        "final_position_rms_mm": float(trace["fit_rms_mm"][-1]),
        "final_surface_gradient_loss": float(trace["surface_gradient_loss"][-1]),
        "final_inverted_tets": int(trace["inverted_all_cells"][-1]),
        "first_inverted_step": int(trace["step"][first_inverted[0]])
        if len(first_inverted)
        else None,
        "sources": inputs,
    }
    if column > 0:
        span = float((objective[-26:].max() - objective[-26:].min()) / objective[-26])
        np.testing.assert_allclose(
            span, trace["recent_objective_relative_span"][-1], rtol=1e-13
        )
        np.testing.assert_allclose(
            record["optimizer_norm_ratio"],
            trace["projected_gradient_ratio"][-1],
            rtol=1e-13,
        )
        record["last_scheduled_relative_span"] = span
        record["plateau_threshold"] = 0.001
        record["stationarity_threshold"] = 0.01
        record["final_thresholds_passed"] = bool(
            span < 0.001 and record["optimizer_norm_ratio"] < 0.01
        )
    return {"record": record, "trace": trace, "color": color, "style": style}


def format_axis(ax: plt.Axes) -> None:
    ax.grid(alpha=0.22)
    ax.set_xlabel("Optimizer updates")
    ax.spines[["top", "right"]].set_visible(False)


def save_figure(fig: plt.Figure, output: Path, stem: str) -> None:
    for suffix in ("png", "svg"):
        path = output / f"{stem}.{suffix}"
        fig.savefig(path, dpi=180, facecolor="white", bbox_inches="tight")
        cherries.log_output(path)
    plt.close(fig)


def plot_overview(cases: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 8.4), layout="constrained")
    for col, title in enumerate(TITLES):
        axes[0, col].set_title(title, fontweight="bold", pad=12)
        axes[0, col].set_ylabel("Own objective J / J(0)")
        axes[0, col].set_ylim(0.0, 1.05)
        axes[1, col].set_yscale("log")
        axes[1, col].set_ylim(0.008, 1.35)
        axes[1, col].yaxis.set_major_formatter(PercentFormatter(1))
        axes[1, col].set_ylabel(
            "Activation-gradient RMS / initial"
            if col == 0
            else "Projected-gradient RMS / initial"
        )
        if col > 0:
            axes[1, col].axhline(0.01, color="#555555", ls=":", lw=1.4)
            axes[1, col].text(
                5, 0.012, "1% stationarity threshold", fontsize=9, color="#555555"
            )
            for row in range(2):
                axes[row, col].axvline(100, color="#bbbbbb", ls=":", lw=1)
            axes[0, col].text(
                103, 1.005, "learning rate halved", fontsize=8, color="#666666"
            )
        for row in range(2):
            format_axis(axes[row, col])
    for case in cases:
        r, t, color, style = (
            case[key] for key in ("record", "trace", "color", "style")
        )
        col = r["column"]
        x, y = t["step"], t["objective"] / t["objective"][0]
        axes[0, col].plot(x, y, color=color, ls=style, lw=2.3, label=r["label"])
        if col > 0 and r["id"].endswith("on"):
            axes[0, col].plot(
                x,
                t["data_objective"] / t["objective"][0],
                color=color,
                ls=":",
                lw=1.8,
                label="On: data term only",
            )
        ratio = t[r["optimizer_norm_key"]] / t[r["optimizer_norm_key"]][0]
        axes[1, col].plot(x, ratio, color=color, ls=style, lw=2.3)
        for row, values in ((0, y), (1, ratio)):
            axes[row, col].scatter(x[-1], values[-1], color=color, s=22, zorder=5)
            offset = 10 if r["id"].endswith("l2") or r["id"].endswith("on") else -15
            if row == 0 and col == 0:
                offset = -15 if r["id"] == "raw6-l2" else 10
            axes[row, col].annotate(
                f"{values[-1]:.3f}" if row == 0 else f"{100 * values[-1]:.1f}%",
                (x[-1], values[-1]),
                xytext=(-5, offset),
                textcoords="offset points",
                ha="right",
                color=color,
                fontsize=10,
                fontweight="bold",
            )
        if r["id"] == "raw6-gradient":
            step = r["first_inverted_step"]
            axes[0, col].axvline(step, color="#9a3030", ls=":", lw=1)
            axes[0, col].text(
                step - 2,
                0.96,
                f"first inversion: {step}",
                rotation=90,
                ha="right",
                va="top",
                color="#9a3030",
                fontsize=8,
            )
    for col in range(3):
        axes[0, col].legend(loc="lower left", frameon=False, fontsize=9)
    fig.suptitle(
        "Inverse-physics loss histories: none met convergence criteria",
        fontsize=18,
        fontweight="bold",
    )
    fig.supxlabel(
        "Each curve uses its own initial objective. Smoothness-on solid curves include the regularizer; dotted curves omit it.\n"
        "Columns use different activation models or objectives; curve height is not a fit-quality ranking.",
        fontsize=10,
    )
    save_figure(fig, output, "loss-and-stationarity")


def plot_tail(cases: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), layout="constrained")
    for col, title in enumerate(TITLES):
        ax = axes[col]
        ax.set_title(title, fontweight="bold", pad=12)
        ax.set_xlabel("Updates relative to recorded endpoint")
        ax.set_ylabel("Objective decrease since start of final 25 updates (%)")
        ax.set_xlim(-25.5, 1.5)
        maximum = max(
            case["record"]["last_25_drop_percent"]
            for case in cases
            if case["record"]["column"] == col
        )
        ax.set_ylim(-0.04 * maximum, 1.25 * maximum)
        ax.grid(alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
        if col > 0:
            ax.axhline(0.1, color="#555555", ls=":", lw=1.2)
            ax.text(
                -1,
                0.16,
                "0.1% plateau threshold",
                color="#555555",
                fontsize=9,
                ha="right",
            )
    for case in cases:
        r, t, color, style = (
            case[key] for key in ("record", "trace", "color", "style")
        )
        x = t["step"][-26:] - t["step"][-1]
        decrease = 100 * (1 - t["objective"][-26:] / t["objective"][-26])
        ax = axes[r["column"]]
        ax.plot(x, decrease, color=color, ls=style, lw=2.3, label=r["label"])
        ax.scatter(0, decrease[-1], color=color, s=25)
        ax.annotate(
            f"{decrease[-1]:.2f}%",
            (0, decrease[-1]),
            xytext=(-5, 8),
            textcoords="offset points",
            ha="right",
            color=color,
            fontweight="bold",
        )
    for ax in axes:
        ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.suptitle(
        "The loss was still falling when each update budget ended",
        fontsize=18,
        fontweight="bold",
    )
    fig.supxlabel(
        "Full-resolution recorded samples; no smoothing or extrapolation. "
        "All six objectives decrease at every recorded update.",
        fontsize=10,
    )
    save_figure(fig, output, "final-25-updates")


def write_report(records: list[dict]) -> Path:
    report = cherries.output(GROUP / "docs/10-results.md", mkdir=True)
    lines = [
        "# Gradient-loss inverse-physics convergence curves",
        "",
        "The saved runs **did not converge within their recorded budgets**. Their actual optimized",
        "objectives were still decreasing, and the recorded inverse-optimization gradient norms",
        "remained substantial. This analysis reads existing histories only; no fitting was resumed.",
        "",
        "![Full loss and stationarity histories](../data/10-curves/loss-and-stationarity.png)",
        "",
        "Each top panel plots the actual objective divided by its own neutral value. For smoothness-on",
        "runs the solid curve includes the activation regularizer and the dotted curve shows only the data term.",
        "Curve heights across objectives are not comparable",
        "fit scores. The lower panels measure optimization gradients, not the surface-gradient data loss.",
        "",
        "| Run | Updates | Initial objective | Final objective | Last 10 updates: decrease | Last 25 updates: decrease | Final optimization gradient / initial | Final inverted tets |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {r['id']} | {r['last_step']} | {r['initial_objective']:.6f} | "
        f"{r['final_objective']:.6f} | {r['last_10_drop_percent']:.2f}% | "
        f"{r['last_25_drop_percent']:.2f}% | {100 * r['optimizer_norm_ratio']:.2f}% | "
        f"{r['final_inverted_tets']} |"
        for r in records
    )
    lines += [
        "",
        "![Final 25 updates](../data/10-curves/final-25-updates.png)",
        "",
        "## Interpretation",
        "",
        "The original Raw6 run has six unrestricted symmetric activation components per active cell.",
        "Its lower panel uses the recorded ordinary activation-gradient RMS, normalized by its initial",
        "value. L2 is included as the original matched control. The gradient branch first inverts at",
        "update 79 and L2 at update 81; both finish with one inverted tetrahedron. They are neither",
        "converged nor valid inversion-free endpoints. Both use constant Adam learning rate 0.3.",
        "",
        "The learned-axis runs use contraction-only activation and a physical rank-one projected-gradient",
        "diagnostic. Their declared rule requires objective relative span below 0.1% over 25 updates and",
        "projected-gradient RMS below 1% of its initial value at two consecutive scheduled checks.",
        "All learned-axis branches fail both thresholds at the endpoint and remain inversion-free.",
        "They halve the learning rate after 100 updates: 0.3 for updates 1-100 and 0.15 for 101-200.",
        "The 0.075 recorded at update 200 is a next-update setting that was never used. A reduced",
        "slope after update 100 therefore cannot by itself be interpreted as convergence.",
        "",
        "The final-window plot uses exactly J[T-25] through J[T], with decrease",
        "100*(J[T-25]-J[T])/J[T-25]. All six histories are strictly decreasing; therefore this equals",
        "the max-minus-min relative span used by the learned-axis stopping rule. No smoothing or",
        "interpolation is applied. Values of 1.0 in unscheduled span rows are placeholders and are not used.",
        "",
        "Forward equilibrium solver success is separate from inverse convergence. A finite loss plateau",
        "alone would also be insufficient without stationarity and feasible geometry. These plots do not",
        "establish how much further fitting could improve, an intrinsic model-capacity limit, or mechanical stability.",
        "",
        "## Reproduction and audit",
        "",
        f"Working directory: `{GROUP}`.",
        "",
        "```bash",
        "OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \\",
        "CHERRIES_NAME='Gradient-loss inverse physics: loss and convergence curves' \\",
        "CHERRIES_TAGS='gradient-loss,inverse-physics,convergence,plots,3d' \\",
        f"{ROOT}/.venv/bin/python src/10-plot.py",
        "```",
        "",
        "Inputs and copied CSV, summary, and protocol receipts are recorded with SHA-256 hashes in",
        "[the analysis receipt](../data/10-curves/summary.json). The script checks consecutive updates,",
        "finite trace values, agreement with endpoint summaries, and independent recomputation of",
        "the final scheduled span and projected-gradient ratio. The numerical inputs are unchanged.",
        "",
        "Editable vector figures: [overview SVG](../data/10-curves/loss-and-stationarity.svg),",
        "[final-window SVG](../data/10-curves/final-25-updates.svg).",
        "",
        "The two gradient-only experiment groups are the primary evidence. The mixed L2/gradient",
        "column is included separately as context for the latest discussion; it is not gradient-only.",
        "",
    ]
    report.write_text("\n".join(lines))
    return report


def main(cfg: Config) -> None:
    logger.info("Plotting six saved fitting histories; no inverse solves are run.")
    output = cherries.output(cfg.output_dir, mkdir=True)
    assert not output.exists() or not any(output.iterdir())
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 13, "legend.fontsize": 10})
    cases = [read_case(case, output) for case in CASES]
    records = [case["record"] for case in cases]
    assert all(r["loss_increase_count"] == 0 for r in records)
    assert all(r["last_25_drop_percent"] > 0.1 for r in records)
    plot_overview(cases, output)
    plot_tail(cases, output)
    source = output / "sources" / Path(__file__).name
    source.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(__file__), source)
    receipt = {
        "scope": "Read-only replot and audit of saved 3D fitting histories; no new inverse solves.",
        "records": records,
        "source": {
            "path": str(Path(__file__).resolve()),
            "snapshot": str(source),
            "sha256": digest(source),
        },
        "runtime": {
            "python": sys.version,
            "executable": sys.executable,
            "matplotlib": mpl.__version__,
        },
        "checks_passed": True,
    }
    (output / "summary.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    )
    write_report(records)
    cherries.log_metrics(
        {f"{r['id']}/last_25_drop_percent": r["last_25_drop_percent"] for r in records}
    )
    cherries.log_metrics(
        {f"{r['id']}/final_gradient_ratio": r["optimizer_norm_ratio"] for r in records}
    )


if __name__ == "__main__":
    cherries.main(main, profile=ProfileNoCommit)
