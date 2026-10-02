"""Plot the exact saved loss and gradient histories without rerunning physics."""

import csv
import hashlib
import json
import logging
import shutil
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pydantic_settings as ps
from liblaf.cherries import core, plugins, profiles

from liblaf import cherries

mpl.use("Agg")
import matplotlib.pyplot as plt

LOG = logging.getLogger(__name__)
GROUP = Path(__file__).resolve().parents[1]
COLORS = {"x_contraction": "#b24b35", "unconstrained": "#246fa6"}
LABELS = {"x_contraction": "x contraction", "unconstrained": "free symmetric B"}


class ProfileRecord(profiles.Profile):
    def init(self):
        run = core.run
        run.plugins.register(plugins.Comet(run=run, disabled=False))
        run.plugins.register(plugins.Git(run=run, commit=False))
        run.plugins.register(plugins.Local(run=run))
        run.plugins.register(plugins.Logging(run=run))
        return run


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    input_dir: Path = Path("10-comparison-final")
    output: Path = Path("40-loss-curves")


def main(cfg: Config):  # noqa: PLR0915
    source = GROUP / "data" / cfg.input_dir
    output = cherries.output(cfg.output)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / Path(__file__).name)
    summaries = json.loads((source / "summary.json").read_text())
    records = []
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained", sharey=True)
    grad_fig, grad_axes = plt.subplots(1, 2, figsize=(11, 3.8), layout="constrained")
    for summary in summaries:
        name, mode, height = summary["name"], summary["mode"], summary["height"]
        path = source / name / "trace.csv"
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        data = {key: np.array([float(r[key]) for r in rows]) for key in rows[0]}
        loss = data["objective_normalized"]
        delta = np.diff(loss)
        rejected_path = path.parent / "rejected-trials.jsonl"
        failures = (
            len(rejected_path.read_text().splitlines()) if rejected_path.exists() else 0
        )
        count = int(summary["function_evaluations"])
        extra_trials = count - len(rows)
        record = {
            "case": name,
            "accepted_updates": len(delta),
            "total_trial_evaluations_including_initial": count,
            "accepted_loss_increases": int(np.sum(delta > 0)),
            "largest_signed_loss_change": float(delta.max()),
            "initial_normalized_loss": float(loss[0]),
            "final_normalized_loss": float(loss[-1]),
            "mse_reduction_percent": float(100 * (1 - loss[-1] / loss[0])),
            "nonaccepted_evaluations": extra_trials,
            "logged_forward_failures": failures,
            "finite_nonaccepted_evaluations_inferred": extra_trials - failures,
            "final_projected_gradient_inf": float(data["projected_gradient_inf"][-1]),
            "stop_reason": summary["optimizer_message"],
            "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        records.append(record)
        col = 0 if height == 0.05 else 1
        color = COLORS[mode]
        for row, xkey in enumerate(("step", "evaluations")):
            axis = axes[row, col]
            x = data[xkey]
            axis.plot(x, loss, color=color, lw=1.8, label=LABELS[mode])
            axis.scatter(x[-1], loss[-1], color=color, s=34, zorder=5)
            if row == 1 and count > x[-1]:
                axis.plot(
                    [x[-1], count], [loss[-1], loss[-1]], color=color, ls=":", lw=1.8
                )
            axis.grid(alpha=0.18)
        axis = grad_axes[col]
        axis.semilogy(
            data["step"],
            data["projected_gradient_inf"],
            color=color,
            label=LABELS[mode],
        )
        axis.scatter(
            data["step"][-1], data["projected_gradient_inf"][-1], color=color, s=28
        )
    for col, height in enumerate((0.05, 0.20)):
        axes[0, col].set_title(f"Target height h = {height:g}")
        axes[0, col].set_xlabel("Accepted inverse update")
        axes[1, col].set_xlabel("Cumulative forward evaluations")
        axes[0, col].legend(loc="lower left", frameon=False)
        for row in range(2):
            axes[row, col].set_ylim(0.425, 0.55)
        grad_axes[col].set_title(f"Target height h = {height:g}")
        grad_axes[col].set_xlabel("Accepted inverse update")
        grad_axes[col].axhline(1e-7, ls="--", color="0.4", label="stopping threshold")
        grad_axes[col].grid(alpha=0.18)
        grad_axes[col].legend(frameon=False, fontsize=9)
        grad_axes[col].set_ylim(5e-8, 0.5)
    for axis in axes[:, 0]:
        axis.set_ylabel("Loss = mean squared top error / h²")
    grad_axes[0].set_ylabel("Projected gradient infinity norm")
    fig.suptitle("Accepted loss decreases in all four runs", fontsize=16)
    fig.supxlabel(
        "Accepted states only: rejected trial losses were not stored. Dotted tails hold the last accepted loss.",
        fontsize=10,
    )
    grad_fig.suptitle("Low loss progress is not inverse convergence", fontsize=15)
    for name in ("accepted-loss.png", "accepted-loss.pdf"):
        fig.savefig(output / name, dpi=180)
    grad_fig.savefig(output / "projected-gradient.png", dpi=180)
    plt.close("all")
    manifest = {
        "input": str(source),
        "loss_definition": "mean(||u_top - target||²)/h², exactly objective_normalized in trace.csv",
        "scope": "accepted states only; finite rejected-trial objectives are absent from the original logs",
        "cases": records,
    }
    (output / "loss-audit.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for item in records:
        LOG.info("%s", json.dumps(item))
    cherries.log_metrics(
        {
            "accepted_loss_increases_total": sum(
                r["accepted_loss_increases"] for r in records
            )
        }
    )
    LOG.info("Wrote loss plots to %s", output)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileRecord)
