"""Compare all logged trial losses with the accepted incumbent in an exact replay."""

import csv
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


class ProfileTrialPlots(profiles.Profile):
    def init(self):
        run = core.run
        run.plugins.register(plugins.Comet(run=run, disabled=False))
        run.plugins.register(plugins.Git(run=run, commit=False))
        run.plugins.register(plugins.Local(run=run))
        run.plugins.register(plugins.Logging(run=run))
        return run


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    input_dir: Path = Path("50-trial-loss-replay")
    output: Path = Path("60-trial-loss-plots")


def load_case(path: Path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    trial = np.array(
        [
            float(r["trial_normalized_loss"]) if r["trial_normalized_loss"] else np.nan
            for r in rows
        ]
    )
    before = np.array(
        [
            float(r["incumbent_normalized_loss"])
            if r["incumbent_normalized_loss"]
            else np.nan
            for r in rows
        ]
    )
    accepted = np.array([r["accepted"].lower() == "true" for r in rows])
    failure = np.array([bool(r["forward_failure"]) for r in rows])
    assert accepted[0], (
        "Initial cached evaluation must be classified as initial accepted state"
    )
    assert np.isfinite(trial[0])
    before[0] = trial[0]
    incumbent = before.copy()
    incumbent[accepted] = trial[accepted]
    assert np.all(np.diff(incumbent) <= 1e-12)
    assert np.all(np.isfinite(incumbent))
    assert np.all(np.isfinite(trial) != failure)
    return trial, before, accepted, failure, incumbent


def main(cfg: Config):  # noqa: PLR0915
    source = GROUP / "data" / cfg.input_dir
    replay = json.loads((source / "summary.json").read_text())
    assert all(
        item["comparison"]["trajectory_match"] for item in replay["cases"].values()
    )
    output = cherries.output(cfg.output)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / Path(__file__).name)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    delta_fig, delta_axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    records = []
    for name, item in replay["cases"].items():
        mode, height = item["summary"]["mode"], item["summary"]["height"]
        row, col = (0 if height == 0.05 else 1), (0 if mode == "x_contraction" else 1)
        axis, delta_axis = axes[row, col], delta_axes[row, col]
        trial, before, accepted, failure, incumbent = load_case(
            source / name / "trial-loss.csv"
        )
        finite = np.isfinite(trial)
        x = np.arange(len(trial))
        rise = trial - before
        tolerance = np.maximum(1e-12, 1e-10 * np.abs(before))
        uphill = finite & (rise > tolerance)
        assert not np.any(uphill & accepted)
        finite_rejected = finite & ~accepted
        title = (
            f"h = {height:g}  |  {'x contraction' if col == 0 else 'free symmetric B'}"
        )
        axis.plot(
            x[finite],
            trial[finite],
            color="#b8c2ca",
            lw=0.6,
            marker=".",
            ms=2,
            label="finite trial loss",
            zorder=1,
        )
        axis.step(
            x,
            incumbent,
            where="post",
            color="#1d2730",
            lw=1.6,
            label="accepted loss",
            zorder=3,
        )
        axis.scatter(
            x[uphill],
            trial[uphill],
            color="#c74727",
            s=13,
            marker="o",
            label="uphill trial (rejected)",
            zorder=4,
        )
        if np.any(failure):
            axis.plot(
                x[failure],
                np.full(np.sum(failure), 0.025),
                transform=axis.get_xaxis_transform(),
                marker="x",
                color="#5d79aa",
                ls="none",
                ms=4,
                label="forward failure; no loss",
            )
        axis.set_title(title)
        axis.text(
            0.98,
            0.94,
            f"{np.sum(uphill)} uphill trials / {np.sum(failure)} forward failures",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85},
        )
        delta_axis.axhline(0, color="0.4", lw=0.8)
        delta_axis.scatter(
            x[finite & ~uphill],
            rise[finite & ~uphill],
            color="#276f88",
            s=7,
            label="non-uphill trial",
        )
        delta_axis.scatter(
            x[uphill], rise[uphill], color="#c74727", s=9, label="uphill trial"
        )
        delta_axis.set_yscale("symlog", linthresh=1e-8)
        delta_axis.set_title(title)
        for ax in (axis, delta_axis):
            ax.set_xlabel("Forward evaluation index (initial = 0)")
            ax.grid(alpha=0.15)
        axis.set_ylabel("Loss = mean squared top error / h²")
        delta_axis.set_ylabel("Trial loss - current accepted loss")
        max_rise = float(np.nanmax(rise))
        records.append(
            {
                "case": name,
                "trajectory_match": True,
                "total_evaluations": len(trial),
                "accepted_states_including_initial": int(np.sum(accepted)),
                "forward_failures": int(np.sum(failure)),
                "finite_rejected_trials": int(np.sum(finite_rejected)),
                "uphill_trials_above_tolerance": int(np.sum(uphill)),
                "uphill_test": "trial_loss - incumbent_loss > max(1e-12, 1e-10*abs(incumbent_loss))",
                "accepted_uphill_trials": int(np.sum(uphill & accepted)),
                "largest_trial_loss_increase": max_rise,
                "largest_relative_trial_loss_increase_percent": float(
                    np.nanmax(100 * rise / before)
                ),
                "maximum_trial_loss": float(np.nanmax(trial)),
                "final_accepted_loss": float(incumbent[-1]),
            }
        )
    common_limits = (
        min(axis.get_ylim()[0] for axis in axes.flat),
        max(axis.get_ylim()[1] for axis in axes.flat),
    )
    for axis in axes.flat:
        axis.set_ylim(*common_limits)
    handles, labels = axes[0, 1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=10)
    delta_fig.legend(
        *delta_axes[0, 1].get_legend_handles_labels(),
        loc="outside lower center",
        ncol=2,
    )
    fig.suptitle("Trial steps can overshoot; the line search rejects them", fontsize=16)
    delta_fig.suptitle(
        "Positive values show trial overshoot (symlog scale)", fontsize=16
    )
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"trial-loss.{suffix}", dpi=180)
    delta_fig.savefig(output / "trial-loss-change.png", dpi=180)
    plt.close("all")
    (output / "trial-audit.json").write_text(json.dumps(records, indent=2) + "\n")
    for record in records:
        LOG.info("%s", json.dumps(record))
    cherries.log_metrics(
        {
            "uphill_trials": sum(r["uphill_trials_above_tolerance"] for r in records),
            "accepted_uphill_trials": sum(r["accepted_uphill_trials"] for r in records),
        }
    )
    LOG.info("Wrote all-trial loss plots to %s", output)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileTrialPlots)
