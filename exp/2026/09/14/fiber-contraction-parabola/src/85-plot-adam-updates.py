"""Expose retained Adam loss increases and the historical learning-rate decay."""

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

GROUP = Path(__file__).resolve().parents[1]
LOG = logging.getLogger(__name__)


class ProfileAdamUpdates(profiles.Profile):
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
    output: Path = Path("85-adam-update-plots")


def main(cfg: Config):
    source = GROUP / "data" / cfg.input_dir
    output = cherries.output(cfg.output)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, output / Path(__file__).name)
    fig, axes = plt.subplots(
        2, 2, figsize=(11, 6.5), sharex="col", layout="constrained"
    )
    records = []
    for col, height in enumerate((0.05, 0.20)):
        for mode, label, color in (
            ("x_contraction", "x contraction", "#b24b35"),
            ("unconstrained", "free symmetric B", "#246fa6"),
        ):
            name = f"h{round(height * 1000):03d}-{mode}"
            path = source / name / "trace.csv"
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            steps = np.array([int(r["step"]) for r in rows])
            losses = np.array([float(r["objective_normalized"]) for r in rows])
            delta = np.diff(losses)
            increases = delta > np.maximum(1e-12, 1e-10 * np.abs(losses[:-1]))
            axes[0, col].plot(steps[1:], delta, color=color, lw=1, label=label)
            axes[0, col].scatter(
                steps[1:][increases], delta[increases], color=color, marker="x", s=20
            )
            # Each source row describes its outgoing update. Exclude any final
            # outgoing proposal without a solved successor.
            rates = np.array([float(r["update_learning_rate"]) for r in rows[:-1]])
            axes[1, col].plot(
                steps[1:],
                rates,
                color=color,
                lw=1.3,
                ls="--" if mode == "unconstrained" else "-",
                label=label,
            )
            axes[1, col].scatter(steps[-1], rates[-1], color=color, s=20)
            records.append(
                {
                    "case": name,
                    "trace_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "loss_increases": int(increases.sum()),
                    "last_solved_step": int(steps[-1]),
                    "last_applied_learning_rate": float(rates[-1]),
                }
            )
        axes[0, col].axhline(0, color="black", lw=0.8)
        axes[0, col].set(yscale="symlog", title=f"h = {height:g}")
        axes[0, col].set_yscale("symlog", linthresh=1e-9)
        axes[0, col].set_ylim(-0.01, 0.01)
        axes[1, col].set(yscale="log", ylim=(1e-7, 0.05), xlabel="Solved Adam update")
        for row in range(2):
            axes[row, col].grid(alpha=0.2)
        axes[0, col].legend(frameon=False)
    axes[0, 0].set_ylabel("Change in MSE / h²\npositive = retained loss increase")
    axes[1, 0].set_ylabel("Learning rate used for update")
    fig.suptitle("Adam: loss changes and the original 0.99-per-update decay")
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"adam-updates.{suffix}", dpi=180)
    plt.close(fig)
    (output / "audit.json").write_text(json.dumps(records, indent=2) + "\n")
    LOG.info("Wrote Adam update diagnostics: %s", json.dumps(records))


if __name__ == "__main__":
    cherries.main(main, profile=ProfileAdamUpdates)
