"""Replay the fixed parabolic protocol while recording every outer trial.

The replay imports the production inverse script and replaces only its imported
``minimize`` binding with a transparent logger.  The optimizer, objective,
forward solve, acceptance test, and controls are otherwise unchanged.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
from liblaf.cherries import core, plugins, profiles

from liblaf import cherries

LOG = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[6]
SOURCE = Path(__file__).with_name("10-run-inverse.py")
ORIGINAL_DIR = (
    ROOT / "exp/2026/09/14/fiber-contraction-parabola/data/10-comparison-final"
)


class ProfileTrialLossReplay(profiles.Profile):
    """Normal Comet recording with Git state recorded but never committed."""

    def init(self):
        run = core.run
        run.plugins.register(plugins.Comet(run=run, disabled=False))
        run.plugins.register(plugins.Git(run=run, commit=False))
        run.plugins.register(plugins.Local(run=run))
        run.plugins.register(plugins.Logging(run=run))
        return run


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    original_dir: Path = ORIGINAL_DIR
    output_dir: Path = cherries.output("50-trial-loss-replay", mkdir=True)


def load_original():
    spec = importlib.util.spec_from_file_location("inverse_replay_source", SOURCE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_protocol(original_dir: Path) -> dict[str, Any]:
    protocol = json.loads((original_dir / "protocol.json").read_text())
    expected = protocol["source_sha256"]
    actual = {
        relative: file_hash(ROOT / relative)
        for relative in (
            "exp/2026/09/14/fiber-contraction-parabola/src/10-run-inverse.py",
            "exp/2026/09/14/fiber-contraction-parabola/src/optimizer.py",
            "exp/2026/09/14/fiber-contraction-parabola/src/physics2d.py",
        )
    }
    for relative, digest in actual.items():
        assert digest == expected[relative], (relative, digest, expected[relative])
    return {"protocol": protocol, "checked_source_sha256": actual}


def replay_case(  # noqa: PLR0915
    module: Any, cfg: Any, mesh: Any, mode: str, height: float, output: Path
):
    """Run original ``run_case`` and emit a row for each optimizer evaluation."""
    trial_rows: list[dict[str, Any]] = []
    incumbent_x: np.ndarray | None = None
    incumbent_loss: float | None = None
    last_trial: dict[str, Any] | None = None
    original_minimize = module.minimize

    def evaluated(q: np.ndarray):
        nonlocal incumbent_loss, incumbent_x, last_trial
        row: dict[str, Any] = {
            "evaluation_id": len(trial_rows),
            "incumbent_normalized_loss": incumbent_loss,
            "trial_normalized_loss": None,
            "forward_failure": "",
            "accepted": False,
            "actual_update_inf": None,
            "trial_gradient_inf": None,
        }
        try:
            value, gradient = evaluate_original(q)
        except Exception as exc:
            row["forward_failure"] = repr(exc)
            trial_rows.append(row)
            last_trial = row
            raise
        row["trial_normalized_loss"] = float(value)
        row["trial_gradient_inf"] = float(np.linalg.norm(gradient, np.inf))
        if incumbent_x is None:
            # ``run_case`` records q0 before minimize.  The first optimizer
            # evaluation reuses that cached state, so it is the accepted
            # initial incumbent, not a rejected trial.
            incumbent_x = q.copy()
            incumbent_loss = float(value)
            row["accepted"] = True
            row["actual_update_inf"] = 0.0
        else:
            row["actual_update_inf"] = float(np.linalg.norm(q - incumbent_x, np.inf))
        trial_rows.append(row)
        last_trial = row
        return value, gradient

    def accepted(q: np.ndarray):
        nonlocal incumbent_x, incumbent_loss
        assert last_trial is not None
        assert last_trial["trial_normalized_loss"] is not None
        last_trial["accepted"] = True
        result = callback_original(q)
        incumbent_x = q.copy()
        incumbent_loss = float(last_trial["trial_normalized_loss"])
        return result

    def rejected(iteration: int, backtrack: int, alpha: float, reason: str):
        assert last_trial is not None
        last_trial["outer_iteration"] = iteration
        last_trial["backtrack"] = backtrack
        last_trial["alpha"] = alpha
        last_trial["forward_failure"] = reason
        return reject_original(iteration, backtrack, alpha, reason)

    def logged_minimize(
        evaluate: Any,
        initial: Any,
        callback: Any,
        reject: Any,
        **kwargs: Any,
    ):
        nonlocal evaluate_original, callback_original, reject_original
        evaluate_original = evaluate
        callback_original = callback
        reject_original = reject
        return original_minimize(evaluated, initial, accepted, rejected, **kwargs)

    evaluate_original: Any
    callback_original: Any
    reject_original: Any
    module.minimize = logged_minimize
    try:
        summary = module.run_case(cfg, mesh, mode, height, output)
    finally:
        module.minimize = original_minimize
    for row in trial_rows:
        row.setdefault("outer_iteration", None)
        row.setdefault("backtrack", None)
        row.setdefault("alpha", None)
    name = f"h{round(height * 1000):03d}-{mode}"
    path = output / name / "trial-loss.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=trial_rows[0].keys())
        writer.writeheader()
        writer.writerows(trial_rows)
    return summary, trial_rows


def compare_case(original_dir: Path, replay_dir: Path, name: str) -> dict[str, Any]:
    original = np.load(original_dir / name / "history.npz", allow_pickle=False)
    replay = np.load(replay_dir / name / "history.npz", allow_pickle=False)
    comparison = {}
    for key in ("u", "controls"):
        difference = np.max(np.abs(original[key] - replay[key]))
        comparison[f"{key}_max_abs_difference"] = float(difference)
        assert difference <= 1e-9, (name, key, difference)
    comparison["trajectory_match"] = True
    return comparison


def main(cfg: Config) -> None:
    output = cfg.output_dir
    output.mkdir(parents=True, exist_ok=False)
    provenance = checked_protocol(cfg.original_dir)
    module = load_original()
    original_config = module.Config(**provenance["protocol"]["config"])
    mesh = module.ph.build_mesh(original_config.nx, original_config.ny)
    shutil.copy2(SOURCE, output / SOURCE.name)
    shutil.copy2(Path(__file__), output / Path(__file__).name)
    results = {}
    all_trials = []
    for height in map(float, original_config.heights.split(",")):
        for mode in original_config.modes.split(","):
            summary, trials = replay_case(
                module, original_config, mesh, mode, height, output
            )
            name = f"h{round(height * 1000):03d}-{mode}"
            comparison = compare_case(cfg.original_dir, output, name)
            results[name] = {"summary": summary, "comparison": comparison}
            all_trials.extend({"case": name, **row} for row in trials)
            cherries.log_metrics(
                {
                    f"{name}/trial_count": len(trials),
                    f"{name}/accepted_trial_count": sum(
                        row["accepted"] for row in trials
                    ),
                    f"{name}/finite_uphill_trials": sum(
                        row["trial_normalized_loss"] is not None
                        and row["incumbent_normalized_loss"] is not None
                        and row["trial_normalized_loss"]
                        > row["incumbent_normalized_loss"]
                        for row in trials
                    ),
                }
            )
    with (output / "trial-loss.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_trials[0].keys())
        writer.writeheader()
        writer.writerows(all_trials)
    summary = {**provenance, "cases": results}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    LOG.info("Replayed %d cases to %s", len(results), output)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileTrialLossReplay)
