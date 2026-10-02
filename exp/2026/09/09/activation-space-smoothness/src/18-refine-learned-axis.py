"""Conditional lower-rate, longer-horizon calibration after the first grid fails."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pydantic_settings as ps
from calibration_core import Calibration, select_weight
from experiment_profile import ProfileCometNoCommit
from study_physics import configure
from study_runner import FIXTURE, GROUP, archive_runtime, digest, write_json

from liblaf import cherries

LOG = logging.getLogger(__name__)
DONE = False


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    fixture: Path = FIXTURE
    failed_calibration: Path = GROUP / "data/18-learned-axis-calibration-v2"
    revision: Path = GROUP / "docs/15-calibration-revision.md"
    output_dir: Path = GROUP / "data/18-learned-axis-calibration-refined"


def main(cfg: Config) -> None:
    global DONE
    prior = json.loads((cfg.failed_calibration / "selection-protocol.json").read_text())
    failed = json.loads((cfg.failed_calibration / "pilots.json").read_text())
    assert {r["name"] for r in failed} == {"lr-0", "lr-1", "lr-2", "lr-half-smallest"}
    assert not any(r["stable"] for r in failed), (
        "Revision requires the entire original bounded grid to fail"
    )
    assert cfg.revision.is_file(), (
        "Document the calibration revision before executing it"
    )
    controls_validation = Path(prior["controls_validation"]["path"])
    assert digest(controls_validation) == prior["controls_validation"]["sha256"]
    assert json.loads(controls_validation.read_text())["status"] == "passed"
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir()), f"Refuse to overwrite calibration: {out}"
    configure()
    started = time.perf_counter()
    matches = json.loads(
        (cfg.failed_calibration / "initial-step-matching.json").read_text()
    )
    rates = [matches[0]["learning_rate"] / factor for factor in (64, 32, 16, 8)]
    settings = {
        k: prior[k]
        for k in (
            "adam_eps",
            "betas",
            "smooth_length_m",
            "inputs",
            "smoothness_field",
            "initialization_seed",
            "initial_strength",
            "initialization_mode",
            "controls_validation",
            "plan",
            "seed_sensitivity_deferred",
        )
    }
    settings.update(
        {
            "status": "calibrating",
            "pilot_steps": 16,
            "candidate_learning_rates": rates,
            "revision": {"path": str(cfg.revision), "sha256": digest(cfg.revision)},
            "failed_initial_grid": {
                "path": str(cfg.failed_calibration),
                "pilots_sha256": digest(cfg.failed_calibration / "pilots.json"),
            },
            "initial_equivalence": json.loads(
                (cfg.failed_calibration / "initial-equivalence.json").read_text()
            ),
            "lr_rule": "Closed four-rate grid at original smallest rate /64,/32,/16,/8. Sixteen fresh updates per rate; all inner solves valid, positive fit progress, final fit loss within 1% of best post-update. Smallest rate within 5% best fractional progress. No fallback or upper expansion.",
            "weight_rule": "C-gradient ratio at selected 16-update endpoint; three fresh 16-update pilots at scale/4,scale,scale*4. Same .8 fit-progress/.9 incremental-projection retention, positive S_C reduction and total-objective stability; smallest weight within .05 best reduction.",
        }
    )
    write_json(out / "config.json", cfg.model_dump(mode="json"))
    write_json(out / "selection-protocol.json", settings)
    write_json(out / "provenance.json", archive_runtime(out, cfg.fixture))
    for name, record in settings["inputs"].items():
        assert digest(cfg.fixture / name) == record["sha256"]
    calibration = Calibration(out, cfg.fixture, settings, "learned-axis", 16)
    candidates = [
        calibration.trial(rate, 0.0, f"lr-{i}") for i, rate in enumerate(rates)
    ]
    valid = [r for r in candidates if r["stable"]]
    assert valid, "No rate passed the closed refined calibration grid"
    best_gain = max(r["fractional_progress"] for r in valid)
    selected = min(
        (r for r in valid if r["fractional_progress"] >= 0.95 * best_gain),
        key=lambda r: r["learning_rate"],
    )
    with np.load(out / selected["name"] / "endpoint.npz") as endpoint:
        gfit = float(np.linalg.norm(endpoint["fit_gradient"]))
        gsmooth = float(np.linalg.norm(endpoint["smooth_gradient"]))
    assert np.isfinite(gfit) and np.isfinite(gsmooth) and gsmooth > 0
    scale = 0.25 * gfit / gsmooth
    chosen = select_weight(calibration, selected["learning_rate"], scale, selected)
    settings.update(
        {
            "status": "frozen_before_primary_runs",
            "learning_rates": {"learned-axis": selected["learning_rate"]},
            "smoothness_weight": chosen["smoothness_weight"],
            "weight_scale": scale,
            "selected_lr_pilot": selected["name"],
            "selected_weight": chosen,
            "largest_tested_rate_selected": selected is candidates[-1],
            "elapsed_s": time.perf_counter() - started,
        }
    )
    write_json(out / "summary.json", settings)
    for path in out.glob("*.json"):
        cherries.log_output(path)
    LOG.info(
        "Frozen refined learned-axis lr %.9g and C weight %.9g",
        selected["learning_rate"],
        chosen["smoothness_weight"],
    )
    DONE = True


if __name__ == "__main__":
    cherries.main(
        main, profile=None if os.getenv("DEBUG") == "1" else ProfileCometNoCommit
    )
    if not DONE:
        raise SystemExit(1)
