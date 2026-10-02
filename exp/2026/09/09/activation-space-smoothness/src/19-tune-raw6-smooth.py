"""Calibrate C smoothing against the completed corrected Raw6 control."""

from __future__ import annotations

import csv
import gc
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pydantic_settings as ps
import torch
from activation_controls import control_c
from calibration_core import Calibration, select_weight
from experiment_profile import ProfileCometNoCommit
from study_physics import FacePhysics, configure
from study_runner import (
    FIXTURE,
    GROUP,
    ROOT,
    Objective,
    archive_runtime,
    digest,
    write_json,
)

from liblaf import cherries

LOG = logging.getLogger(__name__)
DONE = False
OLD_DATA = ROOT / "exp/2026/09/08/local-skin-prestrain/data"


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    fixture: Path = FIXTURE
    previous_settings: Path = GROUP / "data/15-calibration/summary.json"
    controls_validation: Path = (
        GROUP / "data/11-learned-axis-validation-v2/summary.json"
    )
    output_dir: Path = GROUP / "data/19-raw6-smooth-calibration"
    archive: Path = OLD_DATA / "20-forward-no-skin/final.npz"
    reference: Path = OLD_DATA / "30-refit-no-skin"


def main(cfg: Config) -> None:
    global DONE
    old = json.loads(cfg.previous_settings.read_text())
    assert old["status"] == "frozen_before_primary_runs"
    assert json.loads(cfg.controls_validation.read_text())["status"] == "passed"
    assert (
        digest(cfg.archive)
        == "9fb5c34a361328c4b90ef32983dc2f77a6fae3d1ac813ee08016142e7fadab13"
    )
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir()), f"Refuse to overwrite calibration: {out}"
    configure()
    started = time.perf_counter()
    settings = {
        "status": "calibrating",
        "adam_eps": 0.01,
        "betas": [0.9, 0.999],
        "smooth_length_m": 0.005,
        "inputs": old["inputs"],
        "smoothness_field": "C",
        "initialization_mode": "archived_controls_and_seed_fresh_adam",
        "archive_initialization": {
            "path": str(cfg.archive),
            "sha256": digest(cfg.archive),
        },
        "controls_validation": {
            "path": str(cfg.controls_validation),
            "sha256": digest(cfg.controls_validation),
        },
        "reference": {
            "path": str(cfg.reference),
            "provenance_sha256": digest(cfg.reference / "provenance.json"),
            "trace_sha256": digest(cfg.reference / "trace.csv"),
        },
        "plan": {
            "path": str(GROUP / "docs/12-learned-axis-plan.md"),
            "sha256": digest(GROUP / "docs/12-learned-axis-plan.md"),
        },
        "pilot_steps": 10,
        "weight_rule": "Gradient ratio at exact archived initial controls; test scale/4,scale,scale*4 with 10 fresh updates. Compare the recorded off trajectory using positive fit/projection progress and S_C reference denominators; same acceptance and tie rules as learned-axis.",
        "representation_limit": "C=B-I uses Raw6's recorded B representative; B signs can change while preserving BB^T. Track B eigenvalues and S_Z; do not equate this weight with the learned-axis weight.",
    }
    write_json(out / "config.json", cfg.model_dump(mode="json"))
    write_json(out / "selection-protocol.json", settings)
    write_json(out / "provenance.json", archive_runtime(out, cfg.fixture))
    for name, record in old["inputs"].items():
        assert digest(cfg.fixture / name) == record["sha256"]
    with (cfg.reference / "trace.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    row0 = next(r for r in rows if int(r["step"]) == 0)
    row10 = next(r for r in rows if int(r["step"]) == 10)
    calibration = Calibration(out, cfg.fixture, settings, "raw6", 10)
    physics = FacePhysics(cfg.fixture, activation_model="raw6", skin_factor=0.0)
    q, seed = calibration.initialize(physics)
    with np.load(cfg.reference / "step-0000.npz") as state:
        assert np.array_equal(q.detach().cpu().numpy(), state["q"])
        archived_u = state["u"].copy()
    objective = Objective(physics, smoothness_field="C")
    initial = objective(q, seed, component_gradients=True)
    measured0 = calibration.diagnostics.evaluate(initial["u"], initial["Z"])
    assert abs(measured0["fit_rms_mm"] - float(row0["fit_rms_mm"])) < 1e-5
    geometry_error = float(
        1000 * np.linalg.norm(initial["u"] - archived_u) / np.sqrt(len(seed))
    )
    assert geometry_error < 1e-5
    with np.load(cfg.reference / "step-0010.npz") as state:
        c10 = control_c(torch.as_tensor(state["q"]), "raw6")
        smooth10 = float(objective.penalty(c10))
    reference = {
        "name": "archived-raw6-off-first-10",
        "initial": {
            "objective_mm2": float(row0["objective_mm2"]),
            "target_projection": float(row0["target_projection"]),
            "smoothness_C": initial["smoothness_C"],
        },
        "final": {
            "objective_mm2": float(row10["objective_mm2"]),
            "target_projection": float(row10["target_projection"]),
            "smoothness_C": smooth10,
        },
        "progress": float(row0["objective_mm2"]) - float(row10["objective_mm2"]),
    }
    assert reference["progress"] > 0
    assert (
        reference["final"]["target_projection"]
        > reference["initial"]["target_projection"]
    )
    assert smooth10 > 0
    gsmooth = float(initial["smooth_gradient"].norm())
    gfit = float(initial["fit_gradient"].norm())
    assert np.isfinite(gsmooth) and np.isfinite(gfit) and gsmooth > 0
    scale = 0.25 * gfit / gsmooth
    checks = {
        "exact_initial_control_identity": True,
        "fresh_moments_required": True,
        "reequilibrated_geometry_rms_difference_mm": geometry_error,
        "fit_rms_difference_mm": measured0["fit_rms_mm"] - float(row0["fit_rms_mm"]),
        "initial_forward": initial["forward"],
        "initial_adjoint": initial["adjoint"],
        "weight_scale": scale,
        "reference": reference,
    }
    write_json(out / "initial-equivalence.json", checks)
    del physics, objective, q, initial, c10
    gc.collect()
    torch.cuda.empty_cache()
    chosen = select_weight(calibration, 0.3, scale, reference)
    settings.update(
        {
            "status": "frozen_before_primary_runs",
            "learning_rates": {"raw6": 0.3},
            "smoothness_weight": chosen["smoothness_weight"],
            "weight_scale": scale,
            "selected_weight": chosen,
            "initial_equivalence": checks,
            "elapsed_s": time.perf_counter() - started,
        }
    )
    write_json(out / "summary.json", settings)
    for path in out.glob("*.json"):
        cherries.log_output(path)
    LOG.info("Frozen Raw6 C weight %.9g at recorded lr .3", chosen["smoothness_weight"])
    DONE = True


if __name__ == "__main__":
    cherries.main(
        main, profile=None if os.getenv("DEBUG") == "1" else ProfileCometNoCommit
    )
    if not DONE:
        raise SystemExit(1)
