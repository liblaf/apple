"""Calibrate one learned-axis initialization under the approved bounded plan."""

from __future__ import annotations

import gc
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import pydantic_settings as ps
import torch
from activation_controls import learned_axis_raw6
from calibration_core import Calibration, select_weight
from experiment_profile import ProfileCometNoCommit
from study_physics import FacePhysics, configure
from study_runner import (
    FIXTURE,
    GROUP,
    Objective,
    archive_runtime,
    control_z,
    digest,
    write_json,
)

from liblaf import cherries

LOG = logging.getLogger(__name__)
DONE = False


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    fixture: Path = FIXTURE
    previous_settings: Path = GROUP / "data/15-calibration/summary.json"
    controls_validation: Path = (
        GROUP / "data/11-learned-axis-validation-v2/summary.json"
    )
    output_dir: Path = GROUP / "data/18-learned-axis-calibration-v2"
    initialization_seed: int = 20260909
    initial_strength: float = 0.001


def main(cfg: Config) -> None:
    global DONE
    old = json.loads(cfg.previous_settings.read_text())
    assert old["status"] == "frozen_before_primary_runs"
    assert json.loads(cfg.controls_validation.read_text())["status"] == "passed"
    assert cfg.initialization_seed == 20260909 and cfg.initial_strength == 0.001
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir()), f"Refuse to overwrite calibration: {out}"
    configure()
    started = time.perf_counter()
    settings = {
        "status": "calibrating",
        "adam_eps": old["adam_eps"],
        "betas": old["betas"],
        "smooth_length_m": 0.005,
        "inputs": old["inputs"],
        "smoothness_field": "C",
        "initialization_seed": cfg.initialization_seed,
        "initial_strength": cfg.initial_strength,
        "initialization_mode": "seeded_axis_fresh_adam",
        "pilot_steps": 8,
        "controls_validation": {
            "path": str(cfg.controls_validation),
            "sha256": digest(cfg.controls_validation),
        },
        "plan": {
            "path": str(GROUP / "docs/12-learned-axis-plan.md"),
            "sha256": digest(GROUP / "docs/12-learned-axis-plan.md"),
        },
        "previous_settings": {
            "path": str(cfg.previous_settings),
            "sha256": digest(cfg.previous_settings),
        },
        "lr_rule": "Three prior Delta Z targets, 8 updates each; smallest rate within 5% best positive stable fractional fit progress; one half-smallest fallback only if none pass; no upper expansion.",
        "weight_rule": "Gradient ratio scale at selected endpoint; scale/4, scale, scale*4 with fresh 8 updates. Successful stable objective, fit-progress retention>=.8, incremental projection retention>=.9, positive S_C reduction. Smallest weight within .05 of best reduction.",
        "seed_sensitivity_deferred": True,
    }
    write_json(out / "config.json", cfg.model_dump(mode="json"))
    write_json(out / "selection-protocol.json", settings)
    write_json(out / "provenance.json", archive_runtime(out, cfg.fixture))
    for name, record in old["inputs"].items():
        assert digest(cfg.fixture / name) == record["sha256"]
    calibration = Calibration(out, cfg.fixture, settings, "learned-axis", 8)
    physics = FacePhysics(cfg.fixture, activation_model="learned-axis", skin_factor=0.0)
    q, seed = calibration.initialize(physics)
    initial = Objective(physics, smoothness_field="C")(
        q, seed, component_gradients=True
    )
    assert initial["smoothness_C"] == 0 and not torch.count_nonzero(
        initial["smooth_gradient"]
    )
    controls = q.detach().clone()
    gradient = initial["fit_gradient"].clone()
    mass = torch.as_tensor(physics.volumes / physics.volumes.sum())
    del physics, q
    gc.collect()
    torch.cuda.empty_cache()
    mapped_physics = FacePhysics(cfg.fixture, activation_model="raw6", skin_factor=0.0)
    mapped_q = torch.nn.Parameter(learned_axis_raw6(controls))
    mapped = Objective(mapped_physics, smoothness_field="C")(mapped_q, seed)
    v = controls.clone().requires_grad_()
    predicted = torch.autograd.grad((learned_axis_raw6(v) * mapped_q.grad).sum(), v)[0]
    relative_gradient_error = float((predicted - gradient).norm() / gradient.norm())
    geometry_error = float(
        1000 * np.linalg.norm(mapped["u"] - initial["u"]) / np.sqrt(len(seed))
    )
    assert relative_gradient_error < 0.005 and geometry_error < 0.001
    checks = {
        "learned_axis_chain_relative_error": relative_gradient_error,
        "geometry_rms_difference_mm": geometry_error,
        "Z_max_abs_difference": float(np.max(np.abs(mapped["Z"] - initial["Z"]))),
        "initial_fit_rms_mm": float(np.sqrt(3 * initial["objective_mm2"])),
        "initial_smoothness_C": initial["smoothness_C"],
        "axis_forward": initial["forward"],
        "axis_adjoint": initial["adjoint"],
        "mapped_raw6_forward": mapped["forward"],
        "mapped_raw6_adjoint": mapped["adjoint"],
    }
    assert checks["Z_max_abs_difference"] < 1e-12
    write_json(out / "initial-equivalence.json", checks)
    del mapped_physics, mapped_q, mapped, v, predicted, initial
    gc.collect()
    torch.cuda.empty_cache()

    direction = -gradient / (gradient.abs() + settings["adam_eps"])
    z0 = control_z(controls, "learned-axis")

    def physical_delta(lr):
        delta = control_z(controls + lr * direction, "learned-axis") - z0
        rms = float((mass * delta.square().sum((-1, -2))).sum().sqrt())
        maximum = float(delta.square().sum((-1, -2)).max().sqrt())
        return rms, maximum

    matches = []
    for target in (0.01881737, 0.05669795, 0.19279357):
        lo, hi, previous = 0.0, 1e-6, 0.0
        while True:
            actual, _ = physical_delta(hi)
            assert actual >= previous, (
                "Physical step match left its first increasing branch"
            )
            if actual >= target:
                break
            lo, hi, previous = hi, hi * 1.4, actual
            assert hi < 1e8
        for _ in range(48):
            mid = (lo + hi) / 2
            if physical_delta(mid)[0] < target:
                lo = mid
            else:
                hi = mid
        lr = (lo + hi) / 2
        actual, maximum = physical_delta(lr)
        assert abs(actual / target - 1) < 1e-10
        matches.append(
            {
                "target_delta_Z_rms": target,
                "learning_rate": lr,
                "actual_delta_Z_rms": actual,
                "maximum_delta_Z_Frobenius": maximum,
            }
        )
    write_json(out / "initial-step-matching.json", matches)
    del gradient
    gc.collect()
    torch.cuda.empty_cache()
    rates = [
        calibration.trial(m["learning_rate"], 0.0, f"lr-{i}")
        for i, m in enumerate(matches)
    ]
    valid = [r for r in rates if r["stable"]]
    if not valid:
        rates.append(
            calibration.trial(matches[0]["learning_rate"] / 2, 0.0, "lr-half-smallest")
        )
        valid = [r for r in rates if r["stable"]]
    assert valid, "No rate passed the bounded learned-axis calibration"
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
            "largest_tested_rate_selected": selected is rates[2],
            "initial_equivalence": checks,
            "elapsed_s": time.perf_counter() - started,
        }
    )
    write_json(out / "summary.json", settings)
    for path in out.glob("*.json"):
        cherries.log_output(path)
    LOG.info(
        "Frozen learned-axis lr %.9g and C weight %.9g",
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
