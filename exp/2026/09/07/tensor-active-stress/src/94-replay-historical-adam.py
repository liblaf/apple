"""Replay the exact recorded historical Adam rate without recalibration."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
from experiment_profile import ProfileCometNoCommit

from liblaf import cherries

SPEC = importlib.util.spec_from_file_location(
    "frozen_optimizer_audit", Path(__file__).with_name("90-audit-repeatability.py")
)
assert SPEC is not None
assert SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = AUDIT
SPEC.loader.exec_module(AUDIT)
COMPLETED = False


class Config(AUDIT.Config):
    mode: str = "fixed-replay"
    output_dir: Path = AUDIT.ROOT / "data/94-fixed-replay"


def main(cfg: Config) -> None:
    global COMPLETED  # noqa: PLW0603
    assert cfg.mode == "fixed-replay"
    assert cfg.resume.resolve() == AUDIT.STEP64.resolve()
    output = AUDIT.prepare_output(cfg.output_dir)
    state = AUDIT.load_checkpoint(cfg.resume.resolve(), cfg)
    AUDIT.write_json(output / "config.json", cfg.model_dump(mode="json"))
    provenance = AUDIT.BASE.archive(output, cfg)
    AUDIT.verify_parent_sources(AUDIT.parent_provenance(cfg), provenance)
    historical = AUDIT.read_json(cfg.frozen_calibration_summary.resolve())
    with np.load(cfg.frozen_gradient.resolve(), allow_pickle=False) as saved:
        q = saved["q"].copy()
        u = saved["u"].copy()
        gradient = saved["gradient"].copy()
        step = int(saved["step"])
    assert step == 64
    assert np.array_equal(q, state["q"].numpy())
    assert AUDIT.array_sha256(gradient) == historical["gradient_sha256"]
    qref = cfg.stress_reference_mpa
    maximum = cfg.stress_cap_mpa / qref
    g = torch.from_numpy(gradient)
    settings = {
        "baseline": {"learning_rate": 0.3, "adam_eps": 0.01},
        "selected": historical["selected"],
    }
    replay = {}
    for arm, setting in settings.items():
        lr, epsilon = setting["learning_rate"], setting["adam_eps"]
        closed = AUDIT.closed_form_trial(
            state["q"], AUDIT.next_adam_direction(state, g, epsilon), lr, maximum, qref
        )
        installed = [
            AUDIT.installed_trial(state, g, lr, epsilon, maximum, qref)
            for _ in range(2)
        ]
        agreement = AUDIT.installed_agreement(closed, installed[0], qref)
        repeatability = AUDIT.installed_repeatability(*installed)
        assert agreement["passed"]
        assert repeatability["passed"]
        replay[arm] = closed
        replay[f"{arm}_installed_agreement"] = agreement
        replay[f"{arm}_installed_repeatability"] = repeatability
    physical = replay["selected"]["metrics"]["physical_update_rms_mpa"]
    recorded_physical = historical["selected"]["physical_update_rms_mpa"]
    assert abs(physical / recorded_physical - 1) < 1e-5
    with np.load(cfg.rejected_trial.resolve(), allow_pickle=False) as saved:
        rejected_delta_q = saved["q"].copy() - q
    rejected_delta_Q = qref * AUDIT.matrices(torch.from_numpy(rejected_delta_q)).numpy()
    np.savez_compressed(
        output / "fixed-replay.npz",
        q=q,
        u=u,
        gradient=gradient,
        step=np.asarray(64),
        baseline_delta_q=replay["baseline"]["delta_q"],
        baseline_delta_Q=replay["baseline"]["delta_Q"],
        selected_delta_q=replay["selected"]["delta_q"],
        selected_delta_Q=replay["selected"]["delta_Q"],
        rejected_delta_q=rejected_delta_q,
        rejected_delta_Q=rejected_delta_Q,
    )
    replay.update(
        {
            "gradient_sha256": AUDIT.array_sha256(gradient),
            "physical_step_multiplier": historical["physical_step_multiplier"],
            "target_physical_update_rms_mpa": historical["physical_step_multiplier"]
            * replay["baseline"]["metrics"]["physical_update_rms_mpa"],
            "baseline_vs_candidate": AUDIT.tensor_difference(
                replay["baseline"]["delta_Q"], replay["selected"]["delta_Q"]
            ),
            "bisection_trials": [],
            "passed": True,
        }
    )
    summary = {
        "schema_version": 1,
        "status": "passed",
        "mode": "fixed_gradient_replay_step64",
        "checkpoint": AUDIT.checkpoint_receipt(cfg.resume.resolve(), state),
        "cached_gradient": AUDIT.record(cfg.frozen_gradient.resolve()),
        "old_calibration": AUDIT.record(cfg.frozen_calibration_summary.resolve()),
        "rejected_trial": AUDIT.record(cfg.rejected_trial.resolve()),
        "replay": AUDIT.calibration_receipt(replay),
        "closed_form_selected_vs_rejected_trial": AUDIT.tensor_difference(
            replay["selected"]["delta_Q"], rejected_delta_Q
        ),
        "arrays": AUDIT.record(output / "fixed-replay.npz"),
        "config": AUDIT.record(output / "config.json"),
        "provenance": provenance,
        "provenance_file": AUDIT.record(output / "provenance.json"),
        "scope": "Exact historical rate from source73 and serialized gradient only; no forward or adjoint solve, no recalibration, no threshold change.",
        "implementation_correction": {
            "failed_source90_run": str(AUDIT.ROOT / "data/90-fixed-replay"),
            "failed_terminal": AUDIT.record(
                AUDIT.ROOT / "logs/90-fixed-replay-terminal.log"
            ),
            "cause": "Source90 fixed-replay incorrectly re-bisected the already recorded historical rate; numerical recalibration differed from that rate by 1.158241502e-10 and failed its extra 1e-10 rate comparison.",
            "action": "Replay the exact recorded rate 0.0033136508893221615. Source90 and its failed artifacts remain unchanged.",
            "recorded_physical_update_relative_error": abs(
                physical / recorded_physical - 1
            ),
        },
    }
    AUDIT.write_json(output / "summary.json", summary)
    for name in ("summary.json", "fixed-replay.npz", "config.json", "provenance.json"):
        cherries.log_output(output / name)
    print(
        json.dumps(
            {
                "status": summary["status"],
                "same_gradient": summary["replay"]["selected_installed_agreement"],
                "old_rejected_trial_difference": summary[
                    "closed_form_selected_vs_rejected_trial"
                ],
            }
        ),
        flush=True,
    )
    COMPLETED = True


if __name__ == "__main__":
    cherries.main(main, profile=ProfileCometNoCommit)
    if not COMPLETED:
        raise SystemExit(1)
