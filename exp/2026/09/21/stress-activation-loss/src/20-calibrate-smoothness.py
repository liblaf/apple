"""Freeze weak tensor smoothness from one declared L2 pilot checkpoint."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pydantic_settings as ps
import torch
from activation_models import controls_from_matrix
from experiment import Profile
from run_support import archive, receipt, run_stage, verify_sources, write_json
from stress_regularization import calibrated_weight, gradient_balance
from stress_study import StressStudy

from liblaf import cherries


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output: Path = Path("20-calibration-inverse-v2")
    validation: Path = Path("12-validation-inverse-v2")
    steps: int = 8
    learning_rate: float = 0.05
    adam_eps: float = 1e-8
    target_gradient_ratio: float = 0.1


def main(cfg: Config) -> None:
    assert cfg.steps > 0
    gate = cherries.input(cfg.validation)
    assert json.loads((gate / "checks.json").read_text())["passed"]
    protocol = json.loads((gate / "protocol.json").read_text())
    verify_sources(protocol["sources"])
    out = cherries.output(cfg.output)
    out.mkdir(parents=True, exist_ok=False)
    study = StressStudy()
    p = study.physics
    study.save_geometry(out / "mesh.npz")
    control = run_stage(
        study,
        out / "pilot-l2",
        "symmetric6",
        0.0,
        0.0,
        torch.zeros((len(p.ids), 3, 3)),
        np.zeros_like(p.points),
        steps=cfg.steps,
        learning_rate=cfg.learning_rate,
        adam_eps=cfg.adam_eps,
    )
    assert control["status"] == "completed_budget_not_convergence_certified", control
    with np.load(out / "pilot-l2/last.npz", allow_pickle=False) as state:
        q, _ = controls_from_matrix(torch.as_tensor(state["Qhat"]), "symmetric6")
        seed = state["u"].copy()
    q = torch.nn.Parameter(q)
    # Both gradients are evaluated at this one re-equilibrated saved state.
    result = study.evaluate(
        q, "symmetric6", None, seed, 0.0, 0.0, component_gradients=True
    )
    mass = torch.as_tensor(study.active_weights)
    weight = calibrated_weight(
        result["l2_tensor_gradient"],
        result["regularizer_tensor_gradient"],
        mass,
        cfg.target_gradient_ratio,
    )
    balance = gradient_balance(
        result["l2_tensor_gradient"],
        result["regularizer_tensor_gradient"],
        mass,
        weight,
    )
    write_json(
        out / "calibration.json",
        {
            "schema": "smile-stress-gradient-calibration-v2",
            "passed": True,
            "selected_weight": weight,
            "target_gradient_ratio": cfg.target_gradient_ratio,
            "calibration_balance": balance,
            "gradient_metric": "Mandel tensor covectors; dual normalized effective-volume norm",
            "gradient_numerator": "L2 only; normal loss excluded",
            "calibration_state": receipt(out / "pilot-l2/last.npz"),
            "forward": result["forward"],
            "adjoint": result["adjoint"],
            "control": {"l2": control},
            "pilot_steps": cfg.steps,
            "learning_rate": cfg.learning_rate,
            "adam_eps": cfg.adam_eps,
            "validation": receipt(gate / "checks.json"),
            "sources": archive(out),
            "selection": "one fixed weight from the same nonzero-stress L2 pilot state; shared across all eight fits",
            "transfer_limit": "0.1 is a calibration target, not a promised terminal ratio or convergence condition.",
        },
    )
    cherries.log_metric("selected_smooth_weight", weight)


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
