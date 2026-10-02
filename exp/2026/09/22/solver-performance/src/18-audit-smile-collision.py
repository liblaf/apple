# ruff: noqa: CPY001, E402, EM101, TRY003
"""Construct and audit the collision model used by the paired Smile fit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

from liblaf import cherries

EXPERIMENT = Path(__file__).resolve().parent.parent
SOURCE_GROUP = EXPERIMENT.parent.parent / "21/joint-activation-material-mandible"
sys.path.insert(0, str(SOURCE_GROUP / "src"))

from joint_equilibrium import configure_cuda
from joint_expression_inputs import EyeExpressionInputs
from remote_paths import install_loader_path_relocation
from smile_collision import audit_collision_state, audit_required_collision


class Config(cherries.BaseConfig):
    source_root: Path = EXPERIMENT.parents[4]
    inputs_dir: Path = SOURCE_GROUP / "data/expression-inputs-002"
    output_dir: Path = EXPERIMENT / "data/smile-collision-audit-001"


def main(cfg: Config) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    install_loader_path_relocation(source_root=cfg.source_root)
    configure_cuda()
    inputs = EyeExpressionInputs.load(cfg.inputs_dir)
    physics, _ = inputs.build_physics()
    required = audit_required_collision(physics)
    neutral = torch.as_tensor(
        inputs.arrays["neutral_displacement_m"], device=physics.points_t.device
    )
    state = audit_collision_state(
        physics,
        neutral,
        torch.zeros(6, device=neutral.device, dtype=neutral.dtype),
    )
    receipt = {
        "schema": "smile-collision-construction-audit-v1",
        "success": bool(required["success"] and state["state_feasible"]),
        "scope": "Model construction plus shared neutral geometry gate; no equilibrium solve.",
        "required": required,
        "neutral": state,
    }
    path = cfg.output_dir / "summary.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    cherries.log_metrics(
        {
            "collision/construction_success": float(required["success"]),
            "collision/neutral_feasible": float(state["state_feasible"]),
            "collision/active_pairs": state["active_contact_count"],
        }
    )
    cherries.log_output(path)
    if not receipt["success"]:
        raise RuntimeError("shared neutral fails the required collision gate")


if __name__ == "__main__":
    cherries.main(main)
