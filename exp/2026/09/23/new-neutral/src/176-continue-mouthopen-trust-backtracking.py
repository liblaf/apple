"""Continue audited run015 with certified rejection outside the joint trust region."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Literal

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "mouthopen_trust_backtracking_continuation",
    GROUP / "src/163-continue-mouthopen-attainable-descent.py",
)
assert SPEC is not None
assert SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)

from joint_common import ProfileJoint, sha256  # noqa: E402


class Config(runner.Config):
    output_dir: Path = GROUP / "data/inverse-mouthopen-coupled-016"
    initialization_checkpoint: Path | None = (
        GROUP / "data/inverse-mouthopen-coupled-015/checkpoint.pt"
    )
    initialization_refinement: Path | None = None
    projection_bounded_joint: bool = True
    projection_affine_residual: bool = False
    project_pose_at_inversion_limit: bool = False
    projection_feasible_witness: bool = False
    projection_witness_policy: Literal["zero_pose", "attainable_optimum"] = "zero_pose"
    adjoint_relative_shift: float = 1e-5
    predictor_relative_shift: float = 1e-5
    projection_probe_epsilon: float = 5e-5
    initial_trial_alpha: float = 0.5
    gradient_check_epsilons: tuple[float, ...] = (1e-4, 1e-5, 1e-6, 3e-7, 1e-7)


def main(cfg: Config) -> None:
    source = GROUP / "data/inverse-mouthopen-coupled-015"
    assert cfg.initialization_checkpoint == source / "checkpoint.pt"
    assert cfg.initialization_refinement is None
    assert cfg.continue_optimizer_state
    assert not cfg.resume
    assert cfg.forward_atol == 1e-8
    assert cfg.internal_forward_atol == 1e-9
    assert cfg.maximum_inverted_tetrahedra == 100
    assert cfg.maximum_inverted_rest_volume_fraction == 1e-4
    assert cfg.learning_rate == 0.002
    assert cfg.pose_learning_rate == 0.1
    assert cfg.max_rotation_increment_deg is None
    assert cfg.max_translation_increment_m is None
    assert cfg.projection_bounded_joint
    assert not cfg.projection_affine_residual
    audit = json.loads((source / "independent-audit.json").read_text())
    assert audit["valid_forward"]
    for item in audit["inputs"].values():
        assert sha256(Path(item["path"])) == item["sha256"]
    summary = json.loads((source / "summary.json").read_text())
    assert summary["final"]["optimizer_steps"] == {"q": 304, "pose": 304}
    runner.main(cfg)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
