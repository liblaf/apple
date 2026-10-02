"""Resume audited run017 with accurately solved shifted tangents before cutoff."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "mouthopen_accurate_tangent_continuation",
    GROUP / "src/163-continue-mouthopen-attainable-descent.py",
)
assert SPEC is not None
assert SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)

from joint_common import ProfileJoint, sha256  # noqa: E402


class Config(runner.Config):
    output_dir: Path = GROUP / "data/inverse-mouthopen-coupled-018"
    initialization_checkpoint: Path | None = (
        GROUP / "data/inverse-mouthopen-coupled-017/checkpoint.pt"
    )
    initialization_refinement: Path | None = None
    projection_bounded_joint: bool = True
    projection_affine_residual: bool = False
    project_pose_at_inversion_limit: bool = False
    projection_feasible_witness: bool = False
    projection_witness_policy: Literal["zero_pose", "attainable_optimum"] = "zero_pose"
    adjoint_relative_shift: float = 1e-5
    predictor_relative_shift: float = 1e-5
    adjoint_rtol: float = 1e-9
    predictor_rtol: float = 1e-9
    projection_probe_epsilon: float = 5e-5
    initial_trial_alpha: float = 0.015625
    objective_mode: Literal["l2", "l2-normal-smooth"] = "l2-normal-smooth"
    gradient_check_epsilons: tuple[float, ...] = (1e-4, 1e-5, 1e-6, 3e-7, 1e-7)
    computation_cutoff_utc: str | None = "2026-09-30T05:55:00+00:00"


def main(cfg: Config) -> None:
    assert cfg.computation_cutoff_utc == "2026-09-30T05:55:00+00:00"
    assert datetime.now(UTC) < datetime.fromisoformat(cfg.computation_cutoff_utc)
    source = GROUP / "data/inverse-mouthopen-coupled-017"
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
    assert cfg.objective_mode == "l2-normal-smooth"
    assert not cfg.projection_affine_residual
    assert cfg.adjoint_rtol == cfg.predictor_rtol == 1e-9
    assert cfg.adjoint_relative_shift == cfg.predictor_relative_shift == 1e-5
    audit = json.loads((source / "independent-audit.json").read_text())
    assert audit["valid_forward"]
    for item in audit["inputs"].values():
        assert sha256(Path(item["path"])) == item["sha256"]
    summary = json.loads((source / "summary.json").read_text())
    assert summary["status"] == "joint_projection_cache_failed"
    assert summary["final"]["optimizer_steps"] == {"q": 373, "pose": 373}
    source_protocol = json.loads((source / "protocol.json").read_text())
    assert source_protocol["objective_terms"]["mode"] == cfg.objective_mode
    runner.main(cfg)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
