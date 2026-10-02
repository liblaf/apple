"""Continue audited run011 with alpha-specific affine residual pose constraints."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Literal

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "mouthopen_affine_continuation",
    GROUP / "src/163-continue-mouthopen-attainable-descent.py",
)
assert SPEC is not None
assert SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)

from joint_common import ProfileJoint  # noqa: E402


class Config(runner.Config):
    output_dir: Path = GROUP / "data/inverse-mouthopen-coupled-012"
    initialization_checkpoint: Path | None = (
        GROUP / "data/inverse-mouthopen-coupled-011/checkpoint.pt"
    )
    initialization_refinement: Path | None = None
    projection_feasible_witness: bool = True
    projection_witness_policy: Literal["zero_pose", "attainable_optimum"] = (
        "attainable_optimum"
    )
    projection_affine_residual: bool = True
    adjoint_relative_shift: float = 1e-5
    predictor_relative_shift: float = 1e-5
    projection_analytic_determinants: bool = True
    projection_probe_epsilon: float = 5e-5
    initial_trial_alpha: float = 0.125


def main(cfg: Config) -> None:
    assert cfg.forward_atol == 1e-8
    assert cfg.internal_forward_atol == 1e-9
    assert cfg.maximum_inverted_tetrahedra == 100
    assert cfg.maximum_inverted_rest_volume_fraction == 1e-4
    assert (
        cfg.initialization_checkpoint
        == GROUP / "data/inverse-mouthopen-coupled-011/checkpoint.pt"
    )
    assert cfg.initialization_refinement is None
    assert cfg.continue_optimizer_state
    assert not cfg.resume
    runner.main(cfg)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
