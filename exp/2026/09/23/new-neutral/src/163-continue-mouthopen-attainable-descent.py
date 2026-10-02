"""Continue audited run010 with certified attainable-descent pose witnesses."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Literal

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "mouthopen_attainable_continuation",
    GROUP / "src/162-continue-mouthopen-feasible-witness.py",
)
assert SPEC is not None
assert SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)

from joint_common import ProfileJoint  # noqa: E402


class Config(runner.Config):
    output_dir: Path = GROUP / "data/inverse-mouthopen-coupled-011"
    initialization_checkpoint: Path | None = (
        GROUP / "data/inverse-mouthopen-coupled-010/checkpoint.pt"
    )
    initialization_refinement: Path | None = None
    projection_feasible_witness: bool = True
    projection_witness_policy: Literal["zero_pose", "attainable_optimum"] = (
        "attainable_optimum"
    )
    adjoint_relative_shift: float = 1e-4
    predictor_relative_shift: float = 1e-4
    projection_analytic_determinants: bool = True
    projection_probe_epsilon: float = 5e-5
    initial_trial_alpha: float = 0.25


def main(cfg: Config) -> None:
    runner.main(cfg)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
