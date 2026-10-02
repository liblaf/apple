"""Continue audited run009 with an explicit feasible descent witness policy."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "mouthopen_witness_continuation",
    GROUP / "src/160-continue-mouthopen-calibrated-tangent.py",
)
assert SPEC is not None
assert SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)

from joint_common import ProfileJoint  # noqa: E402


class Config(runner.Config):
    output_dir: Path = GROUP / "data/inverse-mouthopen-coupled-010"
    initialization_checkpoint: Path | None = (
        GROUP / "data/inverse-mouthopen-coupled-009/checkpoint.pt"
    )
    initialization_refinement: Path | None = None
    projection_feasible_witness: bool = True
    adjoint_relative_shift: float = 1e-4
    predictor_relative_shift: float = 1e-4
    projection_analytic_determinants: bool = True
    projection_probe_epsilon: float = 5e-5
    initial_trial_alpha: float = 0.25


def main(cfg: Config) -> None:
    runner.main(cfg)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
