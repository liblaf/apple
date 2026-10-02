"""Continue joint fitting with a determinant-constrained jaw proposal."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "mouthopen_projected_continuation", GROUP / "src/150-continue-mouthopen-blocks.py"
)
assert SPEC is not None
assert SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


class Config(runner.Config):
    output_dir: Path = GROUP / "data/inverse-mouthopen-coupled-005"
    q_only_iterations: int = 0
    project_pose_at_inversion_limit: bool = True
    projection_activation_threshold: float = 0.05
    projection_determinant_margin: float = 1e-6
    projection_probe_epsilon: float = 1e-4
    initial_trial_alpha: float = 0.125
    max_backtracks: int = 20


def main(cfg: Config) -> None:
    runner.main(cfg)


if __name__ == "__main__":
    cherries.main(main, profile=runner.runner.runner.runner.ProfileJoint)
