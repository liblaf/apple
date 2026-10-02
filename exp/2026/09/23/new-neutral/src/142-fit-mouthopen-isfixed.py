"""Fit MouthOpen from the corrected neutral with collision and skin prestretch."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "isfixed_mouthopen_runner", GROUP / "src/130-inverse-mouthopen-rigid.py"
)
assert SPEC is not None
assert SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


class Config(runner.Config):
    output_dir: Path = GROUP / "data/inverse-mouthopen-isfixed-003"
    neutral_dir: Path = GROUP / "data/forward-isfixed-001"
    blendshape_dir: Path = GROUP / "data/blendshapes-isfixed-001"
    initialization_checkpoint: Path | None = (
        GROUP / "data/inverse-mouthopen-isfixed-002/checkpoint.pt"
    )
    seed_method: str = "collision_carry"
    maximum_iterations: int = 9
    learning_rate: float = 0.02
    pose_learning_rate: float = 0.02
    adjoint_relative_shift: float = 0.001
    wall_seconds: float | None = 1200
    max_backtracks: int = 14


def main(cfg: Config) -> None:
    runner.main(cfg)


if __name__ == "__main__":
    cherries.main(main, profile=runner.ProfileJoint)
