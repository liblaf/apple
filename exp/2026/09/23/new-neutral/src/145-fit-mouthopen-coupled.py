"""Fit MouthOpen with retained FEM, coupled prediction, and collision."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "coupled_mouthopen_runner", GROUP / "src/130-inverse-mouthopen-rigid.py"
)
assert SPEC is not None
assert SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


class Config(runner.Config):
    output_dir: Path = GROUP / "data/inverse-mouthopen-coupled-001"
    neutral_dir: Path = GROUP / "data/forward-isfixed-001"
    blendshape_dir: Path = GROUP / "data/blendshapes-isfixed-001"
    initialization_checkpoint: Path | None = None
    seed_method: str = "coupled_tangent"
    exclude_fully_fixed_tets: bool = True
    maximum_inverted_tetrahedra: int = 100
    maximum_inverted_rest_volume_fraction: float = 0.0001
    max_rotation_increment_deg: float | None = None
    max_translation_increment_m: float | None = None
    predictor_relative_shift: float = 0.001
    predictor_rtol: float = 1e-7
    maximum_iterations: int = 20
    learning_rate: float = 0.02
    pose_learning_rate: float = 0.1
    adjoint_relative_shift: float = 0.001
    wall_seconds: float | None = 1200
    max_backtracks: int = 16


def main(cfg: Config) -> None:
    runner.main(cfg)


if __name__ == "__main__":
    cherries.main(main, profile=runner.ProfileJoint)
