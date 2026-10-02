"""Test strain-only progress at the inversion boundary, then resume joint updates."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "mouthopen_block_continuation", GROUP / "src/148-continue-mouthopen-coupled.py"
)
assert SPEC is not None
assert SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


class Config(runner.Config):
    output_dir: Path = GROUP / "data/inverse-mouthopen-coupled-004"
    initialization_checkpoint: Path | None = (
        GROUP / "data/inverse-mouthopen-coupled-003/checkpoint.pt"
    )
    q_only_iterations: int = 10
    learning_rate: float = 0.002
    initial_trial_alpha: float = 1.0
    minimum_trial_alpha: float = 1e-6


def main(cfg: Config) -> None:
    runner.main(cfg)


if __name__ == "__main__":
    cherries.main(main, profile=runner.runner.runner.ProfileJoint)
