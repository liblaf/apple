"""Test push-out of finite saved estimates without relabelling their failed solve."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "mouthopen_saved_initializer_comparison",
    GROUP / "src/152-test-mouthopen-collision-off-seed.py",
)
assert SPEC is not None
assert SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


class Config(runner.Config):
    output_dir: Path = GROUP / "data/collision-off-seed-comparison-002"
    saved_estimate_run: Path | None = GROUP / "data/collision-off-seed-comparison-001"


def main(cfg: Config) -> None:
    runner.main(cfg)


if __name__ == "__main__":
    cherries.main(main, profile=runner.ProfileJoint)
