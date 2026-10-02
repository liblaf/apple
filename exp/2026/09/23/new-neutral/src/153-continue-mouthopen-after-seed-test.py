"""Resume the audited run005 optimizer after the isolated initializer test."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "mouthopen_projected_after_seed_test",
    GROUP / "src/151-continue-mouthopen-projected.py",
)
assert SPEC is not None
assert SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)

from joint_common import ProfileJoint  # noqa: E402


class Config(runner.Config):
    output_dir: Path = GROUP / "data/inverse-mouthopen-coupled-006"
    initial_trial_alpha: float = 0.5
    initialization_checkpoint: Path | None = (
        GROUP / "data/inverse-mouthopen-coupled-005/checkpoint.pt"
    )


def main(cfg: Config) -> None:
    runner.main(cfg)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
