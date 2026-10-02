"""Continue from refined run006, preserving its controls and Adam state."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "mouthopen_refined_continuation",
    GROUP / "src/156-continue-mouthopen-descent-projected.py",
)
assert SPEC is not None
assert SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)

from joint_common import ProfileJoint  # noqa: E402


class Config(runner.Config):
    output_dir: Path = GROUP / "data/inverse-mouthopen-coupled-008"
    initialization_checkpoint: Path | None = (
        GROUP / "data/inverse-mouthopen-coupled-006/checkpoint.pt"
    )
    initialization_refinement: Path | None = (
        GROUP
        / "data/mouthopen-equilibrium-polish-001/inverse-mouthopen-coupled-006/result.json"
    )
    internal_forward_atol: float | None = 1e-9


def main(cfg: Config) -> None:
    runner.main(cfg)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
