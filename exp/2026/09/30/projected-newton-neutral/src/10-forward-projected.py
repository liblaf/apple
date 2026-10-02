"""Neutral forward solve with the projected-Newton search matrix."""

from __future__ import annotations

import importlib.util
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from liblaf import cherries
from liblaf.apple.inverse import DifferentiableForward

GROUP = Path(__file__).resolve().parent.parent
NEW_NEUTRAL = GROUP.parents[1] / "23/new-neutral"
SPEC = importlib.util.spec_from_file_location(
    "projected_neutral_forward", GROUP / "src/forward_runner.py"
)
assert SPEC is not None
assert SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


class Config(runner.Config):
    output_dir: Path = GROUP / "data/forward-projected-001"
    reference_dir: Path = NEW_NEUTRAL / "data/reference-clearance-002"
    resume_dir: Path | None = None
    max_newton_steps: int = 5000
    checkpoint_steps: int = 50
    newton_shift_policy: str = "reuse"
    reuse_shift_force_ratio: float = 0.0


def main(cfg: Config) -> None:
    assert cfg.resume_dir is None, "This run must start from zero displacement"
    runner.main(cfg)


if __name__ == "__main__":
    with ExitStack() as guards:
        for name in ("forward", "step", "adjoint_solve", "receipt"):
            guards.enter_context(
                patch.object(DifferentiableForward, name, runner.forbidden_inverse)
            )
        cherries.main(main, profile=runner.benchmark.ProfilePerformance)
