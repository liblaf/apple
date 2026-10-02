"""Continue the saved repaired-reference active-strain Newton phase."""

from __future__ import annotations

import importlib.util
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from liblaf import cherries
from liblaf.apple.inverse import DifferentiableForward

GROUP = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location(
    "continued_neutral_forward", GROUP / "src/30-forward-active-strain.py"
)
assert spec is not None
assert spec.loader is not None
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


class Config(runner.Config):
    output_dir: Path = GROUP / "data/forward-repaired-reference-005"
    reference_dir: Path = GROUP / "data/reference-clearance-002"
    resume_dir: Path = GROUP / "data/forward-repaired-reference-001"
    max_newton_steps: int = 5000
    newton_shift_policy: str = "reuse"
    reuse_shift_force_ratio: float = 0.0


def main(cfg: Config) -> None:
    runner.main(cfg)


if __name__ == "__main__":
    with ExitStack() as guards:
        for name in ("forward", "step", "adjoint_solve", "receipt"):
            guards.enter_context(
                patch.object(DifferentiableForward, name, runner.forbidden_inverse)
            )
        cherries.main(main, profile=runner.benchmark.ProfilePerformance)
