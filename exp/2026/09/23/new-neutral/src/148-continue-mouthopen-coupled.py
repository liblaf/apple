"""Continue the coupled MouthOpen fit with preserved optimizer state."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "coupled_continuation_runner", GROUP / "src/145-fit-mouthopen-coupled.py"
)
assert SPEC is not None
assert SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


class Config(runner.Config):
    output_dir: Path = GROUP / "data/inverse-mouthopen-coupled-003"
    initialization_checkpoint: Path | None = (
        GROUP / "data/inverse-mouthopen-coupled-002/checkpoint.pt"
    )
    continue_optimizer_state: bool = True
    adaptive_trial_alpha: bool = True
    initial_trial_alpha: float = 0.125
    maximum_iterations: int = 10000
    wall_seconds: float | None = None
    convergence_patience: int = 10
    convergence_loss_rtol: float = 1e-6
    convergence_gradient_rtol: float = 1e-3
    convergence_gradient_atol: float = 1e-8


def main(cfg: Config) -> None:
    runner.main(cfg)


if __name__ == "__main__":
    cherries.main(main, profile=runner.runner.ProfileJoint)
