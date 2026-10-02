# Copyright (c) 2026 liblaf
"""Launch the repaired fresh collision-on Smile inverse before its explicit deadline."""

from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "smile_collision_on_expression_gradient_runner",
    GROUP / "src/3000-inverse-smile-expression-gradient.py",
)
assert SPEC is not None
assert SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


class Config(runner.Config):
    """Fresh Smile only; a source MouthOpen checkpoint is never accepted."""

    output_dir: Path = GROUP / "data/inverse-smile-coupled-003"
    expression_name: str = "Smile"
    initialization_checkpoint: Path | None = None
    initialization_refinement: Path | None = None
    continue_optimizer_state: bool = False
    maximum_iterations: int = 10000
    q_only_iterations: int = 0
    learning_rate: float = 0.002
    pose_learning_rate: float = 0.1
    forward_atol: float = 1e-8
    internal_forward_atol: float | None = 1e-9
    adjoint_relative_shift: float = 1e-5
    predictor_relative_shift: float = 1e-5
    maximum_inverted_tetrahedra: int = 100
    maximum_inverted_rest_volume_fraction: float = 1e-4
    exclude_fully_fixed_tets: bool = True
    seed_method: str = "coupled_tangent"
    max_rotation_increment_deg: float | None = None
    max_translation_increment_m: float | None = None
    projection_bounded_joint: bool = True
    projection_descent_fraction: float | None = 0.1
    objective_mode: Literal["l2", "l2-normal-smooth"] = "l2-normal-smooth"
    deadline_iso_utc: str = "2026-09-30T05:50:00+00:00"
    adaptive_trial_alpha: bool = True
    initial_trial_alpha: float = 0.0625
    minimum_trial_alpha: float = 1e-6
    convergence_patience: int = 10
    gradient_check_epsilons: tuple[float, ...] = (
        1e-4,
        1e-5,
        1e-6,
        3e-7,
        1e-7,
    )


def remaining_wall_seconds(cfg: Config) -> float:
    """Bound this run by a portable, explicit UTC delivery cutoff."""
    deadline = datetime.fromisoformat(cfg.deadline_iso_utc)
    assert deadline.tzinfo is not None
    remaining = (deadline - datetime.now(UTC)).total_seconds()
    assert remaining > 0, f"Smile deadline has passed: {cfg.deadline_iso_utc}"
    return remaining if cfg.wall_seconds is None else min(cfg.wall_seconds, remaining)


def main(cfg: Config) -> None:
    assert cfg.expression_name == "Smile"
    assert cfg.initialization_checkpoint is None
    assert cfg.initialization_refinement is None
    assert not cfg.continue_optimizer_state
    assert cfg.q_only_iterations == 0
    assert cfg.objective_mode == "l2-normal-smooth"
    assert cfg.exclude_fully_fixed_tets
    assert cfg.maximum_inverted_tetrahedra == 100
    assert cfg.maximum_inverted_rest_volume_fraction == 1e-4
    assert cfg.forward_atol == 1e-8
    assert cfg.internal_forward_atol == 1e-9
    assert cfg.adjoint_relative_shift == cfg.predictor_relative_shift == 1e-5
    assert cfg.learning_rate == 0.002
    assert cfg.pose_learning_rate == 0.1
    assert cfg.projection_bounded_joint
    assert cfg.projection_descent_fraction == 0.1
    assert cfg.max_rotation_increment_deg is None
    assert cfg.max_translation_increment_m is None
    assert cfg.seed_method == "coupled_tangent"
    assert cfg.adaptive_trial_alpha
    assert cfg.initial_trial_alpha == 0.0625
    assert cfg.minimum_trial_alpha == 1e-6
    assert cfg.convergence_patience == 10
    assert cfg.gradient_check_epsilons == (1e-4, 1e-5, 1e-6, 3e-7, 1e-7)
    bounded = cfg.model_copy(update={"wall_seconds": remaining_wall_seconds(cfg)})
    runner.main(bounded)


if __name__ == "__main__":
    cherries.main(main, profile=runner.ProfileJoint)
