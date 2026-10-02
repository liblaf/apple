"""Behavior coverage for Raw6 MouthOpen Adam-state continuation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "exp/2026/09/23/new-neutral/src/130-inverse-mouthopen-rigid.py"


def _module():
    spec = importlib.util.spec_from_file_location("mouthopen_rigid_inverse", SOURCE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_restored_moments_and_global_step_match_uninterrupted_adam() -> None:
    """The next proposal is identical after checkpoint serialization/restoration."""
    inverse = _module()
    q0 = torch.zeros((2, 6), dtype=torch.float64)
    pose0 = torch.zeros(6, dtype=torch.float64)
    zero_moments = [
        torch.zeros_like(q0),
        torch.zeros_like(q0),
        torch.zeros_like(pose0),
        torch.zeros_like(pose0),
    ]
    first_gradients = (
        torch.linspace(-2.0, 1.0, q0.numel(), dtype=torch.float64).reshape_as(q0),
        torch.linspace(0.1, 0.6, 6, dtype=torch.float64),
    )
    second_gradients = (
        torch.linspace(1.5, -0.5, q0.numel(), dtype=torch.float64).reshape_as(q0),
        torch.linspace(-0.8, 0.4, 6, dtype=torch.float64),
    )
    moments_after_first, _, _ = inverse.adam_update(
        zero_moments,
        first_gradients,
        optimizer_step=1,
        learning_rate=0.02,
        pose_learning_rate=0.1,
    )
    uninterrupted, q_direction, pose_direction = inverse.adam_update(
        moments_after_first,
        second_gradients,
        optimizer_step=2,
        learning_rate=0.02,
        pose_learning_rate=0.1,
    )
    restored = [value.clone() for value in moments_after_first]
    continued, continued_q_direction, continued_pose_direction = inverse.adam_update(
        restored,
        second_gradients,
        optimizer_step=2,
        learning_rate=0.02,
        pose_learning_rate=0.1,
    )

    for expected, actual in zip(uninterrupted, continued, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(continued_q_direction, q_direction, rtol=0, atol=0)
    torch.testing.assert_close(continued_pose_direction, pose_direction, rtol=0, atol=0)
