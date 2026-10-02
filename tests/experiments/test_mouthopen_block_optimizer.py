"""CPU behavior coverage for explicit Raw6 blockwise Adam updates."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "exp/2026/09/23/new-neutral/src"
INVERSE = SOURCE / "130-inverse-mouthopen-rigid.py"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from mouthopen_block_optimizer import block_adam_update  # noqa: E402


def _inverse_module():
    spec = importlib.util.spec_from_file_location("mouthopen_rigid_inverse", INVERSE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inputs() -> tuple[list[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    q = torch.linspace(-1.0, 1.0, 12, dtype=torch.float64).reshape(2, 6)
    pose = torch.linspace(-0.3, 0.4, 6, dtype=torch.float64)
    moments = [q * 0.4, q.square() * 0.2 + 0.1, pose * 0.5, pose.square() + 0.2]
    gradients = (q * -0.7 + 0.2, pose * 1.3 - 0.1)
    return moments, gradients


def test_joint_block_update_bitmatches_existing_global_adam() -> None:
    """Equal counters reproduce the historical joint Adam proposal exactly."""
    inverse = _inverse_module()
    moments, gradients = _inputs()
    expected_moments, expected_dq, expected_dp = inverse.adam_update(
        moments,
        gradients,
        optimizer_step=9,
        learning_rate=0.02,
        pose_learning_rate=0.1,
    )
    actual_moments, steps, actual_dq, actual_dp = block_adam_update(
        moments,
        gradients,
        q_optimizer_step=8,
        pose_optimizer_step=8,
        update_q=True,
        update_pose=True,
        learning_rate=0.02,
        pose_learning_rate=0.1,
    )

    assert steps == {"q": 9, "pose": 9}
    for actual, expected in zip(actual_moments, expected_moments, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_dq, expected_dq, rtol=0, atol=0)
    torch.testing.assert_close(actual_dp, expected_dp, rtol=0, atol=0)


def test_q_only_update_freezes_pose_moments_counter_and_direction() -> None:
    """A q-only accepted update cannot alter pose optimizer state or proposal."""
    moments, gradients = _inputs()
    actual_moments, steps, dq, dp = block_adam_update(
        moments,
        gradients,
        q_optimizer_step=4,
        pose_optimizer_step=7,
        update_q=True,
        update_pose=False,
        learning_rate=0.02,
        pose_learning_rate=0.1,
    )

    assert steps == {"q": 5, "pose": 7}
    assert actual_moments[2] is moments[2]
    assert actual_moments[3] is moments[3]
    torch.testing.assert_close(dp, torch.zeros_like(gradients[1]), rtol=0, atol=0)
    assert torch.count_nonzero(dq)
