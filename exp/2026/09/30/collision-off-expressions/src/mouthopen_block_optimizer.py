"""Explicit blockwise Adam updates for the coupled MouthOpen inverse."""

from __future__ import annotations

import torch


def block_adam_update(
    moments: list[torch.Tensor],
    gradients: tuple[torch.Tensor, torch.Tensor],
    *,
    q_optimizer_step: int,
    pose_optimizer_step: int,
    update_q: bool,
    update_pose: bool,
    learning_rate: float,
    pose_learning_rate: float,
) -> tuple[list[torch.Tensor], dict[str, int], torch.Tensor, torch.Tensor]:
    """Update selected Adam blocks and preserve the state of frozen blocks.

    The supplied step counts describe accepted updates already represented by
    ``moments``.  Each selected block advances its own count once; its Adam
    bias correction uses that incremented count.  A frozen block returns the
    same moment tensors and count, with a zero proposal of the matching shape.
    """
    assert len(moments) == 4
    assert q_optimizer_step >= 0
    assert pose_optimizer_step >= 0
    mq, vq, mp, vp = moments
    gq, gp = gradients
    assert mq.shape == vq.shape == gq.shape
    assert mp.shape == vp.shape == gp.shape

    if update_q:
        q_step = q_optimizer_step + 1
        next_mq = 0.9 * mq + 0.1 * gq
        next_vq = 0.999 * vq + 0.001 * gq.square()
        dq = (
            -learning_rate
            * (next_mq / (1 - 0.9**q_step))
            / ((next_vq / (1 - 0.999**q_step)).sqrt() + 1e-12)
        )
    else:
        q_step = q_optimizer_step
        next_mq, next_vq = mq, vq
        dq = torch.zeros_like(gq)

    if update_pose:
        pose_step = pose_optimizer_step + 1
        next_mp = 0.9 * mp + 0.1 * gp
        next_vp = 0.999 * vp + 0.001 * gp.square()
        dp = (
            -pose_learning_rate
            * (next_mp / (1 - 0.9**pose_step))
            / ((next_vp / (1 - 0.999**pose_step)).sqrt() + 1e-12)
        )
    else:
        pose_step = pose_optimizer_step
        next_mp, next_vp = mp, vp
        dp = torch.zeros_like(gp)

    return (
        [next_mq, next_vq, next_mp, next_vp],
        {"q": q_step, "pose": pose_step},
        dq,
        dp,
    )


__all__ = ["block_adam_update"]
