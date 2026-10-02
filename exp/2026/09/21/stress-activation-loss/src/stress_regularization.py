"""Compare stress covectors in one effective-volume metric across all stages."""

from __future__ import annotations

import torch


def dual_volume_norm(gradient: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
    """Return the dual volume norm of a symmetric tensor-field covector."""
    assert gradient.shape == (*mass.shape, 3, 3)
    assert bool(torch.isfinite(gradient).all())
    assert bool(torch.isfinite(mass).all() and (mass > 0).all())
    assert torch.allclose(mass.sum(), mass.new_tensor(1.0))
    symmetric = (gradient + gradient.mT) / 2
    return (symmetric.square().sum((-2, -1)) / mass).sum().sqrt()


def gradient_balance(
    l2_gradient: torch.Tensor,
    regularizer_gradient: torch.Tensor,
    mass: torch.Tensor,
    weight: float,
) -> dict:
    """Measure component gradients; a zero L2 denominator is undefined."""
    assert weight >= 0
    l2 = float(dual_volume_norm(l2_gradient, mass))
    smooth = float(dual_volume_norm(regularizer_gradient, mass))
    return {
        "l2_gradient_dual_norm": l2,
        "smoothness_gradient_dual_norm": smooth,
        "weighted_smoothness_gradient_dual_norm": weight * smooth,
        "smoothness_to_l2_gradient_ratio": weight * smooth / l2 if l2 > 0 else None,
    }


def calibrated_weight(
    l2_gradient: torch.Tensor,
    regularizer_gradient: torch.Tensor,
    mass: torch.Tensor,
    target_ratio: float = 0.1,
) -> float:
    """Freeze a coefficient from two gradients at the same nonzero-stress state."""
    assert 0 < target_ratio < 1
    l2 = dual_volume_norm(l2_gradient, mass)
    smooth = dual_volume_norm(regularizer_gradient, mass)
    assert float(l2) > 0, "calibration requires a nonzero L2 gradient"
    assert float(smooth) > 0, "calibration requires a nonzero smoothness gradient"
    return float(target_ratio * l2 / smooth)
