"""Conservative monitoring that requests an audit before declaring convergence."""

from __future__ import annotations

import math

import torch


def gradient_metrics(grads: tuple[torch.Tensor, torch.Tensor]) -> dict[str, float]:
    """Measure unrestricted descent, independently of CCD or accepted step size."""
    q, pose = grads
    assert bool(torch.isfinite(q).all())
    assert bool(torch.isfinite(pose).all())
    # Fixed characteristic changes: Raw6 0.02, jaw 1 degree / 1 mm.
    # Pose coordinates are normalized by 10 degrees / 10 mm in the runner.
    strain = 0.02 * float(q.abs().sum())
    jaw = 0.1 * float(pose.abs().sum())
    return {
        "strain_scaled_gradient_l1": strain,
        "pose_scaled_gradient_l1": jaw,
        "scaled_gradient_l1": strain + jaw,
        "strain_gradient_linf": float(q.abs().max()),
        "pose_gradient_linf": float(pose.abs().max()),
    }


def stationarity_candidate(
    history: list[dict],
    *,
    patience: int,
    loss_rtol: float,
    gradient_rtol: float,
    gradient_atol: float,
    gradient_reference: dict[str, float] | None = None,
) -> dict:
    """Flag a plateau only with a small un-clipped gradient and valid forwards.

    Damped gradients are approximate, so passing this check requires a separate
    endpoint audit with a less damped adjoint and resolved objective probes.
    """
    assert patience >= 1
    assert loss_rtol > 0
    assert gradient_rtol > 0
    assert gradient_atol >= 0
    assert history
    initial = (
        history[0]["gradient"] if gradient_reference is None else gradient_reference
    )
    current = history[-1]["gradient"]
    threshold = gradient_atol + gradient_rtol * initial["scaled_gradient_l1"]
    enough = len(history) > patience
    window = history[-(patience + 1) :]
    loss_change = abs(window[0]["loss"] - window[-1]["loss"])
    loss_scale = max(abs(window[0]["loss"]), abs(window[-1]["loss"]), 1e-12)
    relative_loss_change = loss_change / loss_scale
    valid = all(row["valid_forward"] for row in window)
    small_gradient = current["scaled_gradient_l1"] <= threshold
    candidate = (
        enough and valid and small_gradient and relative_loss_change <= loss_rtol
    )
    assert math.isfinite(relative_loss_change)
    return {
        "candidate_requires_independent_audit": candidate,
        "inverse_converged": False,
        "sufficient_history": enough,
        "all_forwards_valid": valid,
        "small_unclipped_gradient": small_gradient,
        "scaled_gradient_threshold": threshold,
        "relative_loss_change": relative_loss_change,
        "loss_rtol": loss_rtol,
        "patience": patience,
    }
