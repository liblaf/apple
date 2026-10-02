"""A clipped step or a flat objective cannot establish inverse convergence."""

import sys
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2] / "exp/2026/09/23/new-neutral/src"
sys.path.insert(0, str(SOURCE))

from mouthopen_convergence import stationarity_candidate  # noqa: E402


def check(gradients: list[float], *, valid: bool = True) -> dict:
    rows = [
        {"loss": 1.0, "gradient": {"scaled_gradient_l1": g}, "valid_forward": valid}
        for g in gradients
    ]
    return stationarity_candidate(
        rows,
        patience=2,
        loss_rtol=1e-6,
        gradient_rtol=1e-3,
        gradient_atol=1e-8,
    )


def test_flat_loss_with_large_gradient_is_not_a_candidate():
    assert not check([1.0, 1.0, 1.0])["candidate_requires_independent_audit"]


def test_small_gradient_and_plateau_requests_audit_not_convergence():
    result = check([1.0, 1e-5, 1e-5])
    assert result["candidate_requires_independent_audit"]
    assert not result["inverse_converged"]


def test_invalid_forward_cannot_pass():
    assert not check([1.0, 1e-5, 1e-5], valid=False)[
        "candidate_requires_independent_audit"
    ]
