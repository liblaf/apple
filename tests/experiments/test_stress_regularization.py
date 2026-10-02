"""CPU tests for physical active-stress component-gradient accounting."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(
    0,
    str(Path(__file__).parents[2] / "exp/2026/09/21/stress-activation-loss/src"),
)

from stress_regularization import (
    calibrated_weight,
    dual_volume_norm,
    gradient_balance,
)
from stress_study import StressStudy

DTYPE = torch.float64


def test_component_tensor_norms_and_calibration_do_not_depend_on_control_map() -> None:
    mass = torch.tensor((0.25, 0.75), dtype=DTYPE)
    l2_gradient = torch.tensor(
        (
            ((2.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            ((-1.0, 0.0, 0.0), (0.0, 3.0, 2.0), (0.0, 2.0, 0.0)),
        ),
        dtype=DTYPE,
    )
    smooth_gradient = 4 * l2_gradient

    weight = calibrated_weight(l2_gradient, smooth_gradient, mass)
    balance = gradient_balance(l2_gradient, smooth_gradient, mass, weight)

    assert weight == pytest.approx(0.025)
    assert balance["smoothness_to_l2_gradient_ratio"] == pytest.approx(0.1)
    assert balance["weighted_smoothness_gradient_dual_norm"] == pytest.approx(
        0.1 * balance["l2_gradient_dual_norm"]
    )
    assert dual_volume_norm(l2_gradient, mass) == pytest.approx(
        dual_volume_norm(l2_gradient.transpose(-1, -2), mass)
    )


def test_zero_l2_denominator_is_reported_and_cannot_be_calibrated() -> None:
    mass = torch.tensor((0.4, 0.6), dtype=DTYPE)
    zero = torch.zeros((2, 3, 3), dtype=DTYPE)
    smooth = torch.eye(3, dtype=DTYPE).expand(2, -1, -1).clone()

    balance = gradient_balance(zero, smooth, mass, 0.1)

    assert balance["l2_gradient_dual_norm"] == 0
    assert balance["smoothness_to_l2_gradient_ratio"] is None
    with pytest.raises(AssertionError, match="nonzero L2 gradient"):
        calibrated_weight(zero, smooth, mass)


class _FakePhysics:
    def check_adjoint(self) -> dict[str, bool]:
        return {"fake_equilibrium": True, "success": True}


def _fake_study() -> StressStudy:
    """Build only the state consumed by ``StressStudy.backward``."""
    study = object.__new__(StressStudy)
    study.physics = _FakePhysics()
    study.edge_i = torch.tensor((0,))
    study.edge_j = torch.tensor((1,))
    study.conductance = torch.tensor((2.0,), dtype=DTYPE)
    study.regularizer_factor = 0.75
    study.active_weights = np.array((0.4, 0.6))
    return study


def test_backward_reports_l2_calibration_gradient_from_one_equilibrium() -> None:
    """Normal loss affects J but is deliberately outside the L2 calibration ratio."""
    study = _fake_study()
    q = torch.tensor(
        (
            ((0.4, 0.1, 0.0), (0.1, -0.2, 0.0), (0.0, 0.0, 0.3)),
            ((-0.1, 0.0, 0.2), (0.0, 0.5, 0.1), (0.2, 0.1, -0.4)),
        ),
        dtype=DTYPE,
        requires_grad=True,
    )
    qhat = (q + q.mT) / 2
    equilibrium = 1.7 * qhat
    position = 0.5 * equilibrium.square().sum()
    normal = (
        (equilibrium[:, 0, 0] - (equilibrium[:, 1, 2] + equilibrium[:, 2, 1]) / 2)
        .square()
        .sum()
    )
    beta, eta = 0.35, 0.2
    regularizer = study.regularizer(qhat)
    loss = position + beta * normal + eta * regularizer
    expected_l2 = torch.autograd.grad(position, qhat, retain_graph=True)[0].detach()
    expected_normal = torch.autograd.grad(normal, qhat, retain_graph=True)[0].detach()
    expected_regularizer = torch.autograd.grad(regularizer, qhat, retain_graph=True)[
        0
    ].detach()
    expected_total = expected_l2 + beta * expected_normal + eta * expected_regularizer
    expected_q = torch.autograd.grad(loss, q, retain_graph=True)[0].detach()
    result = {
        "solver_valid": True,
        "_loss": loss,
        "_position": position,
        "_Qhat": qhat,
        "_normal_weight": beta,
        "_smooth_weight": eta,
    }

    study.backward(q, result, component_gradients=True)

    assert torch.allclose(result["tensor_gradient"], expected_total)
    assert torch.allclose(result["l2_tensor_gradient"], expected_l2)
    assert torch.allclose(result["regularizer_tensor_gradient"], expected_regularizer)
    assert torch.allclose(result["gradient"], expected_q)
    assert not torch.allclose(
        result["tensor_gradient"], expected_l2 + eta * expected_regularizer
    )
    assert result["smoothness_to_l2_gradient_ratio"] == pytest.approx(
        eta
        * float(
            dual_volume_norm(expected_regularizer, torch.tensor(study.active_weights))
        )
        / float(dual_volume_norm(expected_l2, torch.tensor(study.active_weights)))
    )
