"""CPU contracts for selectable additive-stress and native active-strain paths."""
# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch

from liblaf.apple.warp.fem import (
    StableNeoHookean,
    StableNeoHookeanActive,
    WarpPotentialFem,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "exp/2026/09/21/stress-activation-loss/src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from activation_models import (  # ty: ignore[unresolved-import]
    STRESS_REF_MPA,
    matrices,
)
from stress_material import StableNeoHookeanStress  # ty: ignore[unresolved-import]
from stress_physics import (  # ty: ignore[unresolved-import]
    install_exact_bulk_diagonal,
    strain_to_activation_inv,
)
from stress_study import StressStudy  # ty: ignore[unresolved-import]


def test_native_strain_adapter_uses_raw_symmetric_offdiagonals_and_gradients() -> None:
    strain = torch.tensor(
        [[[0.1, 0.2, -0.3], [0.2, -0.4, 0.5], [-0.3, 0.5, 0.6]]],
        dtype=torch.float64,
        requires_grad=True,
    )

    native = strain_to_activation_inv(strain)
    native.sum().backward()

    torch.testing.assert_close(
        native,
        torch.tensor([[0.1, -0.4, 0.6, 0.2, 0.5, -0.3]], dtype=torch.float64),
    )
    assert strain.grad is not None
    torch.testing.assert_close(
        strain.grad,
        torch.tensor(
            [[[1.0, 1.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]],
            dtype=torch.float64,
        ),
    )


class _NormalLoss:
    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        return u.square().sum() * 0.0

    def metrics(self, u: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"normal_angle_rms_deg": u.square().sum() * 0.0}


class _Physics:
    def __init__(self, activation_model: str) -> None:
        self.activation_model = activation_model
        self.diff = type("Diff", (), {"require_convergence": True})()
        self.last_forward = {"success": True, "solver_valid": True}
        self.received: torch.Tensor | None = None

    def solve(self, activation: torch.Tensor, seed: np.ndarray) -> torch.Tensor:
        del seed
        self.received = activation
        return torch.stack(
            (activation[0].sum(), activation[0, 0, 0] * 0.0, activation[0, 0, 0] * 0.0)
        ).unsqueeze(0)

    def detf(self, u: np.ndarray) -> np.ndarray:
        del u
        return np.ones(2)

    def check_adjoint(self) -> dict[str, bool]:
        return {"success": True}


def _study(activation_model: str) -> tuple[StressStudy, _Physics]:
    physics = _Physics(activation_model)
    study = object.__new__(StressStudy)
    study.physics = physics
    study.skin_ids_t = torch.tensor([0])
    study.target = torch.zeros((1, 3), dtype=torch.float64)
    study.weights_t = torch.ones(1, dtype=torch.float64)
    study.normal_loss = _NormalLoss()
    study.gradient_loss = lambda u, target: (u - target).square().sum() * 0.0
    study.edge_i = torch.tensor([0])
    study.edge_j = torch.tensor([1])
    study.conductance = torch.ones(1, dtype=torch.float64)
    study.regularizer_factor = 1.0
    study.active_weights = np.array([0.5, 0.5])
    return study, physics


@pytest.mark.parametrize("activation_model", ["stress", "strain"])
def test_study_scales_only_stress_and_labels_strain_diagnostics(
    activation_model: str,
) -> None:
    study, physics = _study(activation_model)
    q = torch.tensor(
        [[0.2, -0.1, 0.3, 0.4, -0.2, 0.1], [0.0, 0.1, -0.2, 0.3, 0.2, -0.4]],
        dtype=torch.float64,
        requires_grad=True,
    )
    expected = matrices(q, "symmetric6")

    result = study.evaluate(
        q,
        "symmetric6",
        None,
        np.zeros((1, 3)),
        normal_weight=0.0,
        smooth_weight=0.1,
        backward=True,
    )

    assert physics.received is not None
    if activation_model == "stress":
        torch.testing.assert_close(physics.received, expected * STRESS_REF_MPA)
        assert "stress_frobenius_rms_kPa" in result
    else:
        torch.testing.assert_close(physics.received, expected)
        assert result["activation_model"] == "strain"
        assert "strain_frobenius_rms_dimensionless" in result
        assert "B_eigen_min_dimensionless" in result
        assert "stress_frobenius_rms_kPa" not in result
    assert q.grad is not None
    assert bool(torch.isfinite(q.grad).all())


def test_exact_bulk_diagonal_includes_native_active_strain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = {
        material: material.hess_diag_kernel
        for material in (
            StableNeoHookean,
            StableNeoHookeanStress,
            StableNeoHookeanActive,
        )
    }

    def make_kernel(func: object, *, clamp_hess_diag: bool) -> tuple[object, bool]:
        return func, clamp_hess_diag

    monkeypatch.setattr(WarpPotentialFem, "make_hess_diag_kernel", make_kernel)
    try:
        policy = install_exact_bulk_diagonal()
        assert (
            cast("tuple[Any, bool]", StableNeoHookeanActive.hess_diag_kernel)[1]
            is False
        )
        assert "native active strain" in policy["bulk"]
    finally:
        for material, kernel in original.items():
            material.hess_diag_kernel = kernel


def test_study_symmetrizes_native_activation_gradient_before_pullback() -> None:
    study = object.__new__(StressStudy)
    study.physics = type(
        "Physics", (), {"check_adjoint": lambda _: {"success": True}}
    )()
    study.edge_i = torch.tensor([0])
    study.edge_j = torch.tensor([1])
    study.conductance = torch.ones(1, dtype=torch.float64)
    study.regularizer_factor = 0.0
    study.active_weights = np.array([0.5, 0.5])
    q = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    Qhat = q.reshape(1, 1, 1).expand(2, 3, 3) * torch.tensor(
        [[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
        dtype=torch.float64,
    )
    position = Qhat[0, 0, 1]
    result = {
        "_loss": position,
        "_position": position,
        "_Qhat": Qhat,
        "_normal_weight": 1.0,
        "_smooth_weight": 0.0,
        "solver_valid": True,
    }

    study.backward(q, result, component_gradients=True)

    expected = torch.zeros_like(Qhat)
    expected[0, 0, 1] = expected[0, 1, 0] = 0.5
    torch.testing.assert_close(result["tensor_gradient"], expected)
    torch.testing.assert_close(result["l2_tensor_gradient"], expected)
    torch.testing.assert_close(q.grad, torch.ones_like(q))
