"""Focused CPU tests for the staged active-stress parameterizations."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(
    0,
    str(Path(__file__).parents[2] / "exp/2026/09/21/stress-activation-loss/src"),
)

from activation_models import (
    initialize_learned_zero_amplitude_axes,
    learned_controls_from_fixed,
    mandel_to_matrix,
    matrices,
)

DTYPE = torch.float64


def test_zero_amplitude_axes_choose_negative_eigenvalue_without_activation() -> None:
    controls = torch.tensor(
        ((0.0, 1.0, 0.0, 0.0), (0.7, 0.0, 0.0, 1.0), (1e-15, 0.0, 1.0, 0.0)),
        dtype=DTYPE,
    )
    gradient = torch.stack(
        (
            torch.diag(torch.tensor((1.0, -1.0, 1.0), dtype=DTYPE)),
            torch.diag(torch.tensor((-2.0, 1.0, 1.0), dtype=DTYPE)),
            torch.diag(torch.tensor((-3.0, 1.0, 1.0), dtype=DTYPE)),
        )
    )

    initialized = initialize_learned_zero_amplitude_axes(controls, gradient)

    assert torch.equal(initialized[..., 0], controls[..., 0])
    assert torch.equal(initialized[1:], controls[1:])
    assert torch.allclose(
        initialized[0, 1:].abs(), torch.tensor((0.0, 1.0, 0.0), dtype=DTYPE)
    )
    assert torch.equal(
        matrices(initialized, "rankone_learned"),
        matrices(controls, "rankone_learned"),
    )


def test_zero_amplitude_axes_stay_when_gradient_is_psd_in_matrix_or_mandel_form() -> (
    None
):
    controls = torch.tensor(((0.0, 0.0, 1.0, 0.0),), dtype=DTYPE)
    psd = torch.diag(torch.tensor((0.0, 2.0, 3.0), dtype=DTYPE)).unsqueeze(0)
    assert torch.equal(initialize_learned_zero_amplitude_axes(controls, psd), controls)
    mandel = torch.tensor(((0.0, 2.0, 3.0, 0.0, 0.0, 0.0),), dtype=DTYPE)
    assert torch.equal(mandel_to_matrix(mandel), psd)
    assert torch.equal(
        initialize_learned_zero_amplitude_axes(controls, mandel), controls
    )


def test_fixed_to_learned_retains_parent_axes_at_zero_amplitude() -> None:
    q_fixed = torch.tensor(((0.0,), (0.4,)), dtype=DTYPE)
    parent_axes = torch.tensor(((0.0, -2.0, 0.0), (3.0, 0.0, 4.0)), dtype=DTYPE)

    learned = learned_controls_from_fixed(q_fixed, parent_axes)

    expected_axes = parent_axes / torch.linalg.vector_norm(
        parent_axes, dim=-1, keepdim=True
    )
    assert torch.equal(learned[..., :1], q_fixed)
    assert torch.equal(learned[..., 1:], expected_axes)
    assert torch.equal(
        matrices(learned, "rankone_learned"),
        matrices(q_fixed, "rankone_fixed", parent_axes),
    )


def test_rank_one_learned_map_passes_gradcheck_away_from_zero_amplitude() -> None:
    controls = torch.tensor((0.7, 0.2, -0.3, 0.9), dtype=DTYPE, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda q: matrices(q, "rankone_learned"), (controls,), eps=1e-6
    )
