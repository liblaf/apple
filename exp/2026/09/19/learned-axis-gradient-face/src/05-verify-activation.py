"""CPU algebra and derivative gates for contraction-only learned-axis activation."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path

import torch

from liblaf import cherries

sys.path.insert(0, str(Path(__file__).parent))
import activation_model as am


class Config(cherries.BaseConfig):
    output: Path = cherries.output("05-activation/checks.json", mkdir=True)


def _relative_error(actual: float, expected: float) -> float:
    return abs(actual - expected) / max(1.0, abs(actual), abs(expected))


def _central_difference(
    function: Callable[[torch.Tensor], torch.Tensor],
    value: torch.Tensor,
    direction: torch.Tensor,
) -> float:
    epsilon = 1e-6
    with torch.no_grad():
        return float(
            (
                function(value + epsilon * direction)
                - function(value - epsilon * direction)
            )
            / (2 * epsilon)
        )


def main(cfg: Config) -> None:
    torch.manual_seed(20260919)
    dtype = torch.float64
    strength = torch.tensor([0.3, 0.7], dtype=dtype, requires_grad=True)
    axis = torch.tensor(
        [[2.0, -1.0, 3.0], [-2.0, 4.0, 1.0]], dtype=dtype, requires_grad=True
    )
    packed = am.pack(strength, axis)
    B = am.matrices(strength, axis)
    A = am.inverse_matrices(strength, axis)

    probes = torch.tensor(
        [[0.7, -0.4, 0.2, 0.5, -0.3, 0.6], [-0.2, 0.1, 0.8, -0.6, 0.4, 0.3]],
        dtype=dtype,
    )
    response = (packed * probes).sum()
    strength_grad, axis_grad = torch.autograd.grad(response, (strength, axis))
    nonzero_strength_response = float(torch.linalg.vector_norm(strength_grad))
    nonzero_axis_response = float(torch.linalg.vector_norm(axis_grad))

    zero_strength = torch.zeros(2, dtype=dtype, requires_grad=True)
    zero_axis = axis.detach().clone().requires_grad_()
    zero_response = (am.pack(zero_strength, zero_axis) * probes).sum()
    zero_strength_grad, zero_axis_grad = torch.autograd.grad(
        zero_response, (zero_strength, zero_axis)
    )

    projected_strength = torch.tensor([-1.0, 0.2], dtype=dtype)
    projected_axis = torch.tensor([[3.0, 4.0, 0.0], [-1.0, 2.0, 2.0]], dtype=dtype)
    am.project_(projected_strength, projected_axis)

    sign_invariant = float(
        torch.max(
            torch.abs(
                am.pack(strength.detach(), axis.detach())
                - am.pack(strength.detach(), -axis.detach())
            )
        )
    )
    a_eigenvalues = torch.linalg.eigvalsh(A)
    expected_a_eigenvalues = (
        torch.stack(
            (torch.ones_like(strength), torch.ones_like(strength), 1 / (1 + strength)),
            dim=-1,
        )
        .sort(dim=-1)
        .values
    )

    raw_gradient = torch.tensor(
        [[-3.0, 1.0, 2.0, 0.6, -0.2, 0.4], [-1.0, -2.0, 1.0, 0.3, 0.5, -0.7]],
        dtype=dtype,
    )
    eigen_axis, eigenvalues = am.axes_from_packed_gradient(raw_gradient)
    linear_strength = torch.full((2,), 1e-4, dtype=dtype)
    linear_energy = (am.pack(linear_strength, eigen_axis) * raw_gradient).sum()
    expected_linear_energy = (linear_strength * eigenvalues[:, 0]).sum()
    canonical_axis_minimum = torch.einsum(
        "bi,bij,bj->b",
        eigen_axis,
        am.packed_gradient_to_symmetric(raw_gradient),
        eigen_axis,
    )

    smooth_strength = torch.tensor([0.2, 0.6, 0.4], dtype=dtype, requires_grad=True)
    smooth_axis = torch.tensor(
        [[1.0, 2.0, 1.0], [2.0, -1.0, 2.0], [-1.0, 3.0, 1.0]],
        dtype=dtype,
        requires_grad=True,
    )
    edges = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    weights = torch.tensor([0.25, 0.75], dtype=dtype)
    smooth = am.smoothness(smooth_strength, smooth_axis, edges, weights)
    packed_smooth = am.packed_smoothness(
        am.pack(smooth_strength, smooth_axis), edges, weights
    )
    smooth_strength_grad, smooth_axis_grad = torch.autograd.grad(
        smooth, (smooth_strength, smooth_axis)
    )
    strength_direction = torch.tensor([0.2, -0.3, 0.4], dtype=dtype)
    axis_direction = torch.tensor(
        [[0.3, 0.2, -0.1], [-0.2, 0.4, 0.1], [0.1, -0.3, 0.5]], dtype=dtype
    )
    analytic_smooth_derivative = float(
        (smooth_strength_grad * strength_direction).sum()
        + (smooth_axis_grad * axis_direction).sum()
    )

    def smooth_along(t: torch.Tensor) -> torch.Tensor:
        return am.smoothness(
            smooth_strength.detach() + t * strength_direction,
            smooth_axis.detach() + t * axis_direction,
            edges,
            weights,
        )

    smooth_fd = _central_difference(
        smooth_along, torch.tensor(0.0, dtype=dtype), torch.tensor(1.0, dtype=dtype)
    )
    checks = {
        "nonzero_strength_response": nonzero_strength_response,
        "nonzero_axis_response": nonzero_axis_response,
        "zero_strength_gradient_norm": float(
            torch.linalg.vector_norm(zero_strength_grad)
        ),
        "zero_axis_gradient_norm": float(torch.linalg.vector_norm(zero_axis_grad)),
        "B_symmetry_error": float(
            torch.max(torch.abs(B - B.transpose(-1, -2))).detach()
        ),
        "A_eigenvalue_max_error": float(
            torch.max(torch.abs(a_eigenvalues - expected_a_eigenvalues)).detach()
        ),
        "projection_min_strength": float(projected_strength.min()),
        "projection_axis_norm_error": float(
            torch.max(torch.abs(torch.linalg.vector_norm(projected_axis, dim=-1) - 1))
        ),
        "sign_invariance_error": sign_invariant,
        "linearized_axis_energy": float(linear_energy),
        "linearized_expected_energy": float(expected_linear_energy),
        "linearized_axis_relative_error": _relative_error(
            float(linear_energy), float(expected_linear_energy)
        ),
        "eigenaxis_rayleigh_relative_error": float(
            torch.max(torch.abs(canonical_axis_minimum - eigenvalues[:, 0]))
        ),
        "smoothness": float(smooth.detach()),
        "packed_smoothness_difference": float(
            torch.abs(smooth - packed_smooth).detach()
        ),
        "smoothness_finite_difference": smooth_fd,
        "smoothness_analytic_derivative": analytic_smooth_derivative,
        "smoothness_derivative_relative_error": _relative_error(
            smooth_fd, analytic_smooth_derivative
        ),
    }
    passed = (
        checks["nonzero_strength_response"] > 1e-8
        and checks["nonzero_axis_response"] > 1e-8
        and checks["zero_strength_gradient_norm"] > 1e-8
        and checks["zero_axis_gradient_norm"] < 1e-14
        and checks["B_symmetry_error"] < 1e-14
        and checks["A_eigenvalue_max_error"] < 1e-14
        and checks["projection_min_strength"] >= 0
        and checks["projection_axis_norm_error"] < 1e-14
        and checks["sign_invariance_error"] < 1e-14
        and checks["linearized_axis_energy"] < 0
        and checks["linearized_axis_relative_error"] < 1e-14
        and checks["eigenaxis_rayleigh_relative_error"] < 1e-14
        and checks["packed_smoothness_difference"] < 1e-14
        and checks["smoothness_derivative_relative_error"] < 1e-9
    )
    checks["passed"] = passed
    assert passed, checks
    cfg.output.write_text(json.dumps(checks, indent=2) + "\n")
    cherries.log_metrics(
        {
            f"activation/{key}": value
            for key, value in checks.items()
            if isinstance(value, (int, float))
        }
    )


if __name__ == "__main__":
    cherries.main(main)
