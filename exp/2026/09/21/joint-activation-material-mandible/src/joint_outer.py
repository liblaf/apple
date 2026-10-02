"""Small metric-projected quadratic solves for the 19 shared coordinates."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

import torch
from joint_fields import symmetric_coordinates, symmetric_matrices

Projection = Callable[[torch.Tensor], torch.Tensor]


def project_shared_free(
    value: torch.Tensor, constraints: Mapping[str, Any]
) -> torch.Tensor:
    """Project the 18 bulk coordinates and skin log-stiffness coordinate."""
    assert value.shape == (19,)
    epsilon = float(constraints["baseline_epsilon"])
    upper = float(constraints["baseline_upper_mu_multiple"])
    lower_multiplier, upper_multiplier = map(
        float, constraints["skin_multiplier_bounds"]
    )
    output = value.detach().clone()
    matrices = symmetric_matrices(output[:18].reshape(3, 6))
    eigenvalues, eigenvectors = torch.linalg.eigh(matrices)
    bounded = eigenvalues.clamp(min=-(1 - epsilon), max=upper)
    projected = (eigenvectors * bounded.unsqueeze(-2)) @ eigenvectors.transpose(-1, -2)
    output[:18] = symmetric_coordinates(projected).reshape(-1)
    output[18].clamp_(min=math.log(lower_multiplier), max=math.log(upper_multiplier))
    return output


def metric_projected_direction(  # noqa: PLR0915
    x: torch.Tensor,
    gradient: torch.Tensor,
    inverse_hessian: torch.Tensor,
    project: Projection,
    *,
    residual_tolerance: float = 1e-10,
    feasibility_tolerance: float = 1e-12,
    max_iterations: int = 20000,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Minimize the BFGS quadratic over a convex projected feasible set."""
    assert x.ndim == gradient.ndim == 1
    assert x.shape == gradient.shape
    assert inverse_hessian.shape == (len(x), len(x))
    assert residual_tolerance > 0
    assert feasibility_tolerance >= 0
    assert max_iterations > 0
    assert torch.isfinite(x).all()
    assert torch.isfinite(gradient).all()
    assert torch.isfinite(inverse_hessian).all()
    scale = max(1.0, float(inverse_hessian.abs().max()))
    symmetry_error = float(
        (inverse_hessian - inverse_hessian.transpose(-1, -2)).abs().max()
    )
    assert symmetry_error <= 1e-12 * scale
    cholesky, cholesky_info = torch.linalg.cholesky_ex(inverse_hessian)
    assert int(cholesky_info) == 0

    metric = torch.cholesky_inverse(cholesky)
    eigenvalues = torch.linalg.eigvalsh(metric)
    minimum = float(eigenvalues.min())
    maximum = float(eigenvalues.max())
    assert minimum > 0
    assert math.isfinite(maximum)
    lipschitz = maximum
    momentum = (math.sqrt(maximum) - math.sqrt(minimum)) / (
        math.sqrt(maximum) + math.sqrt(minimum)
    )

    def quadratic_gradient(value: torch.Tensor) -> torch.Tensor:
        return gradient + metric @ (value - x)

    def residual(value: torch.Tensor) -> float:
        mapped = project(value - quadratic_gradient(value) / lipschitz)
        return lipschitz * float((value - mapped).abs().max())

    def quadratic(value: torch.Tensor) -> float:
        difference = value - x
        return float(
            torch.dot(gradient, difference)
            + 0.5 * torch.dot(difference, metric @ difference)
        )

    projected_x = project(x)
    initial_feasibility_error = float((x - projected_x).abs().max())
    assert initial_feasibility_error <= feasibility_tolerance

    unconstrained = x - inverse_hessian @ gradient
    projected_unconstrained = project(unconstrained)
    feasibility_error = float((unconstrained - projected_unconstrained).abs().max())
    if feasibility_error <= feasibility_tolerance:
        solution = projected_unconstrained
        kkt_residual = residual(solution)
        assert kkt_residual <= residual_tolerance * 1.05
        direction = solution - x
        slope = float(torch.dot(gradient, direction))
        stationary = float(direction.abs().max()) <= feasibility_tolerance
        assert stationary or slope < 0
        return direction, {
            "method": "direct_unconstrained_feasible",
            "iterations": 0,
            "stationary": stationary,
            "kkt_residual_inf": kkt_residual,
            "residual_tolerance": residual_tolerance,
            "initial_feasibility_error_inf": initial_feasibility_error,
            "unconstrained_feasibility_error_inf": feasibility_error,
            "metric_min_eigenvalue": minimum,
            "metric_max_eigenvalue": maximum,
            "metric_condition_number": maximum / minimum,
            "slope": slope,
        }

    current = projected_x
    extrapolated = current.detach().clone()
    kkt_residual = math.inf
    restarted = 0
    monotone_restarts = 0
    iterations = 0
    for _iteration in range(1, max_iterations + 1):
        iterations += 1
        candidate = project(extrapolated - quadratic_gradient(extrapolated) / lipschitz)
        current_quadratic = quadratic(current)
        candidate_quadratic = quadratic(candidate)
        roundoff = 100 * torch.finfo(x.dtype).eps * max(1.0, abs(current_quadratic))
        if candidate_quadratic > current_quadratic + roundoff:
            extrapolated = current
            candidate = project(current - quadratic_gradient(current) / lipschitz)
            candidate_quadratic = quadratic(candidate)
            assert candidate_quadratic <= current_quadratic + roundoff
            restarted += 1
            monotone_restarts += 1
        kkt_residual = residual(candidate)
        if kkt_residual <= residual_tolerance:
            current = candidate
            break
        accelerated = candidate + momentum * (candidate - current)
        if float(torch.dot(candidate - current, extrapolated - candidate)) > 0:
            extrapolated = candidate
            restarted += 1
        else:
            extrapolated = accelerated
        current = candidate
    else:
        message = (
            "metric-projected quadratic did not meet its KKT residual: "
            f"{kkt_residual} > {residual_tolerance}"
        )
        raise RuntimeError(message)

    direction = current - x
    slope = float(torch.dot(gradient, direction))
    stationary = float(direction.abs().max()) <= feasibility_tolerance
    if not (stationary or slope < 0):
        message = f"metric-projected quadratic is not descent: slope={slope}"
        raise RuntimeError(message)
    return direction, {
        "method": "accelerated_metric_projection",
        "iterations": iterations,
        "restarts": restarted,
        "monotone_restarts": monotone_restarts,
        "stationary": stationary,
        "kkt_residual_inf": kkt_residual,
        "residual_tolerance": residual_tolerance,
        "initial_feasibility_error_inf": initial_feasibility_error,
        "unconstrained_feasibility_error_inf": feasibility_error,
        "metric_min_eigenvalue": minimum,
        "metric_max_eigenvalue": maximum,
        "metric_condition_number": maximum / minimum,
        "slope": slope,
    }
