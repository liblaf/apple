# ruff: noqa: PLR0915, PT018
"""Fixed-state finite differences for the MouthOpen implicit pullback."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

import torch


def _target_objective(
    *,
    target: torch.Tensor,
    target_ids: torch.Tensor,
    weights: torch.Tensor,
    scale2: torch.Tensor | float,
) -> Callable[[torch.Tensor], torch.Tensor]:
    assert target.ndim == 2 and target.shape[1] == 3
    assert target_ids.ndim == 1 and target_ids.shape[0] == target.shape[0]
    assert weights.shape == target.shape[:1]
    denominator = torch.as_tensor(scale2, device=target.device, dtype=target.dtype)
    assert denominator.ndim == 0 and bool(torch.isfinite(denominator))
    assert float(denominator) > 0

    def objective(u: torch.Tensor) -> torch.Tensor:
        return (
            weights[:, None] * (u[target_ids] - target).square()
        ).sum() / denominator

    return objective


def _central_difference(
    evaluate: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    q: torch.Tensor,
    jaw: torch.Tensor,
    q_direction: torch.Tensor,
    jaw_direction: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    assert epsilon > 0
    plus = evaluate(q + epsilon * q_direction, jaw + epsilon * jaw_direction)
    minus = evaluate(q - epsilon * q_direction, jaw - epsilon * jaw_direction)
    return (plus - minus) / (2 * epsilon)


def _comparison(
    predicted: torch.Tensor, finite_difference: torch.Tensor
) -> dict[str, float]:
    assert predicted.ndim == finite_difference.ndim == 0
    assert bool(torch.isfinite(predicted)) and bool(torch.isfinite(finite_difference))
    absolute = float((predicted - finite_difference).abs())
    scale = max(abs(float(predicted)), abs(float(finite_difference)), 1e-12)
    return {
        "predicted": float(predicted),
        "finite_difference": float(finite_difference),
        "absolute_error": absolute,
        "relative_error": absolute / scale,
    }


def check_joint_pullback(
    physics: Any,
    runtime: Any,
    materials: Callable[[torch.Tensor], Mapping[str, Mapping[str, torch.Tensor]]],
    pose: Callable[[torch.Tensor], torch.Tensor],
    q: torch.Tensor,
    jaw: torch.Tensor,
    u: torch.Tensor,
    *,
    grads: tuple[torch.Tensor, torch.Tensor],
    objective: Callable[[torch.Tensor], torch.Tensor] | None = None,
    control_objective: Callable[[torch.Tensor], torch.Tensor] | None = None,
    target: torch.Tensor | None = None,
    target_ids: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    scale2: torch.Tensor | float | None = None,
    epsilons: tuple[float, ...] = (1e-4, 1e-5, 1e-6),
) -> dict[str, Any]:
    """Compare the existing implicit VJP to fixed-state Lagrangian differences.

    This does not solve a neighboring equilibrium.  It keeps free displacement
    coordinates fixed, substitutes each perturbed rigid boundary through the
    model's fixed-DOF map, and evaluates ``L(u) + R(q) + p_free dot r_free(u)``. The
    stored ``p_free`` is the custom joint implicit adjoint for ``MouthOpen``.
    """
    assert epsilons
    assert all(math.isfinite(value) and value > 0 for value in epsilons)
    assert len(set(epsilons)) == len(epsilons)
    assert len({f"{value:.0e}" for value in epsilons}) == len(epsilons)
    assert q.ndim == 2 and q.shape[1] == 6
    assert jaw.shape in {(1,), (6,)}
    assert u.ndim == 2 and u.shape[1] == 3
    gradient_q, gradient_jaw = grads
    assert gradient_q.shape == q.shape and gradient_jaw.shape == jaw.shape
    assert bool(torch.isfinite(q).all()) and bool(torch.isfinite(jaw).all())
    assert bool(torch.isfinite(u).all())
    assert bool(torch.isfinite(gradient_q).all())
    assert bool(torch.isfinite(gradient_jaw).all())
    if objective is None:
        assert target is not None and target_ids is not None and weights is not None
        assert scale2 is not None
        objective = _target_objective(
            target=target,
            target_ids=target_ids,
            weights=weights,
            scale2=scale2,
        )

    model = runtime.forward.model
    p_free = runtime.warm_adjoints["MouthOpen"].detach()
    free_u = model.dof_map.to_free(u.detach()).clone()
    assert p_free.shape == free_u.shape
    assert bool(torch.isfinite(p_free).all())

    generator = torch.Generator(device=q.device).manual_seed(20260923)
    q_direction = torch.randn(
        q.shape, device=q.device, dtype=q.dtype, generator=generator
    )
    q_direction /= torch.linalg.vector_norm(q_direction)
    assert bool(torch.count_nonzero(q_direction[:, 3:]))
    jaw_direction = torch.ones_like(jaw)
    jaw_direction /= torch.linalg.vector_norm(jaw_direction)
    coordinate_names = (
        ("jaw",)
        if jaw.numel() == 1
        else (
            "rotation_x",
            "rotation_y",
            "rotation_z",
            "translation_x",
            "translation_y",
            "translation_z",
        )
    )

    original_materials = model.get_materials()
    original_fixed = model.dof_map.fixed_values
    try:

        def lagrangian(q_value: torch.Tensor, jaw_value: torch.Tensor) -> torch.Tensor:
            model.set_materials(materials(q_value))
            fixed = physics.boundary(pose(jaw_value))
            assert fixed.shape == model.dof_map.fixed_values.shape
            model.dof_map.fixed_values = fixed
            # Rebuild the full state from unchanged free coordinates so jaw
            # differences enter through the actual fixed-coordinate map.
            full_u = model.dof_map.to_full(free_u)
            state = model.State(u=full_u)
            if model.collision is not None:
                state.collision = model.collision.state_at(full_u)
            residual = model.dof_map.to_free_grad(model.grad(state))
            value = objective(full_u) + torch.dot(p_free, residual)
            if control_objective is not None:
                value = value + control_objective(q_value)
            assert value.ndim == 0 and bool(torch.isfinite(value))
            return value

        predicted_q = torch.sum(gradient_q * q_direction)
        predicted_jaw = torch.sum(gradient_jaw * jaw_direction)
        result = {
            "schema": "mouthopen-fixed-state-lagrangian-pullback-v1",
            "definition": (
                "L(u_fixed_free, fixed(jaw)) + R(q) + p_free dot r_free; "
                "central differences do not re-solve equilibrium. This checks "
                "the fixed-state Lagrangian pullback, not a fully resolved "
                "neighboring-equilibrium finite difference."
            ),
            "adjoint_key": "MouthOpen",
            "direct_control_objective_included": control_objective is not None,
            "epsilons": list(epsilons),
            "free_coordinates_held_fixed": True,
            "fixed_coordinates_rebuilt_from_pose": True,
            "fresh_contact_state_per_evaluation": model.collision is not None,
            "q_direction": {
                "norm": float(torch.linalg.vector_norm(q_direction)),
                "offdiagonal_l2": float(torch.linalg.vector_norm(q_direction[:, 3:])),
            },
            "jaw_direction": {"norm": float(torch.linalg.vector_norm(jaw_direction))},
            "q": {},
            "jaw": {},
            "jaw_coordinates": {
                name: {
                    "index": index,
                    "basis_norm": 1.0,
                    "coordinate_space": "normalized jaw parameter",
                }
                for index, name in enumerate(coordinate_names)
            },
        }
        for epsilon in epsilons:
            key = f"epsilon_{epsilon:.0e}"
            q_fd = _central_difference(
                lagrangian,
                q,
                jaw,
                q_direction,
                torch.zeros_like(jaw_direction),
                epsilon,
            )
            jaw_fd = _central_difference(
                lagrangian,
                q,
                jaw,
                torch.zeros_like(q_direction),
                jaw_direction,
                epsilon,
            )
            result["q"][key] = _comparison(predicted_q, q_fd)
            result["jaw"][key] = _comparison(predicted_jaw, jaw_fd)
            for index, name in enumerate(coordinate_names):
                basis = torch.zeros_like(jaw)
                basis[index] = 1
                coordinate_fd = _central_difference(
                    lagrangian,
                    q,
                    jaw,
                    torch.zeros_like(q_direction),
                    basis,
                    epsilon,
                )
                result["jaw_coordinates"][name][key] = _comparison(
                    gradient_jaw[index], coordinate_fd
                )
        return result
    finally:
        model.set_materials(original_materials)
        model.dof_map.fixed_values = original_fixed


__all__ = ["check_joint_pullback"]
