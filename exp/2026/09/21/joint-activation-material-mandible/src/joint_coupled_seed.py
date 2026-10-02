# ruff: noqa: EM101, PT018, TRY003
"""CCD-admitted coupled seeds for expression predictor-corrector solves."""

from __future__ import annotations

import math
from typing import Any

import torch
from joint_coupled_predictor import (
    audit_coupled_motion,
    equilibrium_predictor,
    rotation_sagitta,
)
from joint_equilibrium import ForwardConvergenceError


def _fixed_axis_pose(pose: torch.Tensor, axis: torch.Tensor) -> None:
    assert pose.shape == (6,) and axis.shape == (3,)
    assert bool(torch.isfinite(pose).all() and torch.isfinite(axis).all())
    torch.testing.assert_close(
        torch.linalg.vector_norm(axis), torch.ones((), device=axis.device)
    )
    torch.testing.assert_close(pose[3:], torch.zeros_like(pose[3:]), atol=1e-14, rtol=0)
    rotation = pose[:3]
    component = torch.dot(rotation, axis) * axis
    torch.testing.assert_close(rotation, component, atol=1e-14, rtol=1e-12)


def _maximum_axis_radius(
    vertices: torch.Tensor, axis: torch.Tensor, pivot: torch.Tensor
) -> float:
    assert vertices.ndim == 2 and vertices.shape[1] == 3
    assert pivot.shape == (3,) and bool(torch.isfinite(vertices).all())
    relative = vertices - pivot
    perpendicular = relative - (relative @ axis).unsqueeze(-1) * axis
    return float(torch.linalg.vector_norm(perpendicular, dim=-1).max())


@torch.no_grad()
def predict_expression_seed(
    physics: Any,
    *,
    old_stress: torch.Tensor,
    new_stress: torch.Tensor,
    old_pose: torch.Tensor,
    new_pose: torch.Tensor,
    seed: torch.Tensor,
    axis: torch.Tensor,
    pivot: torch.Tensor,
    linear_rtol: float,
    residual_scale: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """Predict and CCD-admit a full-contact seed for a new stress and hinge pose.

    The returned FEM displacement is only an initialization for the normal IPC
    equilibrium corrector.  It changes no physical material, boundary, or
    collision configuration.
    """
    assert 0 < linear_rtol <= 1e-7 and 0 < residual_scale <= 1
    _fixed_axis_pose(old_pose, axis)
    _fixed_axis_pose(new_pose, axis)
    assert pivot.shape == (3,) and bool(torch.isfinite(pivot).all())
    assert seed.ndim == 2 and seed.shape[1] == 3 and bool(torch.isfinite(seed).all())
    assert old_stress.shape == new_stress.shape
    assert bool(torch.isfinite(old_stress).all() and torch.isfinite(new_stress).all())
    torch.testing.assert_close(old_stress, old_stress.transpose(-1, -2), atol=0, rtol=0)
    torch.testing.assert_close(new_stress, new_stress.transpose(-1, -2), atol=0, rtol=0)

    runtime = physics.runtime
    model = runtime.forward.model
    collision = model.collision
    assert collision is not None
    old_full = physics.full_skull.extend_seed(seed, old_pose)
    fixed_indices = model.dof_map.fixed_indices
    torch.testing.assert_close(
        old_full.flatten()[fixed_indices],
        physics.boundary(old_pose),
        atol=1e-14,
        rtol=0,
    )
    one = torch.ones((), dtype=seed.dtype, device=seed.device)
    old_materials = physics.expression_materials(
        skin_multiplier=one, active_stress=old_stress
    )
    new_materials = physics.expression_materials(
        skin_multiplier=one, active_stress=new_stress
    )
    predicted_full, predictor = equilibrium_predictor(
        model=model,
        solver=runtime.solver,
        displacement=old_full,
        old_materials=old_materials,
        new_materials=new_materials,
        fixed_target=physics.boundary(new_pose),
        linear_rtol=linear_rtol,
        residual_scale=residual_scale,
    )
    delta_rotation = new_pose[:3] - old_pose[:3]
    delta_angle = float(torch.linalg.vector_norm(delta_rotation))
    assert math.isfinite(delta_angle) and delta_angle <= math.pi
    radius = _maximum_axis_radius(collision.vertices, axis, pivot)
    margin = rotation_sagitta(radius, delta_angle)
    geometry = audit_coupled_motion(
        collision, old_full, predicted_full, rotation_margin_m=margin
    )
    receipt = {
        "success": bool(geometry["admitted"]),
        "method": "equilibrium-tangent-coupled-seed-with-curved-hinge-ccd",
        "linear_rtol": linear_rtol,
        "residual_scale": residual_scale,
        "predictor": predictor,
        "geometry": geometry,
        "maximum_collision_vertex_axis_radius_m": radius,
        "delta_hinge_angle_rad": delta_angle,
        "rotation_sagitta_bound_m": margin,
        "physical_model_changed": False,
        "seed_only": True,
    }
    if not geometry["admitted"]:
        raise ForwardConvergenceError(
            "coupled predictor seed failed full contact admission", receipt=receipt
        )
    fem_nodes = physics.full_skull.geometry.fem_node_count
    return predicted_full[:fem_nodes].detach().clone(), receipt


__all__ = ["predict_expression_seed"]
