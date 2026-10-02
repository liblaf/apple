# ruff: noqa: EM102, FBT001, TRY003
"""Fail-fast admission checks for the smile solver-comparison collision model.

The comparison must use the already validated full-source contact construction:
the complete cranium and mandible plus both registered source eyes.  This module
does not construct alternative geometry or modify the physical model; it only
audits the assembled ``RigidEyeJointPhysics`` before a fit starts.
"""

from __future__ import annotations

from typing import Any

import ipctk
import numpy as np
import torch


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(f"smile collision admission failed: {message}")


def audit_required_collision(physics: Any) -> dict[str, Any]:
    """Require complete source bones and fixed registered eyes for a smile fit.

    Returns a JSON-safe receipt that the fit harness records for each solver arm.
    The contact law deliberately contains soft-versus-rigid pairs only: it does
    not claim soft-soft or rigid-rigid collision coverage.
    """
    full = getattr(physics, "full_skull", None)
    _require(full is not None, "physics has no full-source contact adapter")
    geometry = getattr(full, "geometry", None)
    eyes = getattr(full, "eyes", None)
    _require(geometry is not None, "full-source geometry is missing")
    _require(eyes is not None, "registered eye geometry is missing")

    model = physics.runtime.forward.model
    collision = model.collision
    _require(collision is not None, "model collision is disabled")
    definition = physics.contact_definition
    _require(
        definition.get("schema") == "joint-full-source-contact-with-rigid-eyes-v1",
        "contact is not the full-source bone-and-eye model",
    )
    base = definition.get("base_full_skull", {})
    config = definition.get("config", {})
    _require(
        base.get("schema") == "joint-full-skull-physics-binding-v1",
        "complete cranium/mandible binding is absent",
    )
    _require(config.get("enabled") is True, "source-bone contact is disabled")
    _require(
        config.get("surface_selection") == "pure-soft-vs-complete-source-bones",
        "contact does not use the complete source-bone surface",
    )
    _require(
        config.get("attachment_policy") == "no-source-triangle-exclusions",
        "source triangles were excluded from the contact surface",
    )
    _require(
        base["contact"].get("complete_source_triangles_retained") is True,
        "source-bone triangle completeness is not certified",
    )
    _require(
        base["contact"].get("excluded_source_triangles") == 0,
        "source-bone contact reports excluded triangles",
    )
    _require(definition.get("soft_eye_contact") is True, "soft-eye contact is disabled")
    _require(
        definition.get("eyes_fixed_at_registered_neutral_pose") is True,
        "eyes are not fixed at their registered source pose",
    )
    _require(
        definition.get("soft_soft_contact") is False, "unexpected soft-soft contact"
    )
    _require(
        definition.get("rigid_rigid_contact") is False,
        "unexpected rigid-rigid contact",
    )

    expected_ids = np.concatenate(
        (
            geometry.soft_global_ids,
            geometry.cranium_global_ids,
            geometry.mandible_global_ids,
            full.eye_global_ids,
        )
    )
    actual_ids = collision.indices.detach().cpu().numpy()
    _require(
        np.array_equal(actual_ids, expected_ids),
        "collision vertex map differs from the full soft/bone/eye assembly",
    )
    _require(
        tuple(collision.vertices.shape) == (len(expected_ids), 3),
        "collision vertex array has an unexpected shape",
    )
    _require(
        int(definition["collision_vertices"]) == len(expected_ids),
        "receipt collision-vertex count differs from the assembled model",
    )
    for key, count in {
        "soft_triangles": len(geometry.soft_faces),
        "cranium_triangles": len(geometry.cranium_faces),
        "mandible_triangles": len(geometry.mandible_faces),
        "eye_triangles": len(eyes.triangles),
    }.items():
        _require(definition.get(key) == count, f"{key} differs from bound geometry")

    # All appended coordinates must be prescribed.  The mandible is the only
    # moving obstacle, through the inherited rigid-pose boundary mapping.
    appended_begin = geometry.fem_node_count * 3
    expected_fixed = torch.arange(
        appended_begin,
        full.full_node_count * 3,
        device=model.dof_map.fixed_indices.device,
        dtype=model.dof_map.fixed_indices.dtype,
    )
    _require(
        bool(torch.isin(expected_fixed, model.dof_map.fixed_indices).all()),
        "one or more appended bone/eye coordinates are not fixed",
    )
    points = physics.points_t
    pose = torch.zeros(6, device=points.device, dtype=points.dtype)
    moving_pose = pose.clone()
    moving_pose[0] = 1e-3
    neutral = full.full_boundary_displacement(points, physics.jaw_t, pose)
    moved = full.full_boundary_displacement(points, physics.jaw_t, moving_pose)
    cranium = slice(
        geometry.fem_node_count, geometry.fem_node_count + geometry.cranium_node_count
    )
    mandible = slice(cranium.stop, cranium.stop + geometry.mandible_node_count)
    eye = slice(mandible.stop, mandible.stop + eyes.node_count)
    _require(
        bool(torch.count_nonzero(neutral[cranium]) == 0),
        "cranium moves at neutral pose",
    )
    _require(bool(torch.count_nonzero(neutral[eye]) == 0), "eyes move at neutral pose")
    _require(
        bool(torch.count_nonzero(moved[cranium]) == 0),
        "cranium moves with mandible pose",
    )
    _require(bool(torch.count_nonzero(moved[eye]) == 0), "eyes move with mandible pose")
    _require(
        bool(torch.count_nonzero(moved[mandible]) > 0), "mandible does not follow pose"
    )

    return {
        "schema": "smile-required-collision-audit-v1",
        "success": True,
        "enabled": True,
        "surface_pairs": "soft-versus-(complete cranium, complete mandible, fixed eyes)",
        "counts": {
            "soft_vertices": len(geometry.soft_global_ids),
            "soft_triangles": len(geometry.soft_faces),
            "cranium_vertices": int(geometry.cranium_node_count),
            "cranium_triangles": len(geometry.cranium_faces),
            "mandible_vertices": int(geometry.mandible_node_count),
            "mandible_triangles": len(geometry.mandible_faces),
            "eye_vertices": int(eyes.node_count),
            "eye_triangles": len(eyes.triangles),
            "collision_vertices": len(expected_ids),
        },
        "coverage": {
            "complete_source_bones": True,
            "excluded_source_triangles": 0,
            "soft_cranium": True,
            "soft_mandible": True,
            "soft_eyes": True,
            "soft_soft": False,
            "rigid_rigid": False,
        },
        "motion": {
            "cranium": "fixed",
            "eyes": "fixed registered neutral source pose",
            "mandible": "fixed DOFs with differentiable rigid-pose displacement",
            "nonzero_probe_rotation_rad": 1e-3,
        },
        "contact": {
            "dhat_m": float(config["dhat_m"]),
            "ccd_min_distance_m": float(config["ccd_min_distance_m"]),
            "friction": config["friction"],
        },
    }


def audit_collision_state(
    physics: Any, fem_displacement: torch.Tensor, pose: torch.Tensor
) -> dict[str, Any]:
    """Audit one loaded or fitted state against every required rigid obstacle.

    This is intentionally a geometry gate rather than an equilibrium claim.
    The caller records it for the shared neutral seed and for each final arm.
    """
    audit_required_collision(physics)
    full = physics.full_skull
    geometry, eyes = full.geometry, full.eyes
    _require(
        tuple(fem_displacement.shape) == (geometry.fem_node_count, 3),
        "state does not contain exactly the original FEM nodes",
    )
    _require(tuple(pose.shape) == (6,), "rigid mandible pose has wrong shape")
    _require(bool(torch.isfinite(fem_displacement).all()), "state is non-finite")
    _require(bool(torch.isfinite(pose).all()), "mandible pose is non-finite")
    full_displacement = full.extend_seed(fem_displacement, pose)
    collision = physics.runtime.forward.model.collision
    assert collision is not None
    state = collision.state_at(full_displacement)
    positions = (collision.vertices + full_displacement[collision.indices]).numpy(
        force=True
    )
    intersections = bool(
        ipctk.has_intersections(collision.collision_mesh, positions, ipctk.LBVH())
    )
    receipt = collision.diagnostics(state, full_displacement)
    gap = receipt["minimum_active_distance_m"]
    feasible_gap = gap is None or gap >= collision.min_distance
    return {
        "schema": "smile-collision-state-audit-v1",
        "success": True,
        "collision_geometry": "complete cranium, mandible, and fixed eyes",
        "soft_rigid_intersection_free": not intersections,
        "minimum_active_distance_m": gap,
        "minimum_required_distance_m": float(collision.min_distance),
        "minimum_gap_gate": feasible_gap,
        "active_contact_count": int(receipt["active_contact_count"]),
        "contact_numerically_valid": bool(receipt["contact_numerically_valid"]),
        "state_feasible": bool(
            not intersections and feasible_gap and receipt["contact_numerically_valid"]
        ),
        "motion": {
            "cranium_displacement_max_m": float(
                full_displacement[
                    geometry.fem_node_count : geometry.fem_node_count
                    + geometry.cranium_node_count
                ]
                .abs()
                .max()
            ),
            "eyes_displacement_max_m": float(
                full_displacement[-eyes.node_count :].abs().max()
            ),
        },
    }


__all__ = ["audit_collision_state", "audit_required_collision"]
