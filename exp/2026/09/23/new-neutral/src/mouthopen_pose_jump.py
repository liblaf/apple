# ruff: noqa: EM101, PLR0915, SLF001, TRY003
"""Seed-only pose jump helpers for the MouthOpen initializer."""

from __future__ import annotations

import importlib.util
import itertools
import json
import logging
import time
from pathlib import Path
from typing import Any

import ipctk
import numpy as np
import pyvista as pv
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from joint_data import _collision_geometry
from joint_equilibrium import ForwardConvergenceError, rigid_displacement

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parents[4]
REFERENCE_SEED = ROOT / "exp/2026/09/22/neutral-newton/src/reference_seed.py"
spec = importlib.util.spec_from_file_location(
    "mouthopen_reference_seed", REFERENCE_SEED
)
assert spec is not None
assert spec.loader is not None
reference_seed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference_seed)
_separate_contact_planes = reference_seed._separate_contact_planes
repair = reference_seed.repair
LOG = logging.getLogger(__name__)


def _full_with_boundary(
    model: Any, candidate: torch.Tensor, fixed: torch.Tensor
) -> torch.Tensor:
    original = model.dof_map.fixed_values
    try:
        model.dof_map.fixed_values = fixed.detach().clone()
        result = model.dof_map.to_full(model.dof_map.to_free(candidate))
    finally:
        model.dof_map.fixed_values = original
    torch.testing.assert_close(
        result.flatten()[model.dof_map.fixed_indices], fixed, rtol=0, atol=0
    )
    return result


def _posed_points(
    points: np.ndarray, pivot: torch.Tensor, pose_value: torch.Tensor
) -> np.ndarray:
    value = torch.as_tensor(points, device=pivot.device, dtype=pivot.dtype)
    return (value + rigid_displacement(value, pivot, pose_value)).cpu().numpy()


def _surface(points: np.ndarray, faces: np.ndarray) -> pv.PolyData:
    return repair.poly(points, faces).compute_normals(
        cell_normals=True,
        point_normals=False,
        auto_orient_normals=True,
        consistent_normals=True,
        split_vertices=False,
    )


def _harmonic_increment(
    reference: np.ndarray,
    tets: np.ndarray,
    soft_ids: np.ndarray,
    fixed_ids: np.ndarray,
    surface_increment: np.ndarray,
) -> np.ndarray:
    """Extend a surface correction while retaining the existing interior seed."""
    edges = np.concatenate(
        [tets[:, pair] for pair in itertools.combinations(range(4), 2)]
    )
    dm = np.transpose(reference[tets[:, 1:]] - reference[tets[:, :1]], (0, 2, 1))
    volumes = np.linalg.det(dm) / 6
    assert np.all(volumes > 0)
    weights = np.tile(volumes, 6) / np.sum(
        (reference[edges[:, 0]] - reference[edges[:, 1]]) ** 2, axis=1
    )
    adjacency = sp.coo_matrix(
        (
            np.r_[weights, weights],
            (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]]),
        ),
        shape=(len(reference), len(reference)),
    ).tocsr()
    lap = sp.diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency
    boundary = np.union1d(soft_ids, fixed_ids)
    interior = np.setdiff1d(np.arange(len(reference)), boundary)
    increment = np.zeros_like(reference)
    increment[soft_ids] = surface_increment
    matrix = lap[interior][:, interior].tocsr()
    coupling = lap[interior][:, boundary].tocsr()
    preconditioner = sp.diags(1 / matrix.diagonal())
    for axis in range(3):
        value, info = spla.cg(
            matrix,
            -(coupling @ increment[boundary, axis]),
            M=preconditioner,
            rtol=1e-9,
            atol=0,
            maxiter=4000,
        )
        assert info == 0, info
        increment[interior, axis] = value
    return increment


def _contact_receipt(physics: Any, full: torch.Tensor) -> dict[str, Any]:
    model = physics.runtime.forward.model
    collision = model.collision
    assert collision is not None
    state = collision.state_at(full)
    receipt = collision.diagnostics(state, full)
    points = (collision.vertices + full[collision.indices]).numpy(force=True)
    intersects = bool(
        ipctk.has_intersections(collision.collision_mesh, points, ipctk.LBVH())
    )
    active = receipt["minimum_active_distance_m"]
    result = {
        **receipt,
        "no_intersections": not intersects,
        "minimum_active_gap_at_least_buffer": active is None
        or active >= collision.min_distance,
    }
    result["admitted"] = bool(
        result["contact_numerically_valid"]
        and result["no_intersections"]
        and result["minimum_active_gap_at_least_buffer"]
    )
    return result


@torch.no_grad()
def carry_near_mandible(
    physics: Any,
    u: torch.Tensor,
    old_jaw: torch.Tensor,
    new_jaw: torch.Tensor,
    pose: Any,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Carry free FEM vertices within ``dhat`` by the exact hinge increment."""
    started = time.perf_counter()
    model = physics.runtime.forward.model
    collision = model.collision
    assert collision is not None
    assert u.ndim == 2
    assert u.shape[1] == 3
    geometry = physics.full_skull.geometry
    fem_count = len(physics.points)
    assert u.shape[0] >= fem_count
    fixed = physics.boundary(pose(new_jaw))
    current = _full_with_boundary(model, u, physics.boundary(pose(old_jaw)))
    reference = torch.as_tensor(physics.points, device=u.device, dtype=u.dtype)
    current_points = (reference + current[:fem_count]).cpu().numpy()
    pivot = torch.as_tensor(geometry.mandible_pivot_m, device=u.device, dtype=u.dtype)
    mandible_old = _posed_points(geometry.mandible_points_m, pivot, pose(old_jaw))
    faces = np.asarray(geometry.mandible_faces, dtype=np.int64)
    obstacle = _surface(mandible_old, faces)
    _cell, closest = obstacle.find_closest_cell(
        current_points, return_closest_point=True
    )
    distance = np.linalg.norm(current_points - np.asarray(closest), axis=1)
    free = np.ones(fem_count, dtype=bool)
    fixed_ids = model.dof_map.fixed_indices.detach().cpu().numpy() // 3
    free[np.unique(fixed_ids[fixed_ids < fem_count])] = False
    carried = free & (distance <= collision.potential.dhat)
    x = torch.as_tensor(current_points[carried], device=u.device, dtype=u.dtype)
    old_pose, new_pose = pose(old_jaw), pose(new_jaw)
    # Undo the old rigid pose, then apply the new one. Rotation vectors cannot
    # be subtracted for general SE(3) motion because rotations do not commute.
    inverse_rotation = torch.cat((-old_pose[:3], torch.zeros_like(old_pose[3:])))
    unshifted = x - old_pose[3:]
    local = unshifted + rigid_displacement(unshifted, pivot, inverse_rotation)
    carried_points = local + rigid_displacement(local, pivot, new_pose)
    candidate = current.detach().clone()
    carried_ids = torch.as_tensor(
        np.flatnonzero(carried), device=u.device, dtype=torch.long
    )
    candidate[carried_ids] = carried_points - reference[carried_ids]
    candidate = _full_with_boundary(model, candidate, fixed)
    return candidate, {
        "method": "exact-relative-rigid-carry-for-free-fem-vertices-within-current-mandible-dhat",
        "dhat_m": float(collision.potential.dhat),
        "free_fem_vertices": int(free.sum()),
        "carried_fem_vertices": int(carried.sum()),
        "nearest_mandible_distance_min_m": float(distance.min()),
        "nearest_mandible_distance_max_m": float(distance.max()),
        "old_jaw": old_jaw.detach().cpu().tolist(),
        "new_jaw": new_jaw.detach().cpu().tolist(),
        "fixed_values_exact": True,
        "seconds": time.perf_counter() - started,
    }


@torch.no_grad()
def push_out(
    physics: Any,
    u: torch.Tensor,
    new_jaw: torch.Tensor,
    pose: Any,
    outputdir: Path,
    *,
    iterations: int = 8,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Push the pure-soft surface out of posed rigid obstacles and extend it."""
    assert iterations > 0
    started = time.perf_counter()
    model = physics.runtime.forward.model
    geometry = physics.full_skull.geometry
    fem_count = len(physics.points)
    fixed = physics.boundary(pose(new_jaw))
    full = _full_with_boundary(model, u, fixed)
    reference = np.asarray(physics.points, dtype=np.float64)
    soft_ids = np.asarray(geometry.soft_global_ids, dtype=np.int64)
    fixed_ids = np.asarray(geometry.fixed_global_ids, dtype=np.int64)
    assert not np.intersect1d(soft_ids, fixed_ids).size
    surface = reference[soft_ids] + full[:fem_count].cpu().numpy()[soft_ids]
    pivot = torch.as_tensor(geometry.mandible_pivot_m, device=u.device, dtype=u.dtype)
    obstacles = {
        "cranium": _surface(geometry.cranium_points_m, geometry.cranium_faces),
        "mandible": _surface(
            _posed_points(geometry.mandible_points_m, pivot, pose(new_jaw)),
            geometry.mandible_faces,
        ),
        "eyes": _surface(physics.eyes.points_m, physics.eyes.triangles),
    }
    actual_gate_buffer = float(model.collision.min_distance)
    requested_clearance = 1.001 * float(model.collision.potential.dhat)
    trace: list[dict[str, Any]] = []
    original_surface = surface.copy()
    outputdir.mkdir(parents=True, exist_ok=True)
    for step in range(iterations):
        correction = np.zeros_like(surface)
        measures: dict[str, Any] = {}
        for name, obstacle in obstacles.items():
            cell_ids, closest = obstacle.find_closest_cell(
                surface, return_closest_point=True
            )
            closest = np.asarray(closest, dtype=np.float64)
            direction = surface - closest
            distance = np.linalg.norm(direction, axis=1)
            normals = np.asarray(obstacle.cell_normals)[np.asarray(cell_ids)]
            nonzero = distance > 1e-12
            direction[nonzero] /= distance[nonzero, None]
            direction[~nonzero] = normals[~nonzero]
            gap = distance
            if name == "eyes":
                signed = repair.signed_distance(surface, obstacle)
                direction[signed < 0] *= -1
                gap = signed
            correction += np.maximum(requested_clearance - gap, 0)[:, None] * direction
            pairs, *_ = _collision_geometry(
                _surface(surface, geometry.soft_faces), obstacle
            )
            if len(pairs):
                _separate_contact_planes(
                    surface,
                    geometry.soft_faces,
                    pairs,
                    obstacle,
                    requested_clearance,
                )
            measures[name] = {
                "minimum_sample_gap_m": float(gap.min()),
                "intersections": len(pairs),
            }
        surface += correction
        trace.append(
            {
                "iteration": step + 1,
                "maximum_surface_update_m": float(
                    np.linalg.norm(correction, axis=1).max()
                ),
                "obstacles": measures,
            }
        )
        (outputdir / "pose-jump-projection-trace.json").write_text(
            json.dumps(trace, indent=2, allow_nan=False) + "\n"
        )
        LOG.info(
            "Pose-jump pushout %d: max update %.6g m",
            step + 1,
            trace[-1]["maximum_surface_update_m"],
        )
        if all(
            entry["intersections"] == 0
            and entry["minimum_sample_gap_m"] >= requested_clearance
            for entry in measures.values()
        ):
            break
    increment = _harmonic_increment(
        reference,
        np.asarray(physics.tets, dtype=np.int64),
        soft_ids,
        fixed_ids,
        surface - original_surface,
    )
    candidate = full.detach().clone()
    candidate[:fem_count] += torch.as_tensor(increment, device=u.device, dtype=u.dtype)
    candidate = _full_with_boundary(model, candidate, fixed)
    contact = _contact_receipt(physics, candidate)
    geometry_receipt = physics.metrics(candidate[:fem_count])
    receipt = {
        "method": "posed-obstacle-closest-feature-pushout-with-harmonic-increment",
        "iterations": trace,
        "contact": contact,
        "geometry": geometry_receipt,
        "requested_projection_clearance_m": requested_clearance,
        "actual_contact_gate_buffer_m": actual_gate_buffer,
        "fixed_values_exact": True,
        "seed_only": True,
        "seconds": time.perf_counter() - started,
    }
    (outputdir / "pose-jump-pushout.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    )
    if not contact["admitted"]:
        raise ForwardConvergenceError(
            "pose-jump pushout did not reach the contact gate", receipt=receipt
        )
    return candidate, receipt


__all__ = ["carry_near_mandible", "push_out"]
