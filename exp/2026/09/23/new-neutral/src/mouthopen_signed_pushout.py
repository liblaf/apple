"""Signed closest-surface repair against the verified closed rigid obstacles."""

# ruff: noqa: C901, EM101, PLR0915, TRY003
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from joint_common import write_json
from joint_data import _collision_geometry
from joint_equilibrium import ForwardConvergenceError
from mouthopen_collision_off_seed import (
    _check_deadline,
    _contact,
    _poly,
    _save,
    _VolumeExtension,
)
from mouthopen_tet_policy import geometry_metrics
from vtkmodules.vtkFiltersCore import vtkImplicitPolyDataDistance
from vtkmodules.vtkFiltersModeling import vtkSelectEnclosedPoints


class _ClosedSolidQuery:
    """Use ray containment for sign; nearest-feature normals alone misclassify holes."""

    def __init__(self, surface: Any) -> None:
        self.distance = vtkImplicitPolyDataDistance()
        self.distance.SetInput(surface)
        self.enclosed = vtkSelectEnclosedPoints()
        self.enclosed.SetTolerance(1e-9)
        self.enclosed.Initialize(surface)

    def evaluate(self, point: np.ndarray) -> tuple[float, np.ndarray]:
        closest = np.empty(3)
        distance = abs(self.distance.EvaluateFunctionAndGetClosestPoint(point, closest))
        inside = bool(self.enclosed.IsInsideSurface(point))
        if distance > 1e-12:
            direction = (closest - point if inside else point - closest) / distance
        else:
            direction = np.empty(3)
            self.distance.EvaluateGradient(point, direction)
            direction /= np.linalg.norm(direction)
            if self.enclosed.IsInsideSurface(point + 1e-7 * direction):
                direction *= -1
            assert not self.enclosed.IsInsideSurface(point + 1e-7 * direction)
        return -distance if inside else distance, direction

    def EvaluateFunction(self, point: np.ndarray) -> float:
        return self.evaluate(point)[0]

    def EvaluateGradient(self, point: np.ndarray, gradient: np.ndarray) -> None:
        gradient[:] = self.evaluate(point)[1]


def signed_repair(
    physics: Any,
    old: torch.Tensor,
    candidate: torch.Tensor,
    fixed: torch.Tensor,
    pose: torch.Tensor,
    q: torch.Tensor,
    output: Path,
    receipt: dict,
    *,
    iterations: int,
    deadline: float | None,
) -> torch.Tensor:
    model = physics.runtime.forward.model
    collision = model.collision
    geometry = physics.full_skull.geometry
    reference = np.asarray(physics.points)
    soft_ids = np.asarray(geometry.soft_global_ids)
    soft_faces = np.asarray(geometry.soft_faces)
    fixed_mask = np.asarray(physics.mesh.point_data["IsFixed"], dtype=bool)
    free_surface = ~fixed_mask[soft_ids]
    fixed_ids = np.flatnonzero(fixed_mask)
    counts = [
        len(soft_ids),
        len(geometry.cranium_points_m),
        len(geometry.mandible_points_m),
        len(physics.eyes.points_m),
    ]
    offsets = np.r_[0, np.cumsum(counts)]
    assert offsets[-1] == len(collision.vertices)
    np.testing.assert_array_equal(
        collision.indices[: len(soft_ids)].cpu().numpy(), soft_ids
    )
    current_points = (collision.vertices + candidate[collision.indices]).numpy(
        force=True
    )
    old_points = (collision.vertices + old[collision.indices]).numpy(force=True)
    obstacles = []
    topology = []
    for index, (name, faces) in enumerate(
        (
            ("cranium", geometry.cranium_faces),
            ("mandible", geometry.mandible_faces),
            ("eyes", physics.eyes.triangles),
        ),
        1,
    ):
        original = _poly(
            current_points[offsets[index] : offsets[index + 1]], np.asarray(faces)
        )
        closed = original.clean(tolerance=0.0, absolute=True)
        assert closed.n_open_edges == 0
        assert closed.n_cells == original.n_cells
        closed = closed.compute_normals(
            cell_normals=True,
            point_normals=False,
            consistent_normals=True,
            auto_orient_normals=True,
            split_vertices=False,
        )
        signed = _ClosedSolidQuery(closed)
        obstacles.append((name, closed, signed))
        topology.append(
            {
                "name": name,
                "original_points": original.n_points,
                "query_points": closed.n_points,
                "triangles": closed.n_cells,
                "original_open_edges": original.n_open_edges,
                "query_open_edges": closed.n_open_edges,
            }
        )
    extension = _VolumeExtension(
        reference, np.asarray(physics.tets), soft_ids, fixed_ids
    )
    clearance = 1.001 * float(collision.potential.dhat)
    projection = receipt["projection"] = {
        "method": "closed-solid-signed-distance-vertex-and-barycentric-contact-repair",
        "collision_geometry_unchanged": True,
        "inside_outside_method": "vtkSelectEnclosedPoints ray containment, tolerance 1e-9; unsigned nearest-feature distance",
        "exact_seam_welding_for_queries_only": True,
        "topology": topology,
        "requested_clearance_m": clearance,
        "contact_gate_clearance_m": float(collision.min_distance),
        "free_surface_vertices": int(free_surface.sum()),
        "trace": [],
        "source_surface_vertex_count": len(old_points[: len(soft_ids)]),
    }

    def values(query: Any, points: np.ndarray) -> np.ndarray:
        return np.asarray([query.EvaluateFunction(point) for point in points])

    def normal(query: Any, point: np.ndarray) -> np.ndarray:
        gradient = np.empty(3)
        query.EvaluateGradient(point, gradient)
        length = np.linalg.norm(gradient)
        assert np.isfinite(length)
        assert length > 0
        return gradient / length

    for iteration in range(iterations + 1):
        _check_deadline(deadline)
        surface = (
            reference[soft_ids]
            + candidate[: len(reference)].numpy(force=True)[soft_ids]
        )
        contact = _contact(collision, candidate)
        gaps = {name: values(query, surface) for name, _, query in obstacles}
        row = {
            "iteration": iteration,
            "contact": contact,
            "retained_geometry": geometry_metrics(physics, candidate),
            "inside_vertices": {
                name: int((gap < 0).sum()) for name, gap in gaps.items()
            },
            "minimum_signed_gaps_m": {
                name: float(gap.min()) for name, gap in gaps.items()
            },
        }
        projection["trace"].append(row)
        _save(output / "projection-latest.pt", candidate, pose, q)
        write_json(output / "summary.json", receipt)
        if contact["admitted"] and all(
            np.all(gap >= collision.min_distance) for gap in gaps.values()
        ):
            return candidate
        if iteration == iterations:
            break
        projected = surface.copy()
        counts = {}
        for name, obstacle, query in obstacles:
            _check_deadline(deadline)
            gap = values(query, projected)
            moving = np.flatnonzero(free_surface & (gap < clearance))
            for vertex in moving:
                projected[vertex] += (clearance - gap[vertex]) * normal(
                    query, projected[vertex]
                )
            # Vertices can be outside while a triangle still cuts a curved
            # obstacle. Apply the smallest nodal correction at each contact
            # midpoint, distributed through its free barycentric coordinates.
            pairs, _, midpoints, _ = _collision_geometry(
                _poly(projected, soft_faces), obstacle
            )
            for (soft_face, _), midpoint in zip(pairs, midpoints, strict=True):
                ids = soft_faces[soft_face]
                triangle = projected[ids]
                edge = (triangle[1:] - triangle[0]).T
                coordinates = np.linalg.lstsq(edge, midpoint - triangle[0], rcond=None)[
                    0
                ]
                weights = np.maximum(np.r_[1 - coordinates.sum(), coordinates], 0)
                weights /= weights.sum()
                point = weights @ triangle
                distance = float(query.EvaluateFunction(point))
                if distance >= clearance:
                    continue
                free_weights = weights * free_surface[ids]
                denominator = float(free_weights @ free_weights)
                assert denominator > 0
                projected[ids] += (
                    (clearance - distance)
                    * free_weights[:, None]
                    / denominator
                    * normal(query, point)
                )
            counts[name] = {
                "vertex_updates": len(moving),
                "triangle_contact_pairs": len(pairs),
            }
        np.testing.assert_array_equal(projected[~free_surface], surface[~free_surface])
        increment, residuals = extension.extend(projected - surface, deadline)
        row.update(
            obstacles=counts,
            maximum_surface_increment_m=float(
                np.linalg.norm(projected - surface, axis=1).max()
            ),
            extension_absolute_residuals=residuals,
        )
        candidate = candidate.detach().clone()
        candidate[: len(reference)] += torch.as_tensor(
            increment, device=candidate.device, dtype=candidate.dtype
        )
        candidate.flatten()[model.dof_map.fixed_indices] = fixed
        torch.testing.assert_close(
            candidate.flatten()[model.dof_map.fixed_indices], fixed, rtol=0, atol=0
        )
    raise ForwardConvergenceError(
        "signed-distance pushout exhausted its contact repair budget",
        receipt=projection,
    )
