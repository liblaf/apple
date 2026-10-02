"""Collision guard for the solver's straight boundary-vertex motion.

This adds no energy or force. It checks the pure FEM cranial and mandibular
faces, not the complete registered source bones or an anatomical rotation arc.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import ipctk
import numpy as np
from joint_data import _rotation_matrix


def linear_bone_step(mesh: Any, start: np.ndarray, end: np.ndarray) -> dict[str, Any]:
    """Check a linear vertex trajectory from an intersection-free start."""
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    assert start.shape == end.shape
    assert start.ndim == 2
    assert start.shape[1] == 3
    assert np.isfinite(start).all()
    assert np.isfinite(end).all()
    start_intersects = bool(ipctk.has_intersections(mesh, start, ipctk.LBVH()))
    assert not start_intersects, "rigid-bone CCD requires an intersection-free start"
    fraction = float(
        ipctk.compute_collision_free_stepsize(
            mesh, start, end, min_distance=0.0, broad_phase=ipctk.LBVH()
        )
    )
    assert math.isfinite(fraction)
    assert 0.0 <= fraction <= 1.0
    end_intersects = bool(ipctk.has_intersections(mesh, end, ipctk.LBVH()))
    return {
        "schema": "joint-rigid-bone-linear-ccd-v1",
        "start_intersects": start_intersects,
        "end_intersects": end_intersects,
        "collision_free_fraction": fraction,
        "numerically_admissible": fraction == 1.0 and not end_intersects,
        "start_positions_sha256": hashlib.sha256(start.tobytes()).hexdigest(),
        "end_positions_sha256": hashlib.sha256(end.tobytes()).hexdigest(),
        "trajectory": "linear interpolation of boundary vertex positions",
        "rotation_arc_checked": False,
        "anatomical_validation": False,
    }


class RigidBoneCollision:
    """Exact pure-FEM bone map with a native cross-bone candidate filter."""

    def __init__(self, volume: Any, pivot_m: np.ndarray) -> None:
        boundary = volume.extract_surface(algorithm=None).triangulate()
        original = np.asarray(
            boundary.point_data["vtkOriginalPointIds"], dtype=np.int64
        )
        faces = original[np.asarray(boundary.faces).reshape(-1, 4)[:, 1:]]
        names = [
            str(value) for value in np.asarray(volume.field_data["GroupName"]).ravel()
        ]
        labels = np.asarray(volume.point_data["GroupId"])
        cranium, mandible = names.index("Cranium"), names.index("Mandible")
        pure_cranium = np.all(labels[faces] == cranium, axis=1)
        pure_mandible = np.all(labels[faces] == mandible, axis=1)
        assert pure_cranium.any()
        assert pure_mandible.any()
        self.indices, inverse = np.unique(
            faces[pure_cranium | pure_mandible], return_inverse=True
        )
        local_faces = np.ascontiguousarray(inverse.reshape(-1, 3), dtype=np.int32)
        self.points = np.asarray(volume.points, dtype=np.float64).copy()
        self.vertices = self.points[self.indices]
        self.pivot = np.asarray(pivot_m, dtype=np.float64).copy()
        assert self.pivot.shape == (3,)
        assert np.isfinite(self.pivot).all()
        self.mandible = labels[self.indices] == mandible
        self.mesh = ipctk.CollisionMesh(
            self.vertices, ipctk.edges(local_faces), local_faces
        )
        self.mesh.can_collide = ipctk.make_vertex_patches_filter(
            self.mandible.astype(np.int32)
        )
        self.mesh.init_adjacencies()
        self.reference_receipt = linear_bone_step(
            self.mesh, self.vertices, self.vertices
        )
        assert self.reference_receipt["numerically_admissible"]
        self.mapping = {
            "schema": "joint-rigid-bone-map-v1",
            "vertices": len(self.indices),
            "cranium_triangles": int(pure_cranium.sum()),
            "mandible_triangles": int(pure_mandible.sum()),
            "global_node_ids_sha256": hashlib.sha256(
                self.indices.tobytes()
            ).hexdigest(),
            "global_faces_sha256": hashlib.sha256(
                faces[pure_cranium | pure_mandible].tobytes()
            ).hexdigest(),
            "surface_selection": "pure FEM cranium versus pure FEM mandible",
            "complete_source_bones_checked": False,
            "anatomical_validation": False,
            "ipc_version": ipctk.__version__,
        }

    def positions(self, pose_rad_m: np.ndarray) -> np.ndarray:
        pose = np.asarray(pose_rad_m, dtype=np.float64)
        assert pose.shape == (6,)
        assert np.isfinite(pose).all()
        points = self.vertices.copy()
        points[self.mandible] = (
            (points[self.mandible] - self.pivot) @ _rotation_matrix(pose[:3]).T
            + self.pivot
            + pose[3:]
        )
        return points

    def from_displacement(
        self, start_displacement_m: np.ndarray, end_pose_rad_m: np.ndarray
    ) -> dict[str, Any]:
        displacement = np.asarray(start_displacement_m, dtype=np.float64)
        assert displacement.shape == self.points.shape
        start = self.vertices + displacement[self.indices]
        assert (
            np.max(np.abs(start[~self.mandible] - self.vertices[~self.mandible]))
            <= 1e-8
        )
        return linear_bone_step(self.mesh, start, self.positions(end_pose_rad_m))

    def from_poses(
        self, start_pose_rad_m: np.ndarray, end_pose_rad_m: np.ndarray
    ) -> dict[str, Any]:
        return linear_bone_step(
            self.mesh, self.positions(start_pose_rad_m), self.positions(end_pose_rad_m)
        )
