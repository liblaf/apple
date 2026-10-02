# ruff: noqa: EM101, EM102, TRY003
"""Boundary CCD with faces local to inverted tetrahedra excluded.

The exclusion is geometric bookkeeping for a trial that permits a small
number of inverted cells. It does not make those cells physically valid.
"""

from __future__ import annotations

from typing import Any

import ipctk
import numpy as np
import pyvista as pv
from mouthopen_geometry import MouthOpenGeometry


class MouthOpenLocalFolds:
    """Run CCD on the compact remainder of the FEM boundary."""

    def __init__(self, volume: pv.UnstructuredGrid, max_inverted: int) -> None:
        self.full = MouthOpenGeometry(volume)
        self.points = np.asarray(volume.points, dtype=np.float64)
        self.tets = np.asarray(volume.cells).reshape(-1, 5)[:, 1:].copy()
        assert np.all(np.asarray(volume.cells).reshape(-1, 5)[:, 0] == 4)
        self.max_inverted = max_inverted
        self.faces = np.asarray(self.full.mesh.faces, dtype=np.int32)
        assert self.faces.ndim == 2
        assert self.faces.shape[1] == 3
        self.face_volume_ids = self.full.surface_point_ids[self.faces]
        self.rest_det = self._determinants(self.points)
        assert np.all(self.rest_det > 0)
        self.last_step: dict[str, Any] | None = None
        self.step_receipts: list[dict[str, Any]] = []

    def _displacement(self, value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        u = np.asarray(value, dtype=np.float64)
        if u.shape == (3 * len(self.points),):
            u = u.reshape(-1, 3)
        if u.shape != self.points.shape:
            raise ValueError(
                f"expected displacement shape {self.points.shape}, got {u.shape}"
            )
        if not np.isfinite(u).all():
            raise ValueError("displacement contains nonfinite values")
        return u

    def _determinants(self, positions: np.ndarray) -> np.ndarray:
        tet = positions[self.tets]
        return np.einsum(
            "ij,ij->i",
            tet[:, 1] - tet[:, 0],
            np.cross(tet[:, 2] - tet[:, 0], tet[:, 3] - tet[:, 0]),
        )

    def inverted(self, u: Any) -> tuple[np.ndarray, float]:
        displacement = self._displacement(u)
        j = self._determinants(self.points + displacement) / self.rest_det
        assert np.isfinite(j).all()
        return j <= 0, float(j.min())

    def _compact(
        self, inverted: np.ndarray
    ) -> tuple[ipctk.CollisionMesh, np.ndarray, int]:
        excluded_vertices = np.zeros(len(self.points), dtype=bool)
        excluded_vertices[self.tets[inverted].ravel()] = True
        excluded_faces = np.any(excluded_vertices[self.face_volume_ids], axis=1)
        kept_faces = self.faces[~excluded_faces]
        if len(kept_faces) == 0:
            raise ValueError("local fold mask removed the entire FEM boundary")
        kept_vertices, inverse = np.unique(kept_faces, return_inverse=True)
        compact_faces = np.asfortranarray(inverse.reshape(-1, 3), dtype=np.int32)
        assert np.array_equal(kept_vertices[compact_faces], kept_faces)
        rest = np.asfortranarray(
            self.points[self.full.surface_point_ids[kept_vertices]], dtype=np.float64
        )
        edges = np.asfortranarray(ipctk.edges(compact_faces), dtype=np.int32)
        collision_mesh = ipctk.CollisionMesh(
            rest_positions=rest, edges=edges, faces=compact_faces
        )
        return (
            collision_mesh,
            self.full.surface_point_ids[kept_vertices],
            int(excluded_faces.sum()),
        )

    def max_step_size(self, u: Any, du: Any) -> float:
        """Bound motion on faces unaffected by endpoint inverted cells."""
        current = self._displacement(u)
        increment = self._displacement(du)
        current_inverted, current_min_j = self.inverted(current)
        trial_inverted, trial_min_j = self.inverted(current + increment)
        union = current_inverted | trial_inverted
        receipt: dict[str, Any] = {
            "current_inverted_cells": int(current_inverted.sum()),
            "trial_inverted_cells": int(trial_inverted.sum()),
            "union_inverted_cells": int(union.sum()),
            "current_minimum_J": current_min_j,
            "trial_minimum_J": trial_min_j,
            "maximum_allowed_inverted_cells": self.max_inverted,
            "trial_exceeds_inversion_diagnostic_limit": bool(
                trial_inverted.sum() > self.max_inverted
            ),
        }
        mesh, point_ids, excluded = self._compact(union)
        receipt["excluded_faces"] = excluded
        receipt["retained_faces"] = int(self.full.face_count - excluded)
        receipt["retained_vertices"] = len(point_ids)
        vertices_t0 = np.asfortranarray(self.points[point_ids] + current[point_ids])
        vertices_t1 = np.asfortranarray(vertices_t0 + increment[point_ids])
        broad_phase = ipctk.LBVH()
        if ipctk.has_intersections(mesh, vertices_t0, broad_phase):
            raise ValueError("retained FEM boundary intersects at CCD start")
        candidates = ipctk.Candidates()
        candidates.build(
            mesh=mesh,
            vertices_t0=vertices_t0,
            vertices_t1=vertices_t1,
            inflation_radius=0.0,
            broad_phase=broad_phase,
        )
        fraction = float(
            candidates.compute_collision_free_stepsize(
                mesh=mesh,
                vertices_t0=vertices_t0,
                vertices_t1=vertices_t1,
                min_distance=0.0,
                narrow_phase_ccd=ipctk.TightInclusionCCD(),
            )
        )
        if not np.isfinite(fraction) or not 0 <= fraction <= 1:
            raise RuntimeError(f"IPCTK returned invalid CCD fraction {fraction}")
        receipt["collision_fraction"] = fraction
        self.last_step = receipt
        self.step_receipts.append(receipt)
        return fraction

    def take_step_receipts(self) -> list[dict[str, Any]]:
        receipts, self.step_receipts = self.step_receipts, []
        return receipts

    def audit(self, u: Any) -> dict[str, Any]:
        """Report full-boundary and masked-boundary intersections separately."""
        displacement = self._displacement(u)
        inverted, min_j = self.inverted(displacement)
        mesh, point_ids, excluded = self._compact(inverted)
        vertices = np.asfortranarray(self.points[point_ids] + displacement[point_ids])
        masked_intersects = bool(ipctk.has_intersections(mesh, vertices, ipctk.LBVH()))
        full = self.full.audit(displacement)
        return {
            "retained_boundary_no_intersections": not masked_intersects,
            "retained_boundary_has_intersections": masked_intersects,
            "full_boundary_no_intersections": full["no_intersections"],
            "full_boundary_has_intersections": full["has_intersections"],
            "inverted_cells_for_mask": int(inverted.sum()),
            "minimum_J_for_mask": min_j,
            "excluded_faces": excluded,
            "retained_faces": int(self.full.face_count - excluded),
            "retained_vertices": len(point_ids),
            "full_boundary": full,
            "intersection_scope": "self-intersection on retained extracted FEM boundary faces",
            "limitations": [
                "faces incident to any vertex of an inverted tetrahedron are excluded",
                "does not test bone obstacles, containment, or excluded face intersections",
                "inverted cells are not physically valid",
            ],
        }


__all__ = ["MouthOpenLocalFolds"]
