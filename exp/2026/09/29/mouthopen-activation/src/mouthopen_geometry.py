# ruff: noqa: EM101, EM102, TRY003
"""CPU geometry checks for a prescribed MouthOpen FEM boundary path."""

from __future__ import annotations

from typing import Any

import ipctk
import numpy as np
import pyvista as pv


class MouthOpenGeometry:
    """Check self-intersection and CCD on the extracted tetrahedral boundary.

    This is a geometric feasibility check only. It adds no contact energy, and
    it does not test collision against separate bone surfaces or containment.
    """

    def __init__(self, mesh: pv.UnstructuredGrid) -> None:
        if not isinstance(mesh, pv.UnstructuredGrid):
            raise TypeError("expected the tetrahedral volume mesh")
        if not np.all(mesh.celltypes == pv.CellType.TETRA):
            raise ValueError("volume mesh must contain only tetrahedra")

        surface = mesh.extract_surface(algorithm=None, pass_pointid=True)
        point_ids = np.asarray(
            surface.point_data["vtkOriginalPointIds"], dtype=np.int64
        )
        if point_ids.shape != (surface.n_points,):
            raise ValueError("boundary point map has the wrong shape")
        if np.any(point_ids < 0) or np.any(point_ids >= mesh.n_points):
            raise ValueError("boundary point map is outside the volume mesh")
        np.testing.assert_array_equal(
            np.asarray(surface.points), np.asarray(mesh.points)[point_ids]
        )

        faces = np.asarray(surface.faces).reshape(-1, 4)
        if not np.all(faces[:, 0] == 3):
            raise ValueError("extracted boundary contains non-triangle cells")
        faces = np.asfortranarray(faces[:, 1:], dtype=np.int32)
        if faces.size == 0:
            raise ValueError("extracted boundary is empty")
        edges = np.asfortranarray(ipctk.edges(faces), dtype=np.int32)
        rest = np.asfortranarray(np.asarray(mesh.points)[point_ids], dtype=np.float64)
        self.mesh = ipctk.CollisionMesh(rest_positions=rest, edges=edges, faces=faces)
        self.surface_point_ids = point_ids
        self.vertex_count = int(mesh.n_points)
        self.face_count = len(faces)
        self.edge_count = len(edges)

        undirected = np.sort(
            np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])),
            axis=1,
        )
        _, incidence = np.unique(undirected, axis=0, return_counts=True)
        self.boundary_edge_count = int(np.count_nonzero(incidence == 1))
        self.nonmanifold_edge_count = int(np.count_nonzero(incidence > 2))
        self.broad_phase = ipctk.LBVH()
        self.narrow_phase = ipctk.TightInclusionCCD()

    @staticmethod
    def _array(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value, dtype=np.float64)

    def _surface_positions(self, u: Any) -> np.ndarray:
        displacement = self._array(u)
        if displacement.shape == (3 * self.vertex_count,):
            displacement = displacement.reshape(self.vertex_count, 3)
        if displacement.shape != (self.vertex_count, 3):
            raise ValueError(
                f"expected displacement shape {(self.vertex_count, 3)} or "
                f"{(3 * self.vertex_count,)}, got {displacement.shape}"
            )
        if not np.isfinite(displacement).all():
            raise ValueError("displacement contains nonfinite values")
        return np.asfortranarray(
            self.mesh.rest_positions + displacement[self.surface_point_ids],
            dtype=np.float64,
        )

    def audit(self, u: Any) -> dict[str, Any]:
        """Return a pointwise self-intersection and surface-topology receipt."""
        vertices = self._surface_positions(u)
        intersects = bool(
            ipctk.has_intersections(self.mesh, vertices, self.broad_phase)
        )
        return {
            "no_intersections": not intersects,
            "has_intersections": intersects,
            "surface_vertices": len(vertices),
            "surface_faces": self.face_count,
            "surface_edges": self.edge_count,
            "boundary_edges": self.boundary_edge_count,
            "nonmanifold_edges": self.nonmanifold_edge_count,
            "intersection_scope": "self-intersection of complete extracted FEM boundary",
            "limitations": [
                "does not test against separate bone obstacle surfaces",
                "does not test containment or inside/outside orientation",
                "nonmanifold surface edges are reported but not repaired",
            ],
        }

    def max_step_size(self, u: Any, du: Any) -> float:
        """Bound linear motion by IPC continuous collision detection."""
        vertices_t0 = self._surface_positions(u)
        if ipctk.has_intersections(self.mesh, vertices_t0, self.broad_phase):
            raise ValueError(
                "CCD requires an intersection-free starting boundary; "
                "call audit(u) and reject the invalid start"
            )
        increment = self._array(du)
        if increment.shape == (3 * self.vertex_count,):
            increment = increment.reshape(self.vertex_count, 3)
        if increment.shape != (self.vertex_count, 3):
            raise ValueError(
                f"expected increment shape {(self.vertex_count, 3)} or "
                f"{(3 * self.vertex_count,)}, got {increment.shape}"
            )
        if not np.isfinite(increment).all():
            raise ValueError("increment contains nonfinite values")
        vertices_t1 = np.asfortranarray(
            vertices_t0 + increment[self.surface_point_ids], dtype=np.float64
        )
        candidates = ipctk.Candidates()
        candidates.build(
            mesh=self.mesh,
            vertices_t0=vertices_t0,
            vertices_t1=vertices_t1,
            inflation_radius=0.0,
            broad_phase=self.broad_phase,
        )
        fraction = float(
            candidates.compute_collision_free_stepsize(
                mesh=self.mesh,
                vertices_t0=vertices_t0,
                vertices_t1=vertices_t1,
                min_distance=0.0,
                narrow_phase_ccd=self.narrow_phase,
            )
        )
        if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise RuntimeError(f"IPCTK returned invalid CCD fraction {fraction}")
        return fraction


__all__ = ["MouthOpenGeometry"]
