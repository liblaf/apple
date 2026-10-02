"""Full-boundary, frictionless IPC self-contact for expression transitions.

The displacement vector is indexed by the volume's ``GlobalPointId`` values.
Every extracted FEM-boundary triangle is retained: this module deliberately
applies no tissue, bone, or attachment-face exclusions.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import ipctk
import numpy as np
import torch

_APPLE_ROOT = Path(__file__).resolve().parents[6]
_JOINT_SOURCE = _APPLE_ROOT / "exp/2026/09/21/joint-activation-material-mandible/src"
if str(_JOINT_SOURCE) not in sys.path:
    sys.path.insert(0, str(_JOINT_SOURCE))
from joint_contact import OwnedContact  # noqa: E402


def _triangles(surface: Any) -> np.ndarray:
    faces = np.asarray(surface.faces, dtype=np.int64)
    assert faces.ndim == 1
    assert len(faces) % 4 == 0
    packed = faces.reshape(-1, 4)
    assert np.all(packed[:, 0] == 3), "extracted boundary must be triangular"
    return np.ascontiguousarray(packed[:, 1:], dtype=np.int64)


def _mean_edge_length(points: np.ndarray, triangles: np.ndarray) -> float:
    edges = np.concatenate(
        (triangles[:, (0, 1)], triangles[:, (1, 2)], triangles[:, (2, 0)]), axis=0
    )
    edges.sort(axis=1)
    edges = np.unique(edges, axis=0)
    lengths = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
    mean = float(lengths.mean())
    assert math.isfinite(mean)
    assert mean > 0
    return mean


def build_self_contact(
    volume: Any,
    stiffness_mpa: float = 0.0012,
    minimum_distance_m: float = 1e-8,
) -> tuple[OwnedContact, dict[str, Any]]:
    """Build an IPC contact object over the complete extracted FEM boundary.

    ``volume.point_data['GlobalPointId']`` is the only displacement mapping. It
    must be a unique, contiguous index map for the supplied full FEM volume.
    Candidate and active collision sets are rebuilt by :class:`OwnedContact` on
    every ``state_at``/``update`` call.
    """
    assert math.isfinite(stiffness_mpa)
    assert stiffness_mpa > 0
    assert math.isfinite(minimum_distance_m)
    assert minimum_distance_m > 0
    assert "GlobalPointId" in volume.point_data
    global_ids = np.asarray(volume.point_data["GlobalPointId"], dtype=np.int64)
    assert global_ids.shape == (volume.n_points,)
    np.testing.assert_array_equal(global_ids, np.arange(volume.n_points))

    surface = volume.extract_surface(algorithm=None).triangulate()
    assert "GlobalPointId" in surface.point_data
    surface_ids = np.asarray(surface.point_data["GlobalPointId"], dtype=np.int64)
    assert surface_ids.ndim == 1
    assert len(np.unique(surface_ids)) == len(surface_ids)
    assert np.all((surface_ids >= 0) & (surface_ids < volume.n_points))
    np.testing.assert_allclose(
        np.asarray(surface.points),
        np.asarray(volume.points)[surface_ids],
        rtol=0,
        atol=0,
    )
    local_triangles = _triangles(surface)
    global_triangles = surface_ids[local_triangles]
    selected_ids, inverse = np.unique(global_triangles, return_inverse=True)
    np.testing.assert_array_equal(selected_ids, np.sort(surface_ids))
    triangles = np.ascontiguousarray(inverse.reshape(-1, 3), dtype=np.int32)
    positions = np.ascontiguousarray(
        np.asarray(volume.points)[selected_ids], dtype=np.float64
    )
    mean_edge_length_m = _mean_edge_length(positions, triangles)
    dhat_m = 0.5 * mean_edge_length_m

    mesh = ipctk.CollisionMesh(positions, ipctk.edges(triangles), triangles)
    mesh.init_adjacencies()
    potential = ipctk.BarrierPotential(
        dhat=dhat_m, stiffness=stiffness_mpa, use_physical_barrier=True
    )
    contact = OwnedContact(
        collision_mesh=mesh,
        indices=torch.as_tensor(selected_ids, dtype=torch.long),
        potential=potential,
        broad_phase=ipctk.LBVH(),
        narrow_phase_ccd=ipctk.TightInclusionCCD(),
        dmin=0.0,
        min_distance=minimum_distance_m,
        use_physical_barrier=True,
        vertices=torch.as_tensor(positions.copy(), dtype=torch.float64),
    )
    receipt = {
        "schema": "full-fem-boundary-self-contact-v1",
        "surface_map": "exact extracted-surface GlobalPointId -> full volume displacement",
        "surface_vertices": len(selected_ids),
        "surface_triangles": len(triangles),
        "surface_edges": len(mesh.edges),
        "surface_exclusions": "none; every extracted FEM-boundary triangle participates",
        "candidate_policy": "fresh broad-phase candidates on every OwnedContact update",
        "friction": "frictionless",
        "stiffness_mpa": float(stiffness_mpa),
        "mean_boundary_edge_m": mean_edge_length_m,
        "dhat_m": dhat_m,
        "barrier_dmin_m": 0.0,
        "ccd_min_distance_m": float(minimum_distance_m),
        "ipc_version": ipctk.__version__,
    }
    return contact, receipt


def audit_contact(contact: OwnedContact, u: torch.Tensor) -> dict[str, Any]:
    """Return IPC and complete-boundary self-intersection diagnostics at ``u``."""
    assert u.ndim == 2
    assert u.shape[1] == 3
    assert int(contact.indices.max()) < len(u)
    state = contact.state_at(u)
    positions = (contact.vertices + u[contact.indices]).detach().cpu().numpy()
    has_intersections = bool(
        ipctk.has_intersections(contact.collision_mesh, positions, ipctk.LBVH())
    )
    diagnostics = contact.diagnostics(state, u)
    diagnostics.update(
        {
            "contact_valid": diagnostics["contact_numerically_valid"]
            and not has_intersections,
            "complete_boundary_has_intersections": has_intersections,
            "complete_boundary_no_intersections": not has_intersections,
            "intersection_scope": "all extracted FEM-boundary triangles",
            "complete_boundary_vertices": len(contact.indices),
            "complete_boundary_triangles": len(contact.collision_mesh.faces),
        }
    )
    return diagnostics
