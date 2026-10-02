# Copyright (c) 2026 liblaf
# ruff: noqa: PLR0915
"""Rigid source-anatomy IPC for the pruned MouthOpen FEM model.

The physical FEM model is left intact.  Complete registered cranium, mandible,
and eye meshes are appended as fixed, zero-cell nodes solely for contact.  The
collision surface contains pure-soft FEM faces and every source rigid triangle;
mixed FEM attachment faces are explicitly excluded.
"""

from __future__ import annotations

import hashlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import attrs
import ipctk
import numpy as np
import pyvista as pv
import torch

from liblaf.apple.forward import Model
from liblaf.apple.forward.dof_map import DofMap

GROUP = Path(__file__).resolve().parents[1]
ROOT = GROUP.parents[4]
JOINT_SOURCE = ROOT / "exp/2026/09/21/joint-activation-material-mandible/src"
if str(JOINT_SOURCE) not in sys.path:
    sys.path.insert(0, str(JOINT_SOURCE))
from joint_contact import OwnedContact  # noqa: E402

CRANIUM_PATH = ROOT.parent / "melon/exp/2026/05/27/head/data/13-cranium.ply"
MANDIBLE_PATH = ROOT.parent / "melon/exp/2026/05/27/head/data/13-mandible.ply"
EYES_PATH = (
    ROOT
    / "exp/2026/09/21/joint-activation-material-mandible/data/rigid-eyes-001/eyes.vtp"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _triangles(mesh: pv.PolyData, name: str) -> np.ndarray:
    packed = np.asarray(mesh.faces, dtype=np.int64)
    assert packed.ndim == 1, name
    assert len(packed) % 4 == 0, name
    packed = packed.reshape(-1, 4)
    assert np.all(packed[:, 0] == 3), name
    faces = np.ascontiguousarray(packed[:, 1:], dtype=np.int32)
    assert len(faces) > 0
    assert faces.min() >= 0
    assert faces.max() < mesh.n_points
    assert np.all(faces[:, 0] != faces[:, 1])
    assert np.all(faces[:, 1] != faces[:, 2])
    assert np.all(faces[:, 2] != faces[:, 0])
    return faces


def _rigid_displacement(
    points: torch.Tensor, pivot: torch.Tensor, pose: torch.Tensor
) -> torch.Tensor:
    assert points.ndim == 2
    assert points.shape[1] == 3
    assert pivot.shape == (3,)
    assert pose.shape == (6,)
    # scipy is intentionally avoided here so this stays on the model device.
    theta = torch.linalg.vector_norm(pose[:3])
    axis = pose[:3] / torch.where(theta == 0, torch.ones_like(theta), theta)
    skew = torch.stack(
        (
            torch.stack((axis[0] * 0, -axis[2], axis[1])),
            torch.stack((axis[2], axis[1] * 0, -axis[0])),
            torch.stack((-axis[1], axis[0], axis[2] * 0)),
        )
    )
    eye = torch.eye(3, dtype=points.dtype, device=points.device)
    rotation = eye + torch.sin(theta) * skew + (1 - torch.cos(theta)) * (skew @ skew)
    # The saved MouthOpen pose is nonzero.  The explicit zero branch keeps the
    # neutral boundary exact rather than allowing a 0/0 Rodrigues evaluation.
    rotation = torch.where((theta == 0), eye, rotation)
    return (points - pivot) @ rotation.T + pivot + pose[3:] - points


@dataclass(frozen=True)
class FixedReferenceContact:
    """Immutable source map plus its rebuildable :class:`OwnedContact`."""

    physical_point_count: int
    soft_physical_ids: np.ndarray
    physical_mandible_ids: np.ndarray
    cranium_ids: np.ndarray
    mandible_ids: np.ndarray
    eye_ids: np.ndarray
    source_points: np.ndarray
    contact: OwnedContact
    receipt: dict[str, Any]

    @property
    def full_point_count(self) -> int:
        return self.physical_point_count + len(self.source_points)

    @property
    def appended_ids(self) -> np.ndarray:
        return np.arange(
            self.physical_point_count, self.full_point_count, dtype=np.int64
        )

    @property
    def full_mandible_mask(self) -> np.ndarray:
        mask = np.zeros(self.full_point_count, dtype=bool)
        mask[self.physical_mandible_ids] = True
        mask[self.mandible_ids] = True
        return mask

    def full_reference_points(self, physical_points: np.ndarray) -> np.ndarray:
        assert physical_points.shape == (self.physical_point_count, 3)
        return np.concatenate((physical_points, self.source_points))

    def extend_seed(
        self, physical_u: torch.Tensor, pose: torch.Tensor, pivot: torch.Tensor
    ) -> torch.Tensor:
        assert physical_u.shape == (self.physical_point_count, 3)
        assert pose.shape == (6,)
        assert pivot.shape == (3,)
        source = torch.as_tensor(
            self.source_points, dtype=physical_u.dtype, device=physical_u.device
        )
        appended = torch.zeros_like(source)
        mandible = torch.as_tensor(
            self.mandible_ids, dtype=torch.long, device=physical_u.device
        )
        appended[mandible - self.physical_point_count] = _rigid_displacement(
            source[mandible - self.physical_point_count], pivot, pose
        )
        return torch.cat((physical_u, appended))

    def boundary_values(
        self,
        physical_points: torch.Tensor,
        pose: torch.Tensor,
        pivot: torch.Tensor,
    ) -> torch.Tensor:
        """Return full-node displacement values for original plus appended nodes."""
        assert physical_points.shape == (self.physical_point_count, 3)
        assert pose.shape == (6,)
        assert pivot.shape == (3,)
        values = physical_points.new_zeros((self.full_point_count, 3))
        physical_mandible = torch.as_tensor(
            self.physical_mandible_ids, dtype=torch.long, device=physical_points.device
        )
        values[physical_mandible] = _rigid_displacement(
            physical_points[physical_mandible], pivot, pose
        )
        source = torch.as_tensor(
            self.source_points,
            dtype=physical_points.dtype,
            device=physical_points.device,
        )
        mandible = torch.as_tensor(
            self.mandible_ids, dtype=torch.long, device=physical_points.device
        )
        values[mandible] = _rigid_displacement(
            source[mandible - self.physical_point_count], pivot, pose
        )
        return values


def build_fixed_reference_contact(
    volume: pv.UnstructuredGrid,
    *,
    stiffness_mpa: float = 1.3544,
    dhat_m: float = 1e-4,
    minimum_distance_m: float = 1e-8,
    cranium_path: Path = CRANIUM_PATH,
    mandible_path: Path = MANDIBLE_PATH,
    eyes_path: Path = EYES_PATH,
) -> FixedReferenceContact:
    """Build pure-soft-versus-complete-rigid standard IPC contact.

    ``volume`` must be the pruned no-skin fixture.  Its ``GlobalPointId`` map
    must remain the contiguous physical FEM displacement index.  No activation
    field, material, original node, or original DOF index is modified.
    """
    assert math.isfinite(stiffness_mpa)
    assert stiffness_mpa > 0
    assert math.isfinite(dhat_m)
    assert dhat_m > 0
    assert math.isfinite(minimum_distance_m)
    assert minimum_distance_m > 0
    assert "GlobalPointId" in volume.point_data
    physical_ids = np.asarray(volume.point_data["GlobalPointId"], dtype=np.int64)
    np.testing.assert_array_equal(physical_ids, np.arange(volume.n_points))
    assert "GroupId" in volume.point_data
    assert "GroupName" in volume.field_data
    names = [str(value) for value in np.asarray(volume.field_data["GroupName"]).ravel()]
    cranium_group, mandible_group = names.index("Cranium"), names.index("Mandible")
    boundary = volume.extract_surface(algorithm=None, pass_pointid=True).triangulate()
    original = np.asarray(boundary.point_data["vtkOriginalPointIds"], dtype=np.int64)
    faces = _triangles(boundary, "FEM boundary")
    physical_faces = original[faces]
    group = np.asarray(volume.point_data["GroupId"], dtype=np.int64)
    bone = np.isin(group[physical_faces], (cranium_group, mandible_group))
    pure_soft = ~np.any(bone, axis=1)
    assert np.any(pure_soft)
    soft_ids, soft_inverse = np.unique(physical_faces[pure_soft], return_inverse=True)
    soft_faces = np.ascontiguousarray(soft_inverse.reshape(-1, 3), dtype=np.int32)
    assert not np.any(np.isin(group[soft_ids], (cranium_group, mandible_group)))

    source_meshes = {
        "cranium": pv.read(cranium_path),
        "mandible": pv.read(mandible_path),
        "eyes": pv.read(eyes_path),
    }
    source_points = []
    source_faces = []
    source_ids: dict[str, np.ndarray] = {}
    cursor = volume.n_points
    local_cursor = len(soft_ids)
    for name in ("cranium", "mandible", "eyes"):
        mesh = source_meshes[name]
        points = np.ascontiguousarray(np.asarray(mesh.points, dtype=np.float64))
        assert points.shape == (mesh.n_points, 3)
        assert np.isfinite(points).all()
        triangles = _triangles(mesh, name)
        source_points.append(points)
        source_faces.append(triangles + local_cursor)
        source_ids[name] = np.arange(cursor, cursor + len(points), dtype=np.int64)
        cursor += len(points)
        local_cursor += len(points)
    appended_points = np.concatenate(source_points)
    contact_points = np.concatenate(
        (np.asarray(volume.points)[soft_ids], appended_points)
    ).astype(np.float64, copy=False)
    contact_faces = np.concatenate((soft_faces, *source_faces)).astype(
        np.int32, copy=False
    )
    indices = np.concatenate(
        (soft_ids, source_ids["cranium"], source_ids["mandible"], source_ids["eyes"])
    )
    assert len(np.unique(indices)) == len(indices)
    assert indices.max() == cursor - 1
    mesh = ipctk.CollisionMesh(
        contact_points, ipctk.edges(contact_faces), contact_faces
    )
    patches = np.concatenate(
        (
            np.zeros(len(soft_ids), dtype=np.int32),
            np.ones(len(appended_points), dtype=np.int32),
        )
    )
    mesh.can_collide = ipctk.make_vertex_patches_filter(patches)
    mesh.init_adjacencies()
    ccd = ipctk.TightInclusionCCD(max_iterations=100_000)
    ccd.tolerance = 1e-10
    contact = OwnedContact(
        collision_mesh=mesh,
        indices=torch.as_tensor(indices, dtype=torch.long),
        vertices=torch.as_tensor(contact_points.copy(), dtype=torch.float64),
        potential=ipctk.BarrierPotential(
            dhat=dhat_m, stiffness=stiffness_mpa, use_physical_barrier=True
        ),
        broad_phase=ipctk.LBVH(),
        narrow_phase_ccd=ccd,
        min_distance=minimum_distance_m,
        use_physical_barrier=True,
        collision_set_type=ipctk.NormalCollisions.CollisionSetType.IPC,
    )
    fixed = np.asarray(volume.point_data["IsFixed"], dtype=bool)
    physical_mandible_ids = np.flatnonzero(fixed & (group == mandible_group)).astype(
        np.int64
    )
    assert len(physical_mandible_ids) > 0
    mixed = ~pure_soft & ~np.all(bone, axis=1)
    receipt = {
        "schema": "fixed-reference-soft-rigid-contact-v1",
        "policy": "pure soft FEM boundary versus complete registered cranium, mandible, and eyes; soft-soft and rigid-rigid pairs disabled",
        "mapping": "contact.indices maps pure-soft physical GlobalPointId nodes plus appended rigid nodes into the extended displacement vector",
        "physical_fem_nodes": int(volume.n_points),
        "appended_rigid_nodes": len(appended_points),
        "full_nodes": int(cursor),
        "soft_vertices": len(soft_ids),
        "soft_triangles": len(soft_faces),
        "mixed_attachment_triangles_excluded": int(mixed.sum()),
        "pure_bone_fem_triangles_excluded": int(np.all(bone, axis=1).sum()),
        "source": {
            name: {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "vertices": int(source_meshes[name].n_points),
                "triangles": len(_triangles(source_meshes[name], name)),
            }
            for name, path in (
                ("cranium", cranium_path),
                ("mandible", mandible_path),
                ("eyes", eyes_path),
            )
        },
        "original_isfixed_count": int(fixed.sum()),
        "original_mandible_isfixed_count": len(physical_mandible_ids),
        "pure_soft_fixed_vertices": int(fixed[soft_ids].sum()),
        "original_isfixed_policy": "preserve source IsFixed DOFs; only IsFixed∩Mandible receives the rigid jaw pose",
        "appended_policy": "all complete-source rigid DOFs fixed; cranium and eyes zero, mandible rigid pose",
        "soft_soft_contact": False,
        "rigid_rigid_contact": False,
        "source_triangles_excluded": 0,
        "friction": "frictionless",
        "collision_set_type": "IPC",
        "candidate_policy": "fresh broad-phase candidates on every OwnedContact update",
        "barrier": {
            "stiffness_mpa": float(stiffness_mpa),
            "dhat_m": float(dhat_m),
            "dmin_m": 0.0,
            "ccd_min_distance_m": float(minimum_distance_m),
            "ccd_max_iterations": 100_000,
            "ccd_tolerance_m": 1e-10,
        },
        "ipc_version": ipctk.__version__,
    }
    return FixedReferenceContact(
        volume.n_points,
        soft_ids,
        physical_mandible_ids,
        source_ids["cranium"],
        source_ids["mandible"],
        source_ids["eyes"],
        appended_points,
        contact,
        receipt,
    )


def audit_fixed_reference_contact(
    reference: FixedReferenceContact, full_u: torch.Tensor
) -> dict[str, Any]:
    """Audit the exact soft-rigid IPC state without changing solver history."""
    assert full_u.shape == (reference.full_point_count, 3)
    state = reference.contact.state_at(full_u)
    positions = (reference.contact.vertices + full_u[reference.contact.indices]).numpy(
        force=True
    )
    intersections = bool(
        ipctk.has_intersections(
            reference.contact.collision_mesh, positions, ipctk.LBVH()
        )
    )
    diagnostics = reference.contact.diagnostics(state, full_u)
    diagnostics.update(
        {
            "scoped_no_intersections": not intersections,
            "scoped_has_intersections": intersections,
            "scope": "pure-soft FEM triangles versus complete source cranium, mandible, and eyes",
            "clearance_at_least_dhat": len(state.collisions) == 0,
            "clearance_statement": "empty IPC collision set proves all admissible soft-rigid stencils are outside dhat"
            if len(state.collisions) == 0
            else "active IPC stencils exist; inspect minimum_active_distance_m",
        }
    )
    return diagnostics


def extend_dof_map(
    original: DofMap, reference: FixedReferenceContact, fixed_values: torch.Tensor
) -> DofMap:
    """Append all rigid coordinates as prescribed, keeping physical free IDs exact."""
    assert original.n_points == reference.physical_point_count
    assert fixed_values.shape == (reference.full_point_count, 3)
    device = original.fixed_indices.device
    appended = torch.arange(
        original.n_full,
        reference.full_point_count * original.dim,
        device=device,
        dtype=original.fixed_indices.dtype,
    )
    selected = torch.cat((original.fixed_indices, appended)).to(fixed_values.device)
    return DofMap(
        n_points=reference.full_point_count,
        dim=original.dim,
        fixed_indices=torch.cat((original.fixed_indices, appended)),
        fixed_values=fixed_values.flatten()[selected],
        free_indices=original.free_indices.clone(),
    )


def attach_fixed_reference_contact(
    model: Model, reference: FixedReferenceContact, fixed_values: torch.Tensor
) -> Model:
    """Return a model with unchanged physical warp data and appended rigid contact DOFs."""
    assert model.n_points == reference.physical_point_count
    assert model.collision is None
    dof_map = extend_dof_map(model.dof_map, reference, fixed_values)
    assert torch.equal(dof_map.free_indices, model.dof_map.free_indices)
    return attrs.evolve(model, dof_map=dof_map, collision=reference.contact)


__all__ = [
    "FixedReferenceContact",
    "attach_fixed_reference_contact",
    "audit_fixed_reference_contact",
    "build_fixed_reference_contact",
    "extend_dof_map",
]
