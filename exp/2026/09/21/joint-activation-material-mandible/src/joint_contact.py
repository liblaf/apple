"""Owned IPC contact states for free tissue against cranium and mandible.

Use one exact FEM boundary map. Pure bone and pure soft faces participate;
mixed bone/soft attachment triangles remain bonded FEM transition elements.
The physical barrier has MPa stiffness, so area weighting yields MPa m^3,
matching the tissue energy. Broad-phase candidates are rebuilt for every state.
"""

from __future__ import annotations

import math
from typing import Any

import attrs
import ipctk
import numpy as np
import torch

from liblaf.apple.collision import Collision


@attrs.define
class OwnedContactState(Collision.State):
    minimum_ccd_fraction: float = 1.0
    boundary_ccd_fraction: float = 1.0


@attrs.define
class OwnedContact(Collision):
    """Rebuildable collision state; no primal state is shared with an adjoint."""

    collision_set_type: Any = (
        ipctk.NormalCollisions.CollisionSetType.IMPROVED_MAX_APPROX
    )

    def state_at(self, u: torch.Tensor) -> OwnedContactState:
        state = OwnedContactState()
        state.collisions.use_area_weighting = True
        state.collisions.collision_set_type = self.collision_set_type
        self.update(state, u)
        return state

    def init(self) -> OwnedContactState:
        shape = (int(self.indices.max()) + 1, 3)
        return self.state_at(self.vertices.new_zeros(shape))

    def update(self, state: OwnedContactState, u: torch.Tensor) -> None:
        positions = (self.vertices + u[self.indices]).numpy(force=True)
        state.candidates.clear()
        state.candidates.build(
            mesh=self.collision_mesh,
            vertices=positions,
            inflation_radius=self.inflation_radius,
            broad_phase=self.broad_phase,
        )
        state.collisions.clear()
        state.collisions.build(
            candidates=state.candidates,
            mesh=self.collision_mesh,
            vertices=positions,
            dhat=self.potential.dhat,
            dmin=self.dmin,
        )
        state.hess = None

    def max_step_size(
        self, state: OwnedContactState, u: torch.Tensor, p: torch.Tensor
    ) -> torch.Tensor:
        fraction = super().max_step_size(state, u, p)
        state.minimum_ccd_fraction = min(state.minimum_ccd_fraction, float(fraction))
        return fraction

    def diagnostics(self, state: OwnedContactState, u: torch.Tensor) -> dict[str, Any]:
        energy = float(self.fun(state, u))
        count = len(state.collisions)
        positions = (self.vertices + u[self.indices]).numpy(force=True)
        squared_distance = (
            float(
                state.collisions.compute_minimum_distance(
                    self.collision_mesh, positions
                )
            )
            if count
            else None
        )
        # IPC's distance functions and collision minimum return squared distance.
        distance = math.sqrt(squared_distance) if squared_distance is not None else None
        valid = (
            math.isfinite(energy)
            and energy >= 0
            and (
                distance is None
                or (math.isfinite(distance) and distance > self.min_distance)
            )
            and 0 < state.minimum_ccd_fraction <= 1
            and state.boundary_ccd_fraction == 1.0
        )
        return {
            "enabled": True,
            "contact_numerically_valid": valid,
            "barrier_energy": energy,
            "active_contact_count": count,
            "minimum_active_distance_m": distance,
            "minimum_distance_scope": "active IPC pairs only; null when none are active",
            "ccd_boundary_fraction": state.boundary_ccd_fraction,
            "ccd_minimum_inner_fraction": state.minimum_ccd_fraction,
            "dhat_m": float(self.potential.dhat),
            "barrier_dmin_m": self.dmin,
            "ccd_min_distance_m": self.min_distance,
            "friction": "frictionless",
            "collision_set_type": self.collision_set_type.name,
        }


def build_owned_contact(
    volume: Any,
    fixed_node_ids: np.ndarray,
    config: dict[str, Any],
) -> tuple[OwnedContact, dict[str, Any]]:
    assert config["schema"] == "joint-bone-contact-v1"
    assert config["enabled"] is True
    assert config["surface_selection"] == "pure-soft-vs-pure-bone"
    assert config["friction"] == "frictionless"
    assert config["attachment_policy"] == "bonded-mixed-faces"
    assert config["stiffness_mpa"] > 0
    assert config["dhat_m"] > 0
    boundary = volume.extract_surface(algorithm=None).triangulate()
    original = np.asarray(boundary.point_data["vtkOriginalPointIds"], dtype=np.int64)
    faces = original[np.asarray(boundary.faces).reshape(-1, 4)[:, 1:]]
    names = [str(value) for value in np.asarray(volume.field_data["GroupName"]).ravel()]
    labels = np.asarray(volume.point_data["GroupId"])
    cranium, mandible = names.index("Cranium"), names.index("Mandible")
    bone = np.isin(labels, (cranium, mandible))
    pure_cranium = np.all(labels[faces] == cranium, axis=1)
    pure_mandible = np.all(labels[faces] == mandible, axis=1)
    pure_soft = np.all(~bone[faces], axis=1)
    selected = pure_cranium | pure_mandible | pure_soft
    indices, inverse = np.unique(faces[selected], return_inverse=True)
    local_faces = np.ascontiguousarray(inverse.reshape(-1, 3), dtype=np.int32)
    positions = np.asarray(volume.points[indices], dtype=np.float64)
    mesh = ipctk.CollisionMesh(positions, ipctk.edges(local_faces), local_faces)
    mesh.can_collide = ipctk.make_vertex_patches_filter(
        np.asarray(bone[indices], dtype=np.int32)
    )
    mesh.init_adjacencies()
    potential = ipctk.BarrierPotential(
        dhat=config["dhat_m"],
        stiffness=config["stiffness_mpa"],
        use_physical_barrier=True,
    )
    contact = OwnedContact(
        collision_mesh=mesh,
        indices=torch.as_tensor(indices, dtype=torch.long),
        potential=potential,
        broad_phase=ipctk.LBVH(),
        use_physical_barrier=True,
        vertices=torch.as_tensor(positions.copy()),
    )
    receipt = {
        "schema": "joint-contact-surface-map-v1",
        "vertices": len(indices),
        "cranium_triangles": int(pure_cranium.sum()),
        "mandible_triangles": int(pure_mandible.sum()),
        "soft_triangles": int(pure_soft.sum()),
        "bonded_mixed_triangles_omitted": int((~selected).sum()),
        "fixed_node_count": len(fixed_node_ids),
        "bone_bone_contact": False,
        "soft_soft_contact": False,
        "anatomical_validation": False,
        "config": config,
        "ipc_version": ipctk.__version__,
    }
    return contact, receipt
