"""Append exact source eyeballs as fixed obstacles in the full-source IPC model.

The constitutive FEM reference, frozen material state, source cranium, and source
mandible remain unchanged. Eye nodes have no constitutive energy and are fixed at
their registered neutral pose. One collision mesh permits only soft-versus-rigid
pairs; soft-soft and every rigid-rigid pairing remain disabled.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ipctk
import numpy as np
import torch
from joint_common import sha256
from joint_contact import OwnedContact
from joint_data import array_sha256
from joint_equilibrium import Equilibrium
from joint_frozen_neutral import FrozenNeutral
from joint_full_skull_contact import FullSkullContactAdapter, FullSkullJointPhysics

from liblaf.apple.forward import Forward, Model
from liblaf.apple.forward.dof_map import DofMap

EYES_SCHEMA = "joint-rigid-eyes-v1"


@dataclass(frozen=True)
class RigidEyeGeometry:
    """Exact registered source-eye triangles and their hash-bound provenance."""

    points_m: np.ndarray
    triangles: np.ndarray
    source_vertex_ids: np.ndarray
    source_triangle_ids: np.ndarray
    vertex_component_ids: np.ndarray
    triangle_component_ids: np.ndarray
    directory: Path
    manifest: dict[str, Any]
    manifest_sha256: str

    @property
    def node_count(self) -> int:
        return len(self.points_m)

    @property
    def triangle_count(self) -> int:
        return len(self.triangles)

    @property
    def component_count(self) -> int:
        return int(self.vertex_component_ids.max()) + 1

    def binding_receipt(self) -> dict[str, Any]:
        npz = self.manifest["artifacts"]["eyes.npz"]
        source = self.manifest["source"]
        return {
            "schema": "joint-rigid-eye-geometry-binding-v1",
            "directory": str(self.directory),
            "manifest_sha256": self.manifest_sha256,
            "eyes_npz_path": npz["path"],
            "eyes_npz_sha256": npz["sha256"],
            "source_path": source["path"],
            "source_sha256": source["sha256"],
            "units": self.manifest["units"],
            "frame": self.manifest["frame"],
            "vertices": self.node_count,
            "triangles": self.triangle_count,
            "components": self.component_count,
            "all_source_triangles_retained": True,
            "source_coordinates_changed": False,
        }


def _load_array(
    arrays: Any,
    manifest: dict[str, Any],
    key: str,
    *,
    dtype: np.dtype[Any],
    shape_tail: tuple[int, ...],
) -> np.ndarray:
    value = np.asarray(arrays[key])
    expected = manifest["arrays"][key]
    if value.dtype != dtype or value.shape[1:] != shape_tail:
        msg = f"{key} has an unexpected dtype or shape"
        raise ValueError(msg)
    if expected["shape"] != list(value.shape) or expected["dtype"] != value.dtype.str:
        msg = f"{key} layout differs from the eye manifest"
        raise ValueError(msg)
    if array_sha256(value) != expected["sha256"]:
        msg = f"{key} hash differs from the eye manifest"
        raise ValueError(msg)
    return np.ascontiguousarray(value)


def load_rigid_eyes(directory: Path) -> RigidEyeGeometry:  # noqa: C901, PLR0912, PLR0915
    """Load exact source eyes; this validates provenance, not neutral contact."""
    directory = directory.resolve()
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["schema"] != EYES_SCHEMA or manifest["success"] is not True:
        msg = "rigid-eye preparation did not report success"
        raise ValueError(msg)
    if manifest["units"] != "metres":
        msg = "rigid-eye coordinates must remain in metres"
        raise ValueError(msg)
    if manifest["frame"] != "unchanged registered source and FEM world frame":
        msg = "rigid eyes must remain in the registered FEM world frame"
        raise ValueError(msg)
    if (
        manifest["all_source_triangles_retained"] is not True
        or manifest["source_coordinates_changed"] is not False
    ):
        msg = "rigid-eye source geometry was filtered or changed"
        raise ValueError(msg)
    for record in [*manifest["artifacts"].values(), manifest["source"]]:
        path = Path(record["path"])
        if sha256(path) != record["sha256"]:
            msg = f"rigid-eye bound file hash changed: {path}"
            raise ValueError(msg)
    npz_path = directory / "eyes.npz"
    record = manifest["artifacts"]["eyes.npz"]
    if (
        Path(record["path"]).resolve() != npz_path
        or sha256(npz_path) != record["sha256"]
    ):
        msg = "rigid-eye NPZ record does not bind this directory"
        raise ValueError(msg)
    required = {
        "points_m",
        "triangles",
        "source_vertex_ids",
        "source_triangle_ids",
        "vertex_component_ids",
        "triangle_component_ids",
    }
    with np.load(npz_path, allow_pickle=False) as arrays:
        if set(arrays.files) != required or set(manifest["arrays"]) != required:
            msg = "rigid-eye NPZ must contain exactly the declared arrays"
            raise ValueError(msg)
        points = _load_array(
            arrays, manifest, "points_m", dtype=np.dtype("<f8"), shape_tail=(3,)
        )
        triangles = _load_array(
            arrays, manifest, "triangles", dtype=np.dtype("<i8"), shape_tail=(3,)
        )
        vertex_ids = _load_array(
            arrays,
            manifest,
            "source_vertex_ids",
            dtype=np.dtype("<i8"),
            shape_tail=(),
        )
        triangle_ids = _load_array(
            arrays,
            manifest,
            "source_triangle_ids",
            dtype=np.dtype("<i8"),
            shape_tail=(),
        )
        vertex_components = _load_array(
            arrays,
            manifest,
            "vertex_component_ids",
            dtype=np.dtype("<i8"),
            shape_tail=(),
        )
        triangle_components = _load_array(
            arrays,
            manifest,
            "triangle_component_ids",
            dtype=np.dtype("<i8"),
            shape_tail=(),
        )
    if not np.isfinite(points).all() or len(points) == 0 or len(triangles) == 0:
        msg = "rigid-eye geometry must be nonempty and finite"
        raise ValueError(msg)
    if triangles.min() < 0 or triangles.max() >= len(points):
        msg = "rigid-eye triangle references an invalid vertex"
        raise ValueError(msg)
    if np.any(
        (triangles[:, 0] == triangles[:, 1])
        | (triangles[:, 1] == triangles[:, 2])
        | (triangles[:, 2] == triangles[:, 0])
    ):
        msg = "rigid-eye geometry contains a degenerate index triangle"
        raise ValueError(msg)
    if not np.array_equal(vertex_ids, np.arange(len(points))):
        msg = "rigid-eye source vertex ids must retain the exact source order"
        raise ValueError(msg)
    if not np.array_equal(triangle_ids, np.arange(len(triangles))):
        msg = "rigid-eye source triangle ids must retain the exact source order"
        raise ValueError(msg)
    components = int(manifest["components"])
    expected_components = np.arange(components)
    if components <= 0 or not np.array_equal(
        np.unique(vertex_components), expected_components
    ):
        msg = "rigid-eye vertex component ids must be contiguous"
        raise ValueError(msg)
    corner_components = vertex_components[triangles]
    if not np.all(corner_components == corner_components[:, :1]):
        msg = "a rigid-eye triangle crosses source components"
        raise ValueError(msg)
    if not np.array_equal(triangle_components, corner_components[:, 0]):
        msg = "rigid-eye triangle component ids differ from their vertices"
        raise ValueError(msg)
    if (
        manifest["vertices"] != len(points)
        or manifest["triangles"] != len(triangles)
        or components != len(np.unique(triangle_components))
    ):
        msg = "rigid-eye counts differ from the manifest"
        raise ValueError(msg)
    return RigidEyeGeometry(
        points_m=points,
        triangles=triangles,
        source_vertex_ids=vertex_ids,
        source_triangle_ids=triangle_ids,
        vertex_component_ids=vertex_components,
        triangle_component_ids=triangle_components,
        directory=directory,
        manifest=manifest,
        manifest_sha256=sha256(manifest_path),
    )


@dataclass(frozen=True)
class EyeInclusiveContactAdapter:
    """Full-skull adapter extended by fixed source-eye nodes."""

    base: FullSkullContactAdapter
    eyes: RigidEyeGeometry
    collision: OwnedContact
    contact_definition: dict[str, Any]

    @property
    def geometry(self) -> Any:
        """Preserve the runner's original full-skull geometry interface."""
        return self.base.geometry

    @property
    def eye_global_ids(self) -> np.ndarray:
        begin = self.base.geometry.full_node_count
        return np.arange(begin, begin + self.eyes.node_count, dtype=np.int64)

    @property
    def full_node_count(self) -> int:
        return self.base.geometry.full_node_count + self.eyes.node_count

    @property
    def full_reference_points_m(self) -> np.ndarray:
        return np.concatenate((self.base.full_reference_points_m, self.eyes.points_m))

    def extend_seed(
        self, fem_displacement: torch.Tensor, pose: torch.Tensor
    ) -> torch.Tensor:
        skull = self.base.extend_seed(fem_displacement, pose)
        eyes = fem_displacement.new_zeros((self.eyes.node_count, 3))
        return torch.cat((skull, eyes))

    def full_boundary_displacement(
        self,
        original_points: torch.Tensor,
        original_mandible_ids: torch.Tensor,
        pose: torch.Tensor,
    ) -> torch.Tensor:
        skull = self.base.full_boundary_displacement(
            original_points, original_mandible_ids, pose
        )
        eyes = original_points.new_zeros((self.eyes.node_count, 3))
        return torch.cat((skull, eyes))


def _extend_dof_map(original: DofMap, eye_count: int) -> DofMap:
    if eye_count <= 0:
        msg = "at least one rigid-eye node is required"
        raise ValueError(msg)
    appended = torch.arange(
        original.n_full,
        original.n_full + eye_count * original.dim,
        device=original.fixed_indices.device,
        dtype=original.fixed_indices.dtype,
    )
    return DofMap(
        n_points=original.n_points + eye_count,
        dim=original.dim,
        fixed_indices=torch.cat((original.fixed_indices, appended)),
        fixed_values=torch.cat(
            (
                original.fixed_values,
                torch.zeros(
                    len(appended),
                    device=original.fixed_values.device,
                    dtype=original.fixed_values.dtype,
                ),
            )
        ),
        free_indices=original.free_indices.clone(),
    )


def _build_contact(
    base: FullSkullJointPhysics, eyes: RigidEyeGeometry
) -> tuple[OwnedContact, dict[str, Any]]:
    geometry = base.full_skull.geometry
    config = copy.deepcopy(base.contact_definition["config"])
    soft_count = len(geometry.soft_global_ids)
    cranium_count = geometry.cranium_node_count
    mandible_count = geometry.mandible_node_count
    rest = np.concatenate(
        (
            geometry.fem_reference_points_m[geometry.soft_global_ids],
            geometry.cranium_points_m,
            geometry.mandible_points_m,
            eyes.points_m,
        )
    )
    faces = np.concatenate(
        (
            geometry.soft_faces,
            geometry.cranium_faces + soft_count,
            geometry.mandible_faces + soft_count + cranium_count,
            eyes.triangles + soft_count + cranium_count + mandible_count,
        )
    ).astype(np.int32, copy=False)
    global_ids = np.concatenate(
        (
            geometry.soft_global_ids,
            geometry.cranium_global_ids,
            geometry.mandible_global_ids,
            np.arange(
                geometry.full_node_count,
                geometry.full_node_count + eyes.node_count,
                dtype=np.int64,
            ),
        )
    )
    mesh = ipctk.CollisionMesh(rest, ipctk.edges(faces), faces)
    patches = np.concatenate(
        (
            np.zeros(soft_count, dtype=np.int32),
            np.ones(cranium_count + mandible_count + eyes.node_count, dtype=np.int32),
        )
    )
    mesh.can_collide = ipctk.make_vertex_patches_filter(patches)
    mesh.init_adjacencies()
    potential = ipctk.BarrierPotential(
        dhat=float(config["dhat_m"]),
        stiffness=float(config["stiffness_mpa"]),
        use_physical_barrier=True,
    )
    collision = OwnedContact(
        collision_mesh=mesh,
        indices=torch.as_tensor(global_ids, dtype=torch.long),
        potential=potential,
        broad_phase=ipctk.LBVH(),
        narrow_phase_ccd=ipctk.TightInclusionCCD(
            max_iterations=config.get("ccd_max_iterations", 10000000)
        ),
        min_distance=config.get("ccd_min_distance_m", 0.0),
        use_physical_barrier=True,
        vertices=torch.as_tensor(rest.copy()),
        collision_set_type=ipctk.NormalCollisions.CollisionSetType.__members__[
            config.get("collision_set_type", "IMPROVED_MAX_APPROX")
        ],
    )
    definition = {
        "schema": "joint-full-source-contact-with-rigid-eyes-v1",
        "base_full_skull": base.full_skull_receipt(),
        "eyes": eyes.binding_receipt(),
        "collision_vertices": len(global_ids),
        "soft_triangles": len(geometry.soft_faces),
        "cranium_triangles": len(geometry.cranium_faces),
        "mandible_triangles": len(geometry.mandible_faces),
        "eye_triangles": eyes.triangle_count,
        "excluded_source_triangles": 0,
        "soft_soft_contact": False,
        "rigid_rigid_contact": False,
        "soft_eye_contact": True,
        "eyes_fixed_at_registered_neutral_pose": True,
        "friction": "frictionless",
        "config": config,
        "ipc_version": ipctk.__version__,
        "neutral_contact_admitted": False,
        "required_before_admission": [
            "exact soft-rigid intersection audit",
            "numerically valid IPC state",
            "force convergence with the eye-inclusive model",
        ],
    }
    return collision, definition


class RigidEyeJointPhysics:
    """Existing frozen-neutral physics with fixed rigid-eye contact appended."""

    def __init__(
        self,
        base: FullSkullJointPhysics,
        eyes: RigidEyeGeometry,
        baseline: dict[str, dict[str, torch.Tensor]],
        *,
        neutral: FrozenNeutral,
    ) -> None:
        collision, definition = _build_contact(base, eyes)
        original_model = base.runtime.forward.model
        model = Model(
            dof_map=_extend_dof_map(original_model.dof_map, eyes.node_count),
            warp_model=original_model.warp_model,
            collision=collision,
            device=original_model.device,
        )
        model.set_materials(baseline)
        tolerances = base.runtime.tolerances
        solver = base.runtime.forward_solver
        self.runtime = Equilibrium(
            Forward(model),
            rtol=tolerances["rtol"],
            atol=tolerances["atol"],
            adjoint_rtol=tolerances["adjoint_rtol"],
            max_steps=tolerances["max_steps"],
            forward_method=solver["method"],
            newton_linear_rtol=solver["newton_linear_rtol"],
            newton_max_steps=solver["newton_max_steps"],
        )
        self.base = base
        self.baseline = baseline
        self.eyes = eyes
        self.full_skull = EyeInclusiveContactAdapter(
            base.full_skull, eyes, collision, definition
        )
        self.contact_definition = definition
        self.neutral = neutral

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    def boundary(self, pose: torch.Tensor) -> torch.Tensor:
        values = self.full_skull.full_boundary_displacement(
            self.points_t, self.jaw_t, pose
        )
        return values.flatten()[self.runtime.forward.model.dof_map.fixed_indices]

    def solve(
        self,
        skin_multiplier: torch.Tensor,
        active_stress: torch.Tensor,
        pose: torch.Tensor,
        seed: torch.Tensor,
        *,
        seed_pose: torch.Tensor,
        key: str,
    ) -> torch.Tensor:
        """Solve an expression with the hash-bound neutral baseline.

        Deliberately expose no bulk- or skin-baseline-stress arguments.  The
        adopted neutral carries those fields, and accepting replacement values
        here would silently invalidate the claimed fixed-material experiment.
        """
        active_ids = self.base.active_t
        if active_stress.shape != (len(active_ids), 3, 3):
            msg = "active_stress must contain one symmetric tensor per muscle tet"
            raise ValueError(msg)
        if not bool(torch.allclose(active_stress, active_stress.transpose(-1, -2))):
            msg = "active_stress must be exactly symmetric"
            raise ValueError(msg)
        if pose.shape != (6,):
            msg = "mandible pose must have six degrees of freedom"
            raise ValueError(msg)
        if seed.shape != (self.full_skull.geometry.fem_node_count, 3):
            msg = "seed must contain original FEM nodes only"
            raise ValueError(msg)
        extended_seed = self.full_skull.extend_seed(seed, seed_pose)
        full = self.runtime.solve(
            self.neutral.expression_materials(
                self.baseline,
                active_ids,
                skin_multiplier=skin_multiplier,
                active_stress=active_stress,
            ),
            self.boundary(pose),
            extended_seed,
            key=key,
        )
        return full[: self.full_skull.geometry.fem_node_count]

    def expression_materials(
        self, *, skin_multiplier: torch.Tensor, active_stress: torch.Tensor
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Return the only legal expression material variation."""
        return self.neutral.expression_materials(
            self.baseline,
            self.base.active_t,
            skin_multiplier=skin_multiplier,
            active_stress=active_stress,
        )

    def full_skull_receipt(self) -> dict[str, Any]:
        return {
            "schema": "joint-rigid-eye-physics-binding-v1",
            "frozen_neutral": {
                "directory": str(self.neutral.directory.resolve()),
                "manifest_sha256": sha256(self.neutral.directory / "manifest.json"),
            },
            "base": self.base.full_skull_receipt(),
            "eyes": self.eyes.binding_receipt(),
            "contact": copy.deepcopy(self.contact_definition),
            "original_fem_nodes": self.full_skull.geometry.fem_node_count,
            "appended_fixed_cranium_mandible_nodes": (
                self.full_skull.geometry.cranium_node_count
                + self.full_skull.geometry.mandible_node_count
            ),
            "appended_fixed_eye_nodes": self.eyes.node_count,
            "reported_displacement": "original FEM node slice only",
            "eye_motion_parameters": 0,
            "eye_pose": "fixed registered neutral source pose",
            "boundary_ccd": "linear vertex path includes FEM and appended mandible motion; eye displacement is fixed zero",
            "neutral_contact_admitted": False,
        }


def build_eye_collision_physics(
    neutral: FrozenNeutral, eyes_dir: Path
) -> tuple[RigidEyeJointPhysics, dict[str, dict[str, torch.Tensor]]]:
    """Build the current frozen baseline with eye-inclusive contact, unadmitted."""
    base, baseline = neutral.build_physics()
    eyes = load_rigid_eyes(eyes_dir)
    physics = RigidEyeJointPhysics(base, eyes, baseline, neutral=neutral)
    for name, fields in baseline.items():
        current = physics.runtime.forward.model.get_materials()[name]
        for key, value in fields.items():
            if not torch.equal(current[key], value):
                msg = f"eye-inclusive model changed frozen material {name}.{key}"
                raise ValueError(msg)
    return physics, baseline


__all__ = [
    "EYES_SCHEMA",
    "EyeInclusiveContactAdapter",
    "RigidEyeGeometry",
    "RigidEyeJointPhysics",
    "build_eye_collision_physics",
    "load_rigid_eyes",
]
