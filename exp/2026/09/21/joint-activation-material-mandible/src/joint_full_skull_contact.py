"""Complete registered source bones as appended fixed IPC obstacle nodes.

The original FEM node numbering and all constitutive potentials remain unchanged.
Complete source cranium and mandible meshes are appended to the model displacement
vector as fixed nodes without FEM cells.  This lets the existing collision Hessian,
Dirichlet CCD, and implicit fixed-value pullback include source-bone contact exactly.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import attrs
import ipctk
import numpy as np
import torch
from joint_contact import OwnedContact
from joint_equilibrium import Equilibrium, rigid_displacement
from joint_physics import JointPhysics

from liblaf.apple.forward import Forward, Model
from liblaf.apple.forward.dof_map import DofMap

GEOMETRY_SCHEMA = "joint-full-skull-initialization-audit-v1"
ADMISSION_SCHEMA = "joint-full-skull-contact-admission-v2"
CONTACT_SCHEMA = "joint-full-source-bone-contact-v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _integer_array(
    value: np.ndarray, *, name: str, ndim: int, width: int | None = None
) -> np.ndarray:
    result = np.asarray(value)
    if result.dtype.kind not in "iu" or result.ndim != ndim:
        msg = f"{name} must be a {ndim}-D integer array"
        raise ValueError(msg)
    if width is not None and result.shape[-1] != width:
        msg = f"{name} must have width {width}"
        raise ValueError(msg)
    return np.ascontiguousarray(result, dtype=np.int64)


def _points(value: np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.dtype != np.float64 or result.ndim != 2 or result.shape[1] != 3:
        msg = f"{name} must be float64 [N,3] metres"
        raise ValueError(msg)
    if not np.isfinite(result).all():
        msg = f"{name} contains non-finite coordinates"
        raise ValueError(msg)
    return np.ascontiguousarray(result)


def _faces(value: np.ndarray, *, name: str, point_count: int) -> np.ndarray:
    result = _integer_array(value, name=name, ndim=2, width=3)
    if result.size and (result.min() < 0 or result.max() >= point_count):
        msg = f"{name} references a point outside [0,{point_count})"
        raise ValueError(msg)
    if np.any(
        (result[:, 0] == result[:, 1])
        | (result[:, 1] == result[:, 2])
        | (result[:, 2] == result[:, 0])
    ):
        msg = f"{name} contains a degenerate index triangle"
        raise ValueError(msg)
    return np.ascontiguousarray(result, dtype=np.int32)


def _exact_ids(value: np.ndarray, *, name: str, count: int) -> np.ndarray:
    result = _integer_array(value, name=name, ndim=1)
    if result.shape != (count,) or not np.array_equal(result, np.arange(count)):
        msg = f"{name} must preserve every original source id in order"
        raise ValueError(msg)
    return result


@attrs.frozen
class FullSkullGeometry:
    """Immutable arrays from the full-source geometry audit."""

    fem_reference_points_m: np.ndarray
    soft_global_ids: np.ndarray
    soft_faces: np.ndarray
    fixed_global_ids: np.ndarray
    mandible_pivot_m: np.ndarray
    cranium_points_m: np.ndarray
    cranium_faces: np.ndarray
    cranium_source_vertex_ids: np.ndarray
    cranium_source_triangle_ids: np.ndarray
    mandible_points_m: np.ndarray
    mandible_faces: np.ndarray
    mandible_source_vertex_ids: np.ndarray
    mandible_source_triangle_ids: np.ndarray
    geometry_path: Path
    geometry_sha256: str
    audit_path: Path
    audit_sha256: str
    audit: dict[str, Any]

    @property
    def fem_node_count(self) -> int:
        return len(self.fem_reference_points_m)

    @property
    def cranium_node_count(self) -> int:
        return len(self.cranium_points_m)

    @property
    def mandible_node_count(self) -> int:
        return len(self.mandible_points_m)

    @property
    def full_node_count(self) -> int:
        return self.fem_node_count + self.cranium_node_count + self.mandible_node_count

    @property
    def cranium_global_ids(self) -> np.ndarray:
        return np.arange(
            self.fem_node_count,
            self.fem_node_count + self.cranium_node_count,
            dtype=np.int64,
        )

    @property
    def mandible_global_ids(self) -> np.ndarray:
        return np.arange(
            self.fem_node_count + self.cranium_node_count,
            self.full_node_count,
            dtype=np.int64,
        )

    @property
    def full_reference_points_m(self) -> np.ndarray:
        return np.concatenate(
            (
                self.fem_reference_points_m,
                self.cranium_points_m,
                self.mandible_points_m,
            )
        )

    def binding_receipt(self) -> dict[str, Any]:
        return {
            "schema": "joint-full-skull-geometry-binding-v1",
            "geometry_path": str(self.geometry_path.resolve()),
            "geometry_sha256": self.geometry_sha256,
            "audit_path": str(self.audit_path.resolve()),
            "audit_sha256": self.audit_sha256,
            "input_arrays_sha256": self.audit["input_arrays_sha256"],
            "input_manifest_sha256": self.audit["input_manifest_sha256"],
            "units": self.audit["units"],
            "frame": self.audit["frame"],
            "fem_nodes": self.fem_node_count,
            "soft_boundary_nodes": len(self.soft_global_ids),
            "soft_boundary_triangles": len(self.soft_faces),
            "cranium_vertices": self.cranium_node_count,
            "cranium_triangles": len(self.cranium_faces),
            "mandible_vertices": self.mandible_node_count,
            "mandible_triangles": len(self.mandible_faces),
            "complete_source_triangles_retained": True,
            "source_coordinates_changed": False,
            "excluded_source_triangles": 0,
        }


def load_full_skull_geometry(  # noqa: C901, PLR0912, PLR0915 - binding audit
    geometry_path: Path, audit_path: Path
) -> FullSkullGeometry:
    """Load exact audit arrays; loading does not claim contact admission."""
    geometry_path = geometry_path.resolve()
    audit_path = audit_path.resolve()
    audit = json.loads(audit_path.read_text())
    if audit["schema"] != GEOMETRY_SCHEMA:
        msg = f"unexpected full-skull audit schema: {audit['schema']}"
        raise ValueError(msg)
    if audit["units"] != "metres":
        msg = "full-skull geometry must remain in metres"
        raise ValueError(msg)
    if audit["frame"] != "unchanged registered source and FEM world frame":
        msg = "full-skull geometry must remain in the registered FEM world frame"
        raise ValueError(msg)
    geometry_hash = sha256(geometry_path)
    if audit["geometry"]["sha256"] != geometry_hash:
        msg = "full-skull geometry hash does not match its audit"
        raise ValueError(msg)
    for bone in ("cranium", "mandible"):
        record = audit["bones"][bone]
        if record["all_source_triangles_retained"] is not True:
            msg = f"{bone} audit does not retain every source triangle"
            raise ValueError(msg)
        if record["source_coordinates_changed"] is not False:
            msg = f"{bone} source coordinates were changed"
            raise ValueError(msg)
    with np.load(geometry_path) as arrays:
        required = {
            "fem_reference_points_m",
            "soft_global_ids",
            "soft_faces",
            "fixed_global_ids",
            "mandible_pivot_m",
            "cranium_points_m",
            "cranium_faces",
            "cranium_source_vertex_ids",
            "cranium_source_triangle_ids",
            "mandible_points_m",
            "mandible_faces",
            "mandible_source_vertex_ids",
            "mandible_source_triangle_ids",
        }
        missing = required.difference(arrays.files)
        if missing:
            msg = f"full-skull geometry is missing arrays: {sorted(missing)}"
            raise ValueError(msg)
        fem = _points(arrays["fem_reference_points_m"], name="fem_reference_points_m")
        soft_ids = _integer_array(
            arrays["soft_global_ids"], name="soft_global_ids", ndim=1
        )
        if soft_ids.size and (soft_ids.min() < 0 or soft_ids.max() >= len(fem)):
            msg = "soft_global_ids references an invalid FEM node"
            raise ValueError(msg)
        if len(np.unique(soft_ids)) != len(soft_ids):
            msg = "soft_global_ids contains duplicates"
            raise ValueError(msg)
        soft_faces = _faces(
            arrays["soft_faces"], name="soft_faces", point_count=len(soft_ids)
        )
        fixed_ids = _integer_array(
            arrays["fixed_global_ids"], name="fixed_global_ids", ndim=1
        )
        cranium = _points(arrays["cranium_points_m"], name="cranium_points_m")
        mandible = _points(arrays["mandible_points_m"], name="mandible_points_m")
        cranium_faces = _faces(
            arrays["cranium_faces"],
            name="cranium_faces",
            point_count=len(cranium),
        )
        mandible_faces = _faces(
            arrays["mandible_faces"],
            name="mandible_faces",
            point_count=len(mandible),
        )
        cranium_vertex_ids = _exact_ids(
            arrays["cranium_source_vertex_ids"],
            name="cranium_source_vertex_ids",
            count=len(cranium),
        )
        cranium_triangle_ids = _exact_ids(
            arrays["cranium_source_triangle_ids"],
            name="cranium_source_triangle_ids",
            count=len(cranium_faces),
        )
        mandible_vertex_ids = _exact_ids(
            arrays["mandible_source_vertex_ids"],
            name="mandible_source_vertex_ids",
            count=len(mandible),
        )
        mandible_triangle_ids = _exact_ids(
            arrays["mandible_source_triangle_ids"],
            name="mandible_source_triangle_ids",
            count=len(mandible_faces),
        )
        pivot = np.asarray(arrays["mandible_pivot_m"])
        if (
            pivot.dtype != np.float64
            or pivot.shape != (3,)
            or not np.isfinite(pivot).all()
        ):
            msg = "mandible_pivot_m must be finite float64 [3] metres"
            raise ValueError(msg)
    if len(fem) != audit["fem_nodes"]:
        msg = "FEM node count differs from the full-skull audit"
        raise ValueError(msg)
    if (
        len(soft_ids) != audit["soft_boundary_nodes"]
        or len(soft_faces) != audit["soft_boundary_triangles"]
    ):
        msg = "soft boundary counts differ from the full-skull audit"
        raise ValueError(msg)
    for name, points, faces in (
        ("cranium", cranium, cranium_faces),
        ("mandible", mandible, mandible_faces),
    ):
        record = audit["bones"][name]
        if len(points) != record["vertices"] or len(faces) != record["triangles"]:
            msg = f"{name} source counts differ from the full-skull audit"
            raise ValueError(msg)
    return FullSkullGeometry(
        fem_reference_points_m=fem.copy(),
        soft_global_ids=soft_ids.copy(),
        soft_faces=soft_faces.copy(),
        fixed_global_ids=fixed_ids.copy(),
        mandible_pivot_m=pivot.copy(),
        cranium_points_m=cranium.copy(),
        cranium_faces=cranium_faces.copy(),
        cranium_source_vertex_ids=cranium_vertex_ids.copy(),
        cranium_source_triangle_ids=cranium_triangle_ids.copy(),
        mandible_points_m=mandible.copy(),
        mandible_faces=mandible_faces.copy(),
        mandible_source_vertex_ids=mandible_vertex_ids.copy(),
        mandible_source_triangle_ids=mandible_triangle_ids.copy(),
        geometry_path=geometry_path,
        geometry_sha256=geometry_hash,
        audit_path=audit_path,
        audit_sha256=sha256(audit_path),
        audit=audit,
    )


def validate_full_skull_admission(  # noqa: C901, PLR0912, PLR0915 - receipt audit
    receipt: dict[str, Any], geometry: FullSkullGeometry
) -> dict[str, Any]:
    """Require a hash-bound soft-bone initialization before mechanics."""
    if receipt["schema"] != ADMISSION_SCHEMA or receipt["success"] is not True:
        msg = "full-skull contact geometry is not admitted"
        raise ValueError(msg)
    if receipt["geometry_sha256"] != geometry.geometry_sha256:
        msg = "full-skull admission binds a different geometry artifact"
        raise ValueError(msg)
    if receipt["geometry_audit_sha256"] != geometry.audit_sha256:
        msg = "full-skull admission binds a different geometry audit"
        raise ValueError(msg)
    if receipt["complete_source_triangles_retained"] is not True:
        msg = "full-skull admission does not retain every source triangle"
        raise ValueError(msg)
    if receipt["source_coordinates_changed"] is not False:
        msg = "full-skull admission changes source bone coordinates"
        raise ValueError(msg)
    if receipt["excluded_source_triangles"] != 0:
        msg = "full-skull admission excludes source bone triangles"
        raise ValueError(msg)
    if receipt["initialization_intersection_free"] is not True:
        msg = "full-skull initialization is not intersection-free"
        raise ValueError(msg)
    if receipt["fixed_nodes_unchanged"] is not True:
        msg = "full-skull initialization moved original fixed FEM nodes"
        raise ValueError(msg)
    if not 0.25 <= receipt["detF_min"] <= receipt["detF_max"] <= 2.0:
        msg = "full-skull initialization violates the declared volume bounds"
        raise ValueError(msg)
    if not 0 <= receipt["observation_surface_rms_m"] <= 0.00025:
        msg = "full-skull initialization violates the observation-surface budget"
        raise ValueError(msg)
    ipc = receipt["ipc_initialization"]
    if ipc["enabled"] is not True or ipc["contact_numerically_valid"] is not True:
        msg = "full-skull initialization lacks valid soft-bone IPC evidence"
        raise ValueError(msg)
    if ipc["minimum_active_distance_m"] is not None and not (
        ipc["minimum_active_distance_m"] > 0
    ):
        msg = "full-skull initialization has a nonpositive active IPC distance"
        raise ValueError(msg)
    if receipt["equilibrium_converged"] is not False:
        msg = "initialization admission cannot claim equilibrium convergence"
        raise ValueError(msg)
    if receipt["final_launch_ready"] is not False:
        msg = "soft-bone initialization cannot claim final-launch readiness"
        raise ValueError(msg)

    initialization_path = Path(receipt["initialization_path"]).resolve()
    if sha256(initialization_path) != receipt["initialization_sha256"]:
        msg = "full-skull initialization artifact hash mismatch"
        raise ValueError(msg)
    key = receipt["initialization_array_key"]
    if key != "initial_displacement_m":
        msg = "unexpected full-skull initialization array key"
        raise ValueError(msg)
    with np.load(initialization_path) as archive:
        if set(archive.files) != {key}:
            msg = "full-skull initialization artifact has unexpected arrays"
            raise ValueError(msg)
        displacement = np.asarray(archive[key])
    if (
        displacement.dtype != np.float64
        or displacement.shape != (geometry.fem_node_count, 3)
        or not np.isfinite(displacement).all()
    ):
        msg = "full-skull initialization must be finite float64 [fem_nodes,3]"
        raise ValueError(msg)
    displacement_hash = hashlib.sha256(
        np.ascontiguousarray(displacement).tobytes()
    ).hexdigest()
    if displacement_hash != receipt["initialization_displacement_sha256"]:
        msg = "full-skull initialization displacement hash mismatch"
        raise ValueError(msg)
    if np.any(displacement[geometry.fixed_global_ids] != 0):
        msg = "full-skull initialization artifact moves original fixed nodes"
        raise ValueError(msg)
    return copy.deepcopy(receipt)


def load_admitted_initialization(
    receipt: dict[str, Any], geometry: FullSkullGeometry
) -> np.ndarray:
    """Return the exact FEM displacement bound by a validated v2 admission."""
    validated = validate_full_skull_admission(receipt, geometry)
    with np.load(Path(validated["initialization_path"])) as archive:
        displacement = np.asarray(
            archive[validated["initialization_array_key"]], dtype=np.float64
        )
    return np.ascontiguousarray(displacement)


@attrs.frozen
class FullSkullContactAdapter:
    """Binding between original FEM nodes and appended complete source bones."""

    geometry: FullSkullGeometry
    collision: OwnedContact
    contact_definition: dict[str, Any]

    @property
    def full_reference_points_m(self) -> np.ndarray:
        return self.geometry.full_reference_points_m

    def extend_seed(
        self, fem_displacement: torch.Tensor, pose: torch.Tensor
    ) -> torch.Tensor:
        """Append fixed cranium and rigid mandible displacement to an FEM seed."""
        if fem_displacement.shape != (self.geometry.fem_node_count, 3):
            msg = "FEM seed must contain every original FEM node and no appended node"
            raise ValueError(msg)
        cranium = fem_displacement.new_zeros((self.geometry.cranium_node_count, 3))
        mandible_points = torch.as_tensor(
            self.geometry.mandible_points_m,
            device=fem_displacement.device,
            dtype=fem_displacement.dtype,
        )
        pivot = torch.as_tensor(
            self.geometry.mandible_pivot_m,
            device=fem_displacement.device,
            dtype=fem_displacement.dtype,
        )
        mandible = rigid_displacement(mandible_points, pivot, pose)
        return torch.cat((fem_displacement, cranium, mandible))

    def full_boundary_displacement(
        self,
        original_points: torch.Tensor,
        original_mandible_ids: torch.Tensor,
        pose: torch.Tensor,
    ) -> torch.Tensor:
        """Return pose-dependent fixed displacements on original and appended nodes."""
        if original_points.shape != (self.geometry.fem_node_count, 3):
            msg = "original_points does not match the bound FEM geometry"
            raise ValueError(msg)
        result = original_points.new_zeros((self.geometry.full_node_count, 3))
        pivot = torch.as_tensor(
            self.geometry.mandible_pivot_m,
            device=original_points.device,
            dtype=original_points.dtype,
        )
        result[original_mandible_ids] = rigid_displacement(
            original_points[original_mandible_ids], pivot, pose
        )
        source_mandible = torch.as_tensor(
            self.geometry.mandible_points_m,
            device=original_points.device,
            dtype=original_points.dtype,
        )
        source_ids = torch.as_tensor(
            self.geometry.mandible_global_ids,
            device=original_points.device,
            dtype=torch.long,
        )
        result[source_ids] = rigid_displacement(source_mandible, pivot, pose)
        return result


def build_full_skull_contact(
    geometry: FullSkullGeometry,
    config: dict[str, Any],
) -> FullSkullContactAdapter:
    """Build soft-versus-complete-source-bones IPC without triangle exclusions."""
    if config["schema"] != CONTACT_SCHEMA or config["enabled"] is not True:
        msg = "unexpected full-source contact configuration"
        raise ValueError(msg)
    if config["surface_selection"] != "pure-soft-vs-complete-source-bones":
        msg = "full-source contact must use the audited pure-soft surface"
        raise ValueError(msg)
    if config["attachment_policy"] != "no-source-triangle-exclusions":
        msg = "full-source contact cannot broaden attachment exclusions"
        raise ValueError(msg)
    if config["friction"] != "frictionless":
        msg = "only the validated frictionless contact law is implemented"
        raise ValueError(msg)
    if config["dhat_m"] <= 0 or config["stiffness_mpa"] <= 0:
        msg = "contact distance and stiffness must be positive"
        raise ValueError(msg)

    soft_count = len(geometry.soft_global_ids)
    cranium_count = geometry.cranium_node_count
    mandible_count = geometry.mandible_node_count
    rest = np.concatenate(
        (
            geometry.fem_reference_points_m[geometry.soft_global_ids],
            geometry.cranium_points_m,
            geometry.mandible_points_m,
        )
    )
    faces = np.concatenate(
        (
            geometry.soft_faces,
            geometry.cranium_faces + soft_count,
            geometry.mandible_faces + soft_count + cranium_count,
        )
    ).astype(np.int32, copy=False)
    expected_faces = (
        len(geometry.soft_faces)
        + len(geometry.cranium_faces)
        + len(geometry.mandible_faces)
    )
    if len(faces) != expected_faces:
        msg = "source triangle assembly changed the audited triangle count"
        raise ValueError(msg)
    global_ids = np.concatenate(
        (
            geometry.soft_global_ids,
            geometry.cranium_global_ids,
            geometry.mandible_global_ids,
        )
    )
    mesh = ipctk.CollisionMesh(rest, ipctk.edges(faces), faces)
    # One patch for FEM soft tissue and one for both source bones gives exactly
    # soft-bone pairs. It rejects soft-soft, bone-bone, and either bone's self pairs.
    patches = np.concatenate(
        (
            np.zeros(soft_count, dtype=np.int32),
            np.ones(cranium_count + mandible_count, dtype=np.int32),
        )
    )
    mesh.can_collide = ipctk.make_vertex_patches_filter(patches)
    mesh.init_adjacencies()
    potential = ipctk.BarrierPotential(
        dhat=float(config["dhat_m"]),
        stiffness=float(config["stiffness_mpa"]),
        use_physical_barrier=True,
    )
    contact = OwnedContact(
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
        **geometry.binding_receipt(),
        "schema": "joint-full-source-contact-map-v1",
        "collision_vertices": len(global_ids),
        "soft_triangles": len(geometry.soft_faces),
        "cranium_triangles": len(geometry.cranium_faces),
        "mandible_triangles": len(geometry.mandible_faces),
        "excluded_source_triangles": 0,
        "soft_soft_contact": False,
        "bone_bone_contact": False,
        "source_bone_self_contact": False,
        "friction": "frictionless",
        "config": copy.deepcopy(config),
        "ipc_version": ipctk.__version__,
    }
    return FullSkullContactAdapter(geometry, contact, definition)


def extend_dof_map(original: DofMap, geometry: FullSkullGeometry) -> DofMap:
    """Append every source-bone coordinate as fixed while preserving original IDs."""
    if original.n_points != geometry.fem_node_count:
        msg = "original DOF map does not match the audited FEM node count"
        raise ValueError(msg)
    device = original.fixed_indices.device
    appended = torch.arange(
        original.n_full,
        geometry.full_node_count * original.dim,
        device=device,
        dtype=original.fixed_indices.dtype,
    )
    return DofMap(
        n_points=geometry.full_node_count,
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


class FullSkullJointPhysics(JointPhysics):
    """JointPhysics with complete source bones appended as rigid obstacle DOFs."""

    def __init__(
        self,
        *args: Any,
        full_skull_geometry: FullSkullGeometry,
        full_skull_admission: dict[str, Any],
        full_skull_contact_config: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        if kwargs.get("contact_config") is not None:
            msg = "partial FEM contact cannot be combined with full-source contact"
            raise ValueError(msg)
        kwargs["contact_config"] = None
        super().__init__(*args, **kwargs)
        if not np.array_equal(self.points, full_skull_geometry.fem_reference_points_m):
            msg = "JointPhysics volume points differ from the full-skull geometry"
            raise ValueError(msg)
        # Admit the immutable source geometry first. Its old support union is
        # provenance, not authority to constrain FEM nodes in the current run.
        admission = validate_full_skull_admission(
            full_skull_admission, full_skull_geometry
        )
        fixed = np.flatnonzero(np.asarray(self.mesh.point_data["IsFixed"]))
        assert np.array_equal(
            np.asarray(self.mesh.point_data["FixedMask"]).any(axis=1).nonzero()[0],
            fixed,
        )
        source_fixed = np.asarray(full_skull_geometry.fixed_global_ids)
        assert np.isin(fixed, source_fixed).all()
        self.fixed_boundary_receipt = {
            "schema": "isfixed-fem-boundary-v1",
            "policy": "Only IsFixed prescribes original FEM nodes; Mandible selects rigid motion within that set.",
            "source_geometry_sha256": full_skull_geometry.geometry_sha256,
            "source_fixed_count": int(source_fixed.size),
            "source_fixed_ids_sha256": hashlib.sha256(
                np.asarray(source_fixed, dtype="<i8").tobytes()
            ).hexdigest(),
            "runtime_fixed_count": int(fixed.size),
            "runtime_fixed_ids_sha256": hashlib.sha256(
                np.asarray(fixed, dtype="<i8").tobytes()
            ).hexdigest(),
            "runtime_mandible_fixed_count": int(self.jaw_t.numel()),
            "anatomy_labels_preserved": True,
            "appended_rigid_geometry_prescribed": True,
            "historical_equilibrium_reused_as_converged": False,
        }
        full_skull_geometry = attrs.evolve(
            full_skull_geometry, fixed_global_ids=fixed.copy()
        )
        # The admitted displacement was zero on a superset of IsFixed. Validate
        # that same geometric seed under the corrected support before use.
        validate_full_skull_admission(admission, full_skull_geometry)
        adapter = build_full_skull_contact(
            full_skull_geometry, full_skull_contact_config
        )
        original_model = self.runtime.forward.model
        model = Model(
            dof_map=extend_dof_map(original_model.dof_map, full_skull_geometry),
            warp_model=original_model.warp_model,
            collision=adapter.collision,
            device=original_model.device,
        )
        tolerances = self.runtime.tolerances
        solver = self.runtime.forward_solver
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
        self.full_skull = adapter
        self.full_skull_admission = admission
        self.full_skull_initialization = load_admitted_initialization(
            admission, full_skull_geometry
        )
        self.contact_definition = adapter.contact_definition

    def boundary(self, pose: torch.Tensor) -> torch.Tensor:
        values = self.full_skull.full_boundary_displacement(
            self.points_t, self.jaw_t, pose
        )
        return values.flatten()[self.runtime.forward.model.dof_map.fixed_indices]

    def solve(
        self,
        bulk_stress: torch.Tensor,
        skin_resultant_n_m: torch.Tensor,
        skin_multiplier: torch.Tensor,
        active_stress: torch.Tensor | None,
        pose: torch.Tensor,
        seed: torch.Tensor,
        *,
        seed_pose: torch.Tensor,
        key: str,
    ) -> torch.Tensor:
        """Solve with explicit accepted seed pose so source-bone CCD is complete."""
        extended_seed = self.full_skull.extend_seed(seed, seed_pose)
        full = self.runtime.solve(
            self.materials(
                bulk_stress, skin_resultant_n_m, skin_multiplier, active_stress
            ),
            self.boundary(pose),
            extended_seed,
            key=key,
        )
        return full[: self.full_skull.geometry.fem_node_count]

    def full_skull_receipt(self) -> dict[str, Any]:
        return {
            "schema": "joint-full-skull-physics-binding-v1",
            "geometry": self.full_skull.geometry.binding_receipt(),
            "fixed_boundary": copy.deepcopy(self.fixed_boundary_receipt),
            "admission": copy.deepcopy(self.full_skull_admission),
            "contact": copy.deepcopy(self.contact_definition),
            "original_fem_nodes": self.full_skull.geometry.fem_node_count,
            "appended_fixed_nodes": (
                self.full_skull.geometry.cranium_node_count
                + self.full_skull.geometry.mandible_node_count
            ),
            "reported_displacement": "original FEM node slice only",
            "pose_pullback": "existing fixed-DOF implicit HVP over original and appended mandible nodes",
            "boundary_ccd": "linear vertex path includes original and appended prescribed mandible motion",
        }


__all__ = [
    "ADMISSION_SCHEMA",
    "CONTACT_SCHEMA",
    "FullSkullContactAdapter",
    "FullSkullGeometry",
    "FullSkullJointPhysics",
    "build_full_skull_contact",
    "extend_dof_map",
    "load_admitted_initialization",
    "load_full_skull_geometry",
    "validate_full_skull_admission",
]
