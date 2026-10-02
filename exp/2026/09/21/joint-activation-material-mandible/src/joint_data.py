# ruff: noqa: EM101, EM102, TRY003
"""Frozen input contract for the joint face inverse experiment."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

GROUP = Path(__file__).resolve().parent.parent
DEFAULT_NPZ = GROUP / "data/prepared/inputs.npz"
DEFAULT_MANIFEST = GROUP / "data/prepared/manifest.json"


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class PreparedInputs:
    """Compact arrays plus their auditable manifest.

    The tetrahedral volume and skin are deliberately referenced by immutable
    paths and hashes rather than copied into the NPZ.
    """

    arrays: dict[str, np.ndarray]
    manifest: dict[str, Any]
    npz_path: Path
    manifest_path: Path

    @classmethod
    def load(
        cls,
        npz_path: Path = DEFAULT_NPZ,
        manifest_path: Path = DEFAULT_MANIFEST,
        *,
        verify_sources: bool = True,
    ) -> PreparedInputs:
        npz_path = Path(npz_path).resolve()
        manifest_path = Path(manifest_path).resolve()
        manifest = json.loads(manifest_path.read_text())
        if manifest["schema_version"] != 1:
            raise ValueError(
                f"unsupported prepared-input schema: {manifest['schema_version']}"
            )
        if file_sha256(npz_path) != manifest["artifact"]["sha256"]:
            raise ValueError("prepared NPZ SHA-256 mismatch")
        with np.load(npz_path, allow_pickle=False) as archive:
            arrays = {name: np.asarray(archive[name]) for name in archive.files}
        expected = manifest["arrays"]
        if set(arrays) != set(expected):
            raise ValueError("prepared NPZ array names differ from manifest")
        for name, value in arrays.items():
            record = expected[name]
            if (
                list(value.shape) != record["shape"]
                or value.dtype.str != record["dtype"]
            ):
                raise ValueError(f"prepared array layout mismatch: {name}")
            if array_sha256(value) != record["sha256"]:
                raise ValueError(f"prepared array SHA-256 mismatch: {name}")
        if verify_sources:
            for record in manifest["sources"].values():
                path = Path(record["path"])
                if not path.is_file() or file_sha256(path) != record["sha256"]:
                    raise ValueError(f"frozen source identity mismatch: {path}")
        cls._validate(arrays, manifest)
        return cls(
            arrays=arrays,
            manifest=manifest,
            npz_path=npz_path,
            manifest_path=manifest_path,
        )

    @staticmethod
    def _validate(arrays: dict[str, np.ndarray], manifest: dict[str, Any]) -> None:
        active = arrays["active_cell_ids"]
        n_active = len(active)
        if n_active != 288_235 or np.unique(active).size != n_active:
            raise ValueError("historical active-cell contract changed")
        graph_i = arrays["graph_i"]
        graph_j = arrays["graph_j"]
        conductance = arrays["graph_conductance_m"]
        if not (graph_i.shape == graph_j.shape == conductance.shape):
            raise ValueError("activation graph arrays have inconsistent shapes")
        if len(graph_i) and (
            graph_i.min() < 0
            or graph_j.min() < 0
            or graph_i.max() >= n_active
            or graph_j.max() >= n_active
        ):
            raise ValueError("activation graph contains an invalid packed index")
        if (
            np.any(graph_i >= graph_j)
            or np.any(~np.isfinite(conductance))
            or np.any(conductance <= 0)
        ):
            raise ValueError(
                "activation graph is not a unique positive undirected graph"
            )
        mandible = arrays["mandible_node_ids"]
        cranium = arrays["cranium_node_ids"]
        if np.intersect1d(mandible, cranium).size:
            raise ValueError("mandible and cranium supports overlap")
        ids = arrays["observation_node_ids"]
        weights = arrays["observation_weight_normalized"]
        targets = arrays["target_displacement_m"]
        if targets.shape != (len(manifest["cohort"]["names"]), len(ids), 3):
            raise ValueError("target displacement layout differs from cohort")
        if (
            np.any(~np.isfinite(targets))
            or np.any(weights <= 0)
            or not np.isclose(weights.sum(), 1.0)
        ):
            raise ValueError("invalid observation target or weight")
        adapt = arrays["reserved_adapt_mask"].astype(bool)
        score = arrays["reserved_score_mask"].astype(bool)
        if np.any(adapt & score) or not np.all(adapt | score):
            raise ValueError(
                "reserved observation split is not disjoint and exhaustive"
            )

    @property
    def target_names(self) -> tuple[str, ...]:
        return tuple(self.manifest["cohort"]["names"])

    def target(self, name: str) -> np.ndarray:
        return self.arrays["target_displacement_m"][self.target_names.index(name)]

    @property
    def volume_path(self) -> Path:
        return Path(self.manifest["fixture"]["volume_path"])

    @property
    def skin_path(self) -> Path:
        return Path(self.manifest["fixture"]["skin_path"])


def _rotation_matrix(rotation_vector: np.ndarray) -> np.ndarray:
    """Return the Rodrigues matrix for a world-frame rotation vector."""
    theta = float(np.linalg.norm(rotation_vector))
    if theta == 0.0:
        return np.eye(3)
    axis = rotation_vector / theta
    x, y, z = axis
    skew = np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    return np.eye(3) + np.sin(theta) * skew + (1 - np.cos(theta)) * (skew @ skew)


def _collision_geometry(
    first: Any, second: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return paired cell IDs, contact segments, midpoints, and lengths."""
    import pyvista as pv
    from vtkmodules.vtkCommonMath import vtkMatrix4x4
    from vtkmodules.vtkCommonTransforms import vtkTransform
    from vtkmodules.vtkFiltersModeling import vtkCollisionDetectionFilter

    collision = vtkCollisionDetectionFilter()
    collision.SetInputData(0, first)
    collision.SetTransform(0, vtkTransform())
    collision.SetInputData(1, second)
    collision.SetMatrix(1, vtkMatrix4x4())
    collision.SetBoxTolerance(0.0)
    collision.SetCellTolerance(0.0)
    collision.SetNumberOfCellsPerNode(2)
    collision.SetCollisionModeToAllContacts()
    collision.Update()
    count = collision.GetNumberOfContacts()
    if count == 0:
        return (
            np.empty((0, 2), dtype=np.int64),
            np.empty((0, 2, 3)),
            np.empty((0, 3)),
            np.empty(0),
        )
    first_ids = np.asarray(
        pv.wrap(collision.GetOutput(0)).field_data["ContactCells"], dtype=np.int64
    )
    second_ids = np.asarray(
        pv.wrap(collision.GetOutput(1)).field_data["ContactCells"], dtype=np.int64
    )
    segments = np.asarray(pv.wrap(collision.GetContactsOutput()).points).reshape(
        count, 2, 3
    )
    return (
        np.column_stack((first_ids, second_ids)),
        segments,
        segments.mean(axis=1),
        np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1),
    )


def _collision_details(
    first: Any, second: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return paired cell IDs, contact midpoints, and segment lengths."""
    pairs, _, midpoints, lengths = _collision_geometry(first, second)
    return pairs, midpoints, lengths


def _collision_pairs(first: Any, second: Any) -> tuple[np.ndarray, np.ndarray]:
    pairs, midpoints, _ = _collision_details(first, second)
    return pairs, midpoints


def _fem_lip_surfaces(
    volume: Any, points: np.ndarray
) -> tuple[Any, Any, dict[str, int]]:
    """Build a conservative, exactly node-mapped FEM lip audit surface."""
    boundary = volume.extract_surface(algorithm=None).triangulate()
    original = np.asarray(boundary.point_data["vtkOriginalPointIds"], dtype=np.int64)
    boundary.points = points[original]
    boundary.cell_data["BoundaryCellId"] = np.arange(boundary.n_cells, dtype=np.int64)
    names = [
        str(value) for value in np.asarray(volume.field_data["GroupName"]).reshape(-1)
    ]
    point_group = np.asarray(volume.point_data["GroupId"], dtype=np.int32)[original]
    faces = np.asarray(boundary.faces).reshape(-1, 4)[:, 1:]
    upper_names = ("LipTop", "LipInnerTop", "LipOuterTop")
    lower_names = ("LipBottom", "LipInnerBottom", "LipOuterBottom")
    upper_ids = [names.index(name) for name in upper_names]
    lower_ids = [names.index(name) for name in lower_names]
    upper = np.any(np.isin(point_group[faces], upper_ids), axis=1)
    lower = np.any(np.isin(point_group[faces], lower_ids), axis=1)
    ambiguous = upper & lower
    upper &= ~ambiguous
    lower &= ~ambiguous

    def subset(mask: np.ndarray) -> Any:
        return (
            boundary.extract_cells(np.flatnonzero(mask))
            .extract_surface(algorithm=None)
            .triangulate()
        )

    return (
        subset(upper),
        subset(lower),
        {
            "upper_triangles": int(upper.sum()),
            "lower_triangles": int(lower.sum()),
            "ambiguous_triangles_omitted": int(ambiguous.sum()),
        },
    )


def _fem_jaw_oral_surfaces(
    volume: Any, points: np.ndarray
) -> tuple[Any, Any, Any, dict[str, int]]:
    """Build exact-node FEM mandible and oral boundary classifiers."""
    boundary = volume.extract_surface(algorithm=None).triangulate()
    original = np.asarray(boundary.point_data["vtkOriginalPointIds"], dtype=np.int64)
    boundary.points = points[original]
    boundary.cell_data["BoundaryCellId"] = np.arange(boundary.n_cells, dtype=np.int64)
    names = [
        str(value) for value in np.asarray(volume.field_data["GroupName"]).reshape(-1)
    ]
    group = np.asarray(volume.point_data["GroupId"], dtype=np.int32)[original]
    faces = np.asarray(boundary.faces).reshape(-1, 4)[:, 1:]
    mandible_id = names.index("Mandible")
    upper_ids = [
        names.index(name)
        for name in ("LipTop", "LipInnerTop", "LipOuterTop", "MouthSocketTop")
    ]
    lower_ids = [
        names.index(name)
        for name in (
            "LipBottom",
            "LipInnerBottom",
            "LipOuterBottom",
            "MouthSocketBottom",
        )
    ]
    mandible = np.all(group[faces] == mandible_id, axis=1)
    upper = np.any(np.isin(group[faces], upper_ids), axis=1)
    lower = np.any(np.isin(group[faces], lower_ids), axis=1)
    ambiguous = (mandible & (upper | lower)) | (upper & lower)
    mandible &= ~ambiguous
    upper &= ~ambiguous
    lower &= ~ambiguous

    def subset(mask: np.ndarray) -> Any:
        return (
            boundary.extract_cells(np.flatnonzero(mask))
            .extract_surface(algorithm=None)
            .triangulate()
        )

    return (
        subset(mandible),
        subset(upper),
        subset(lower),
        {
            "mandible_triangles": int(mandible.sum()),
            "upper_oral_triangles": int(upper.sum()),
            "lower_oral_triangles": int(lower.sum()),
            "ambiguous_triangles_omitted": int(ambiguous.sum()),
        },
    )


def _fem_group_surface(volume: Any, points: np.ndarray, name: str) -> Any:
    """Build an exact-node homogeneous boundary surface for one GroupId."""
    boundary = volume.extract_surface(algorithm=None).triangulate()
    original = np.asarray(boundary.point_data["vtkOriginalPointIds"], dtype=np.int64)
    boundary.points = points[original]
    boundary.cell_data["BoundaryCellId"] = np.arange(boundary.n_cells, dtype=np.int64)
    names = [
        str(value) for value in np.asarray(volume.field_data["GroupName"]).reshape(-1)
    ]
    group = np.asarray(volume.point_data["GroupId"], dtype=np.int32)[original]
    faces = np.asarray(boundary.faces).reshape(-1, 4)[:, 1:]
    mask = np.all(group[faces] == names.index(name), axis=1)
    return (
        boundary.extract_cells(np.flatnonzero(mask))
        .extract_surface(algorithm=None)
        .triangulate()
    )


def _fem_bone_soft_surfaces(
    volume: Any, points: np.ndarray
) -> tuple[Any, Any, Any, dict[str, int]]:
    """Partition one FEM boundary into pure bone, pure soft, and bonded seams."""
    boundary = volume.extract_surface(algorithm=None).triangulate()
    original = np.asarray(boundary.point_data["vtkOriginalPointIds"], dtype=np.int64)
    boundary.points = points[original]
    boundary.cell_data["BoundaryCellId"] = np.arange(boundary.n_cells, dtype=np.int64)
    names = [
        str(value) for value in np.asarray(volume.field_data["GroupName"]).reshape(-1)
    ]
    group = np.asarray(volume.point_data["GroupId"], dtype=np.int32)[original]
    faces = np.asarray(boundary.faces).reshape(-1, 4)[:, 1:]
    is_cranium = group[faces] == names.index("Cranium")
    is_mandible = group[faces] == names.index("Mandible")
    cranium = np.all(is_cranium, axis=1)
    mandible = np.all(is_mandible, axis=1)
    soft = np.all(~(is_cranium | is_mandible), axis=1)
    mixed = ~(cranium | mandible | soft)

    def subset(mask: np.ndarray) -> Any:
        return (
            boundary.extract_cells(np.flatnonzero(mask))
            .extract_surface(algorithm=None)
            .triangulate()
        )

    return (
        subset(cranium),
        subset(mandible),
        subset(soft),
        {
            "boundary_triangles": int(boundary.n_cells),
            "pure_cranium_triangles": int(cranium.sum()),
            "pure_mandible_triangles": int(mandible.sum()),
            "pure_soft_triangles": int(soft.sum()),
            "bonded_mixed_transition_triangles": int(mixed.sum()),
        },
    )


def _source_pose_audit(
    prepared: PreparedInputs, pose_rad_m: np.ndarray
) -> dict[str, Any]:
    import pyvista as pv

    contract = prepared.manifest["joint_pilot_contract"]
    sources = prepared.manifest["sources"]
    mandible = pv.read(sources["mandible_surface"]["path"]).triangulate()
    cranium = pv.read(sources["cranium_surface"]["path"]).triangulate()
    skin = pv.read(sources["template_skin"]["path"]).triangulate()
    pivot = prepared.arrays["mandible_pivot_m"]
    rotation = _rotation_matrix(pose_rad_m[:3])
    mandible.points = (
        (np.asarray(mandible.points) - pivot) @ rotation.T + pivot + pose_rad_m[3:]
    )
    bone_pairs, bone_midpoints = _collision_pairs(mandible, cranium)
    boxes = contract["posterior_joint_exclusion_region"]["world_aabbs_m"]
    in_posterior = np.zeros(len(bone_midpoints), dtype=bool)
    for bounds in boxes:
        lower, upper = np.asarray(bounds[0]), np.asarray(bounds[1])
        in_posterior |= np.all(
            (bone_midpoints >= lower) & (bone_midpoints <= upper), axis=1
        )
    names = [
        str(value) for value in np.asarray(skin.field_data["GroupName"]).reshape(-1)
    ]
    group = np.asarray(skin.cell_data["GroupId"], dtype=np.int32)
    lower_names = (
        "LipBottom",
        "LipInnerBottom",
        "LipOuterBottom",
        "MouthSocketBottom",
    )
    upper_names = ("LipTop", "LipInnerTop", "LipOuterTop", "MouthSocketTop")

    def skin_subset(selected: tuple[str, ...], *, invert: bool = False) -> Any:
        selected_ids = [names.index(name) for name in selected]
        mask = np.isin(group, selected_ids)
        if invert:
            mask = ~mask
        return (
            skin.extract_cells(np.flatnonzero(mask))
            .extract_surface(algorithm=None)
            .triangulate()
        )

    upper_pairs, _ = _collision_pairs(mandible, skin_subset(upper_names))
    lower_pairs, _ = _collision_pairs(mandible, skin_subset(lower_names))
    nonlower_pairs, _ = _collision_pairs(
        mandible, skin_subset(lower_names, invert=True)
    )
    return {
        "skull_mandible_contact_pairs": len(bone_pairs),
        "posterior_joint_contact_pairs": int(in_posterior.sum()),
        "contacts_outside_posterior_region": int((~in_posterior).sum()),
        "upper_oral_contact_pairs": len(upper_pairs),
        "lower_oral_contact_pairs": len(lower_pairs),
        "nonlower_skin_contact_pairs": len(nonlower_pairs),
    }


def _boundary_faces(volume: Any) -> np.ndarray:
    boundary = volume.extract_surface(algorithm=None).triangulate()
    original = np.asarray(boundary.point_data["vtkOriginalPointIds"], dtype=np.int64)
    return original[np.asarray(boundary.faces).reshape(-1, 4)[:, 1:]]


def _mapped_pairs(local: np.ndarray, first: Any, second: Any) -> np.ndarray:
    if not len(local):
        return np.empty((0, 2), dtype=np.int64)
    return np.column_stack(
        (
            np.asarray(first.cell_data["BoundaryCellId"], dtype=np.int64)[local[:, 0]],
            np.asarray(second.cell_data["BoundaryCellId"], dtype=np.int64)[local[:, 1]],
        )
    )


def _contact_topology(
    pairs: np.ndarray,
    segments: np.ndarray,
    boundary_faces: np.ndarray,
    points: np.ndarray,
) -> dict[str, Any]:
    """Classify collision segments against exact shared FEM topology."""
    coordinate_scale = max(1.0, float(np.max(np.abs(points))))
    segment_dtype = (
        segments.dtype
        if np.issubdtype(segments.dtype, np.floating)
        else np.dtype("<f8")
    )
    tolerance = max(1e-12, 32 * float(np.finfo(segment_dtype).eps) * coordinate_scale)
    shared_nodes = [
        np.intersect1d(boundary_faces[first], boundary_faces[second])
        for first, second in pairs
    ]
    shared_count = np.asarray([len(nodes) for nodes in shared_nodes], dtype=np.int8)
    confined = np.zeros(len(pairs), dtype=bool)
    for index in np.flatnonzero(shared_count == 1):
        vertex = points[shared_nodes[index][0]]
        confined[index] = bool(
            np.all(np.linalg.norm(segments[index] - vertex, axis=1) <= tolerance)
        )
    for index in np.flatnonzero(shared_count == 2):
        start, end = points[shared_nodes[index]]
        vector = end - start
        length = float(np.linalg.norm(vector))
        if length == 0.0:
            continue
        coordinate = ((segments[index] - start) @ vector) / (length**2)
        closest = start + np.clip(coordinate, 0.0, 1.0)[:, None] * vector
        distance = np.linalg.norm(segments[index] - closest, axis=1)
        coordinate_tolerance = tolerance / length
        confined[index] = bool(
            np.all(distance <= tolerance)
            and np.all(coordinate >= -coordinate_tolerance)
            and np.all(coordinate <= 1.0 + coordinate_tolerance)
        )
    nonadjacent = shared_count == 0
    adjacency_overrun = (shared_count > 0) & ~confined
    return {
        "coordinate_tolerance_m": tolerance,
        "shared_node_count": shared_count,
        "confined_to_shared_topology": confined,
        "nonadjacent": nonadjacent,
        "adjacency_overrun": adjacency_overrun,
        "summary": {
            "pairs": len(pairs),
            "nonadjacent_pairs": int(nonadjacent.sum()),
            "shared_vertex_pairs": int(np.count_nonzero(shared_count == 1)),
            "shared_edge_pairs": int(np.count_nonzero(shared_count == 2)),
            "other_shared_topology_pairs": int(np.count_nonzero(shared_count > 2)),
            "confined_adjacency_pairs": int(confined.sum()),
            "adjacency_overrun_pairs": int(adjacency_overrun.sum()),
        },
    }


def _mapped_contact_change(
    first: Any,
    second: Any,
    neutral_first: Any,
    neutral_second: Any,
    boundary_faces: np.ndarray,
    points: np.ndarray,
    neutral_points: np.ndarray,
) -> dict[str, Any]:
    """Compare raw contacts while excluding proven bonded-topology adjacency."""
    current, current_segments, current_midpoints, current_length = _collision_geometry(
        first, second
    )
    neutral, neutral_segments, _, neutral_length = _collision_geometry(
        neutral_first, neutral_second
    )
    current_pairs = _mapped_pairs(current, first, second)
    neutral_pairs = _mapped_pairs(neutral, neutral_first, neutral_second)
    current_topology = _contact_topology(
        current_pairs, current_segments, boundary_faces, points
    )
    neutral_topology = _contact_topology(
        neutral_pairs, neutral_segments, boundary_faces, neutral_points
    )
    baseline = {
        tuple(pair): float(length)
        for pair, length in zip(neutral_pairs.tolist(), neutral_length, strict=True)
    }
    new_mask = np.asarray(
        [tuple(pair) not in baseline for pair in current_pairs.tolist()], dtype=bool
    )
    new_pairs = current_pairs[new_mask].tolist()
    worsened = []
    for pair, length in zip(current_pairs.tolist(), current_length, strict=True):
        baseline_length = baseline.get(tuple(pair))
        if baseline_length is not None and length > baseline_length + 1e-8:
            worsened.append(
                {
                    "boundary_cell_pair": pair,
                    "baseline_segment_length_m": baseline_length,
                    "current_segment_length_m": float(length),
                }
            )
    nonadjacent = current_topology["nonadjacent"]
    adjacency_overrun = current_topology["adjacency_overrun"]
    penetrating = nonadjacent | adjacency_overrun
    new_penetrating = new_mask & penetrating
    return {
        "numerical_geometry_ok": bool(not np.any(penetrating)),
        "baseline_contact_pairs": len(neutral_pairs),
        "current_contact_pairs": len(current_pairs),
        "new_contact_pairs": len(new_pairs),
        "new_boundary_cell_pairs": new_pairs,
        "worsened_inherited_pairs": len(worsened),
        "worsened_pair_details": worsened,
        "baseline_intersection_segment_length_sum_m": float(neutral_length.sum()),
        "current_intersection_segment_length_sum_m": float(current_length.sum()),
        "intersection_segment_length_limit": "diagnostic proxy, not penetration depth or a contact law",
        "current_contact_midpoint_bounds_m": (
            [
                current_midpoints.min(axis=0).tolist(),
                current_midpoints.max(axis=0).tolist(),
            ]
            if len(current_midpoints)
            else None
        ),
        "baseline_topology": neutral_topology["summary"],
        "current_topology": current_topology["summary"],
        "new_nonadjacent_or_overrun_pairs": int(new_penetrating.sum()),
        "new_nonadjacent_or_overrun_boundary_cell_pairs": current_pairs[
            new_penetrating
        ].tolist(),
        "admission_semantics": "raw VTK pairs sharing a FEM vertex/edge are excluded only when the reported contact segment is confined to that shared topology; nonadjacent and adjacency-overrun pairs remain failures",
    }


def audit_deformed_oral_geometry(
    prepared: PreparedInputs,
    deformed_points: np.ndarray,
    pose_rad_m: np.ndarray,
    *,
    neutral: bool = False,
) -> dict[str, Any]:
    """Separate exact-FEM numerical geometry from source-anatomy validation."""
    import pyvista as pv

    deformed_points = np.asarray(deformed_points, dtype=np.float64)
    pose_rad_m = np.asarray(pose_rad_m, dtype=np.float64)
    volume = pv.read(prepared.volume_path)
    if deformed_points.shape != (volume.n_points, 3):
        raise ValueError(
            f"deformed point shape {deformed_points.shape} != {(volume.n_points, 3)}"
        )
    if pose_rad_m.shape != (6,) or np.any(~np.isfinite(pose_rad_m)):
        raise ValueError("pose_rad_m must be six finite world-frame values")
    if neutral and np.any(pose_rad_m != 0.0):
        raise ValueError("neutral oral audit requires exactly zero pose_rad_m")
    if np.any(~np.isfinite(deformed_points)):
        raise ValueError("deformed points must be finite")
    neutral_points = np.asarray(volume.points, dtype=np.float64)
    boundary_faces = _boundary_faces(volume)
    upper, lower, classifier = _fem_lip_surfaces(volume, deformed_points)
    neutral_upper, neutral_lower, _ = _fem_lip_surfaces(volume, neutral_points)
    fem_lip_change = _mapped_contact_change(
        upper,
        lower,
        neutral_upper,
        neutral_lower,
        boundary_faces,
        deformed_points,
        neutral_points,
    )
    fem_jaw, fem_upper_oral, fem_lower_oral, fem_jaw_classifier = (
        _fem_jaw_oral_surfaces(volume, deformed_points)
    )
    neutral_jaw, neutral_upper_oral, neutral_lower_oral, _ = _fem_jaw_oral_surfaces(
        volume, neutral_points
    )
    fem_jaw_upper_change = _mapped_contact_change(
        fem_jaw,
        fem_upper_oral,
        neutral_jaw,
        neutral_upper_oral,
        boundary_faces,
        deformed_points,
        neutral_points,
    )
    fem_jaw_lower_change = _mapped_contact_change(
        fem_jaw,
        fem_lower_oral,
        neutral_jaw,
        neutral_lower_oral,
        boundary_faces,
        deformed_points,
        neutral_points,
    )
    fem_cranium, fem_mandible, fem_soft, fem_contact_classifier = (
        _fem_bone_soft_surfaces(volume, deformed_points)
    )
    neutral_cranium, neutral_mandible, neutral_soft, _ = _fem_bone_soft_surfaces(
        volume, neutral_points
    )
    fem_soft_cranium_change = _mapped_contact_change(
        fem_soft,
        fem_cranium,
        neutral_soft,
        neutral_cranium,
        boundary_faces,
        deformed_points,
        neutral_points,
    )
    fem_soft_mandible_change = _mapped_contact_change(
        fem_soft,
        fem_mandible,
        neutral_soft,
        neutral_mandible,
        boundary_faces,
        deformed_points,
        neutral_points,
    )
    fem_mandible_cranium_change = _mapped_contact_change(
        fem_mandible,
        fem_cranium,
        neutral_mandible,
        neutral_cranium,
        boundary_faces,
        deformed_points,
        neutral_points,
    )
    source = _source_pose_audit(prepared, pose_rad_m)
    mandible_ids = prepared.arrays["mandible_node_ids"]
    pivot = prepared.arrays["mandible_pivot_m"]
    rotation = _rotation_matrix(pose_rad_m[:3])
    expected_mandible = (
        (np.asarray(volume.points)[mandible_ids] - pivot) @ rotation.T
        + pivot
        + pose_rad_m[3:]
    )
    support_error = np.linalg.norm(
        deformed_points[mandible_ids] - expected_mandible, axis=1
    )
    lower_bound = np.asarray(
        prepared.manifest["joint_pilot_contract"]["lower_pose_rad_m"]
    )
    upper_bound = np.asarray(
        prepared.manifest["joint_pilot_contract"]["upper_pose_rad_m"]
    )
    pose_in_candidate_box = bool(
        np.all(pose_rad_m >= lower_bound) and np.all(pose_rad_m <= upper_bound)
    )
    expression_candidate_ok = bool(
        pose_in_candidate_box
        and fem_lip_change["numerical_geometry_ok"]
        and fem_jaw_upper_change["numerical_geometry_ok"]
        and fem_jaw_lower_change["numerical_geometry_ok"]
        and fem_soft_cranium_change["numerical_geometry_ok"]
        and fem_soft_mandible_change["numerical_geometry_ok"]
        and fem_mandible_cranium_change["numerical_geometry_ok"]
        and support_error.max() <= 1e-8
    )
    neutral_pose_ok = bool(np.all(pose_rad_m == 0.0))
    neutral_invariants_ok = bool(
        neutral_pose_ok
        and fem_lip_change["numerical_geometry_ok"]
        and fem_jaw_upper_change["numerical_geometry_ok"]
        and fem_jaw_lower_change["numerical_geometry_ok"]
        and fem_soft_cranium_change["numerical_geometry_ok"]
        and fem_soft_mandible_change["numerical_geometry_ok"]
        and fem_mandible_cranium_change["numerical_geometry_ok"]
        and support_error.max() <= 1e-8
    )
    geometry_ok = neutral_invariants_ok if neutral else expression_candidate_ok
    full_gate = prepared.manifest["gates"]["full_six_dof_jaw"]
    return {
        "mode": "neutral" if neutral else "expression_candidate",
        "admissible": bool(full_gate == "pass" and geometry_ok),
        "numerical_geometry_admissible": geometry_ok,
        "anatomical_validation": False,
        "provisional_geometry_ok": expression_candidate_ok,
        "neutral_invariants_ok": neutral_invariants_ok,
        "neutral_pose_ok": neutral_pose_ok,
        "pose_in_candidate_box": pose_in_candidate_box,
        "fem_lip": {
            **classifier,
            **fem_lip_change,
        },
        "fem_mandible_oral": {
            **fem_jaw_classifier,
            "upper_oral": fem_jaw_upper_change,
            "lower_oral": fem_jaw_lower_change,
            "mapping": "exact fixture boundary-node IDs and deformed FEM points",
            "teeth_limit": "fixture has only a 2 mm IsTeeth proximity mask and no separate lower-teeth surface or verified rigid lower-teeth correspondence",
        },
        "fem_contact_surfaces": {
            **fem_contact_classifier,
            "soft_cranium": fem_soft_cranium_change,
            "soft_mandible": fem_soft_mandible_change,
            "mandible_cranium": fem_mandible_cranium_change,
            "mapping": "one connected FEM boundary with global fixture node IDs; pure soft and pure bone faces remain disjoint, mixed bone-soft transition faces are bonded and excluded from sliding contact",
        },
        "source_rigid_jaw": source,
        "source_geometry_role": "anatomical QA diagnostic only; source surfaces do not map one-to-one to the FEM boundary and do not determine numerical geometry admission",
        "mandible_pose_consistency": {
            "rms_error_m": float(np.sqrt(np.mean(support_error**2))),
            "max_error_m": float(support_error.max()),
            "tolerance_m": 1e-8,
            "passed": bool(support_error.max() <= 1e-8),
        },
        "gate": full_gate,
        "limitations": prepared.manifest["joint_pilot_contract"]["limitations"],
    }


def audit_neutral_oral_geometry(
    prepared: PreparedInputs, deformed_points: np.ndarray
) -> dict[str, Any]:
    """Check a zero-jaw neutral solve against frozen inherited contacts."""
    return audit_deformed_oral_geometry(
        prepared,
        deformed_points,
        np.zeros(6, dtype=np.float64),
        neutral=True,
    )
