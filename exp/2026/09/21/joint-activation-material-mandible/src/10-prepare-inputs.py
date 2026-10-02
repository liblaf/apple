# ruff: noqa: EM101, EM102, TRY003
"""Freeze and audit compact inputs for the joint face inverse experiment."""

from __future__ import annotations

import json
import logging
import os
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
import scipy.sparse as sp
from joint_common import ROOT, ProfileJoint, sha256, write_json
from joint_data import array_sha256
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkFiltersModeling import vtkCollisionDetectionFilter

from liblaf import cherries

logger = logging.getLogger(__name__)

EXPECTED = {
    "volume": "8131d6944b322d7c1e21918688f297e2887b42bf4dbc19ce36259b007e8dc563",
    "skin": "4c7ddce893eed4a8d0590042488ae1b35f0cae23383db6bc9814427eb6f7cc6f",
    "melon_volume": "824464f109a4e97c3176091bb21e8fdb533def6fc5616845bc36bb377c2a7752",
    "mandible_surface": "24c4dc7ad394ec2066024c3dc5b77f2235a2955e54de4fe2ae1741bb2bbcc246",
    "cranium_surface": "aec5a30e4c772f1fcd12577de9bfecd2633075022c6ee2a5cf285f2993182145",
}

TRAINING = ("Smile", "LipsFunnel", "BrowUpLeft", "MouthOpenSlightly")
RESERVED = ("SmileClosed", "JawLeft")


class Config(cherries.BaseConfig):
    volume: Path = cherries.input(
        ROOT
        / "exp/2026/06/17/human-face-smile-prestrain-v2/data/10-human-face-prepared.vtu"
    )
    skin: Path = cherries.input(
        ROOT
        / "exp/2026/08/18/human-face-smile-plane-stress-skin/data/10-corrected-baseline/skin-isface-e0200-p000.vtp"
    )
    melon_volume: Path = cherries.input(
        Path(os.environ["APPLE_MELON_HEAD"]) / "62-tetmesh-3191k.vtu"
    )
    mandible_surface: Path = cherries.input(
        Path(os.environ["APPLE_MELON_HEAD"]) / "13-mandible.ply"
    )
    cranium_surface: Path = cherries.input(
        Path(os.environ["APPLE_MELON_HEAD"]) / "13-cranium.ply"
    )
    mandible_landmarks: Path = cherries.input(
        Path(os.environ["APPLE_MELON_HEAD"]) / "11-mandible.landmarks.json"
    )
    template_skin: Path = cherries.input(
        Path(os.environ["APPLE_MELON_HEAD"]) / "22-skin.vtp"
    )
    output_npz: Path = cherries.output("prepared/inputs.npz", mkdir=True)
    output_manifest: Path = cherries.output("prepared/manifest.json", mkdir=True)


def require_identity(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    records = {}
    for name, configured_path in paths.items():
        path = configured_path.resolve()
        digest = sha256(path)
        expected = EXPECTED.get(name)
        if expected is not None and digest != expected:
            raise ValueError(
                f"{name} identity changed: expected {expected}, got {digest}"
            )
        records[name] = {
            "path": str(path),
            "sha256": digest,
            "bytes": path.stat().st_size,
        }
    return records


def tetrahedra(mesh: pv.UnstructuredGrid) -> np.ndarray:
    if set(np.unique(mesh.celltypes)) != {int(pv.CellType.TETRA)}:
        raise ValueError("fixture must contain tetrahedra only")
    result = np.asarray(mesh.cells_dict[pv.CellType.TETRA], dtype=np.int64)
    if result.shape != (mesh.n_cells, 4):
        raise ValueError("unexpected tetrahedral connectivity")
    return result


def build_graph(
    points: np.ndarray,
    tets: np.ndarray,
    active_ids: np.ndarray,
    muscle_ids: np.ndarray,
    fraction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    active_tets = tets[active_ids]
    pattern = np.asarray(((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)))
    faces = np.sort(active_tets[:, pattern].reshape(-1, 3), axis=1)
    owner = np.repeat(np.arange(len(active_ids), dtype=np.int32), 4)
    order = np.lexsort(faces.T[::-1])
    faces, owner = faces[order], owner[order]
    paired = np.flatnonzero(np.all(faces[1:] == faces[:-1], axis=1))
    if np.any(np.diff(paired) == 1):
        raise ValueError("non-manifold face in active tetrahedral graph")
    i, j = owner[paired], owner[paired + 1]
    same = muscle_ids[active_ids[i]] == muscle_ids[active_ids[j]]
    i, j, shared = i[same], j[same], faces[paired[same]]
    swap = i > j
    i[swap], j[swap] = j[swap], i[swap]
    xyz = points[shared]
    area = (
        np.linalg.norm(np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1)
        / 2
    )
    centers = points[active_tets].mean(axis=1)
    distance = np.linalg.norm(centers[i] - centers[j], axis=1)
    phi = fraction[active_ids]
    harmonic = 2 * phi[i] * phi[j] / (phi[i] + phi[j])
    conductance = area / distance * harmonic
    if (
        np.any(distance <= 0)
        or np.any(~np.isfinite(conductance))
        or np.any(conductance <= 0)
    ):
        raise ValueError("invalid activation graph conductance")
    adjacency = sp.coo_matrix(
        (np.ones(2 * len(i), dtype=np.int8), (np.r_[i, j], np.r_[j, i])),
        shape=(len(active_ids), len(active_ids)),
    ).tocsr()
    n_components, labels = connected_components(adjacency, directed=False)
    component_size = np.bincount(labels, minlength=n_components)
    stats = {
        "edges": len(i),
        "components": int(n_components),
        "singletons": int(np.count_nonzero(component_size == 1)),
        "component_size_min": int(component_size.min()),
        "component_size_max": int(component_size.max()),
        "conductance_m": distribution(conductance),
        "definition": "shared-face area / centroid distance times harmonic MuscleFraction; same MuscleId only",
    }
    return i.astype(np.int32), j.astype(np.int32), conductance.astype(np.float64), stats


def distribution(value: np.ndarray) -> dict[str, float]:
    value = np.asarray(value, dtype=np.float64)
    return {
        "min": float(value.min()),
        "q01": float(np.quantile(value, 0.01)),
        "median": float(np.median(value)),
        "q99": float(np.quantile(value, 0.99)),
        "max": float(value.max()),
        "mean": float(value.mean()),
        "sum": float(value.sum()),
    }


def support_mapping(
    volume: pv.UnstructuredGrid, melon: pv.UnstructuredGrid
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    names = [
        str(value) for value in np.asarray(volume.field_data["GroupName"]).reshape(-1)
    ]
    full_names = [
        str(value) for value in np.asarray(melon.field_data["GroupName"]).reshape(-1)
    ]
    if names != full_names:
        raise ValueError("fixture and Melon GroupName tables differ")
    fixture_group = np.asarray(volume.point_data["GroupId"], dtype=np.int32)
    full_group = np.asarray(melon.point_data["GroupId"], dtype=np.int32)
    original = np.asarray(volume.point_data["vtkOriginalPointIds"], dtype=np.int64)
    if (
        original.shape != (volume.n_points,)
        or np.unique(original).size != volume.n_points
    ):
        raise ValueError("fixture original point IDs are not one-to-one")
    if not np.array_equal(
        np.asarray(volume.points), np.asarray(melon.points)[original]
    ):
        raise ValueError(
            "fixture coordinates do not exactly map to Melon source points"
        )
    if not np.array_equal(fixture_group, full_group[original]):
        raise ValueError("fixture GroupId does not exactly map to Melon source")
    support = {}
    records = {}
    surface_ids = np.asarray(
        volume.extract_surface(algorithm=None).point_data["vtkOriginalPointIds"],
        dtype=np.int64,
    )
    is_surface = np.zeros(volume.n_points, dtype=bool)
    is_surface[surface_ids] = True
    for name in ("Mandible", "Cranium"):
        group_id = names.index(name)
        ids = np.flatnonzero(fixture_group == group_id).astype(np.int64)
        if not np.all(is_surface[ids]):
            raise ValueError(f"{name} support contains non-boundary nodes")
        full_ids = np.flatnonzero(full_group == group_id)
        retained = np.intersect1d(original, full_ids, assume_unique=False)
        if retained.size != ids.size:
            raise ValueError(f"{name} source-to-fixture mapping is incomplete")
        support[name.lower()] = ids
        records[name.lower()] = {
            "group_id": group_id,
            "fixture_nodes": int(ids.size),
            "full_melon_nodes": int(full_ids.size),
            "fixture_retains_all_full_nodes": bool(ids.size == full_ids.size),
            "all_fixture_nodes_on_boundary": True,
            "mapping": "fixture vtkOriginalPointIds -> exact Melon coordinate and GroupId",
        }
    if np.intersect1d(support["mandible"], support["cranium"]).size:
        raise ValueError("mandible and cranium support sets overlap")
    historical = np.flatnonzero(
        np.asarray(volume.point_data["IsFixed"], dtype=bool)
    ).astype(np.int64)
    union = np.union1d(support["mandible"], support["cranium"])
    outside = np.setdiff1d(historical, union, assume_unique=True)
    newly_classified = np.setdiff1d(union, historical, assume_unique=True)
    records["historical_fixed"] = {
        "nodes": int(historical.size),
        "outside_recovered_support": int(outside.size),
        "support_nodes_not_historically_fixed": int(newly_classified.size),
        "reason": "historical IsFixed removed 2 mm teeth/gingiva/lip proximity masks; recovered support uses GroupId",
    }
    return {
        **support,
        "historical": historical,
        "outside": outside,
        "newly_classified": newly_classified,
    }, records


def jaw_frame(
    landmark_path: Path, support_points: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    landmarks = np.asarray(
        [
            [row[axis] for axis in ("x", "y", "z")]
            for row in json.loads(landmark_path.read_text())
        ],
        dtype=np.float64,
    )
    if landmarks.shape != (16, 3):
        raise ValueError("unexpected registered mandible landmark table")
    # Pair 1/9 is bilateral, lies in the two posterior skull-mandible contact
    # clusters, and supplies a reproducible hinge-axis candidate. The source has
    # no semantic landmark names, so the manifest deliberately limits the claim.
    left, right = landmarks[[1, 9]]
    lateral = right - left
    lateral /= np.linalg.norm(lateral)
    vertical = np.asarray((0.0, 1.0, 0.0))
    vertical -= lateral * np.dot(vertical, lateral)
    vertical /= np.linalg.norm(vertical)
    anterior = np.cross(lateral, vertical)
    if anterior[2] < 0:
        vertical = -vertical
        anterior = -anterior
    frame = np.column_stack((lateral, vertical, anterior))
    if (
        not np.allclose(frame.T @ frame, np.eye(3), atol=1e-12)
        or np.linalg.det(frame) < 0.999999
    ):
        raise ValueError("mandible reference frame is not right-handed orthonormal")
    pivot = (left + right) / 2
    return (
        pivot,
        frame,
        {
            "pivot_definition": "midpoint of registered mandible landmarks 1 and 9",
            "frame_columns": "bilateral landmark axis; projected world +Y; right-handed anterior",
            "semantic_limit": "landmarks are unnamed; posterior contact localization supports a hinge interpretation but does not prove TMJ anatomy",
            "support_centroid_m": support_points.mean(axis=0).tolist(),
            "landmark_pair_m": [left.tolist(), right.tolist()],
        },
    )


def rotate(
    points: np.ndarray, pivot: np.ndarray, axis: np.ndarray, degrees: float
) -> np.ndarray:
    theta = np.deg2rad(degrees)
    x, y, z = axis
    skew = np.asarray(((0.0, -z, y), (z, 0.0, -x), (-y, x, 0.0)))
    rotation = np.eye(3) + np.sin(theta) * skew + (1 - np.cos(theta)) * (skew @ skew)
    return (points - pivot) @ rotation.T + pivot


def collision_count(first: pv.PolyData, second: pv.PolyData) -> tuple[int, np.ndarray]:
    output, count = first.collision(
        second, contact_mode=0, box_tolerance=0.0, cell_tolerance=0.0
    )
    ids = np.unique(
        np.asarray(output.field_data.get("ContactCells", []), dtype=np.int64)
    )
    return int(count), ids


def collision_details(
    first: pv.PolyData, second: pv.PolyData
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return paired cell IDs, contact midpoints, and segment lengths."""
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
        segments.mean(axis=1),
        np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1),
    )


def collision_pairs(
    first: pv.PolyData, second: pv.PolyData
) -> tuple[np.ndarray, np.ndarray]:
    pairs, midpoints, _ = collision_details(first, second)
    return pairs, midpoints


def apply_pose(
    points: np.ndarray, pivot: np.ndarray, pose_rad_m: np.ndarray
) -> np.ndarray:
    rotation = Rotation.from_rotvec(pose_rad_m[:3]).as_matrix()
    return (points - pivot) @ rotation.T + pivot + pose_rad_m[3:]


def oral_audit(  # noqa: PLR0915
    mandible: pv.PolyData,
    cranium: pv.PolyData,
    template_skin: pv.PolyData,
    volume: pv.UnstructuredGrid,
    pivot: np.ndarray,
    frame: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    mandible = mandible.triangulate()
    cranium = cranium.triangulate()
    template_skin = template_skin.triangulate()
    template_skin.cell_data["SourceCellId"] = np.arange(
        template_skin.n_cells, dtype=np.int64
    )
    skin_names = [
        str(value)
        for value in np.asarray(template_skin.field_data["GroupName"]).reshape(-1)
    ]
    skin_group = np.asarray(template_skin.cell_data["GroupId"], dtype=np.int32)

    def skin_part(groups: tuple[str, ...]) -> pv.PolyData:
        group_ids = [skin_names.index(name) for name in groups]
        ids = np.flatnonzero(np.isin(skin_group, group_ids))
        return (
            template_skin.extract_cells(ids)
            .extract_surface(algorithm=None)
            .triangulate()
        )

    upper_oral_names = (
        "LipTop",
        "LipInnerTop",
        "LipOuterTop",
        "MouthSocketTop",
    )
    lower_oral_names = (
        "LipBottom",
        "LipInnerBottom",
        "LipOuterBottom",
        "MouthSocketBottom",
    )
    upper_oral, lower_oral = skin_part(upper_oral_names), skin_part(lower_oral_names)
    lower_oral_ids = [skin_names.index(name) for name in lower_oral_names]
    nonlower = (
        template_skin.extract_cells(
            np.flatnonzero(~np.isin(skin_group, lower_oral_ids))
        )
        .extract_surface(algorithm=None)
        .triangulate()
    )
    rest_pairs, rest_midpoints = collision_pairs(mandible, cranium)
    rest_posterior = rest_midpoints[:, 2] < 0.02
    hinge_samples = []
    for degrees in (0.0, 0.25, 0.5, 1.0, 2.0, 3.0):
        moved = mandible.copy(deep=True)
        moved.points = rotate(np.asarray(moved.points), pivot, frame[:, 0], degrees)
        skull_pairs, skull_midpoints = collision_pairs(moved, cranium)
        upper_count, _ = collision_count(moved, upper_oral)
        lower_count, _ = collision_count(moved, lower_oral)
        all_skin_count, _ = collision_count(moved, template_skin)
        hinge_samples.append(
            {
                "hinge_degrees": degrees,
                "skull_contact_pairs": len(skull_pairs),
                "skull_contact_cells": int(np.unique(skull_pairs[:, 0]).size),
                "posterior_contact_cells": int(
                    np.unique(skull_pairs[skull_midpoints[:, 2] < 0.02, 0]).size
                ),
                "anterior_contact_cells": int(
                    np.unique(skull_pairs[skull_midpoints[:, 2] > 0.05, 0]).size
                ),
                "upper_oral_contact_pairs": upper_count,
                "lower_oral_contact_pairs": lower_count,
                "all_skin_contact_pairs": all_skin_count,
            }
        )
    faces = np.asarray(template_skin.faces).reshape(-1, 4)[:, 1:]

    def group_points(name: str) -> np.ndarray:
        return np.unique(faces[skin_group == skin_names.index(name)].reshape(-1))

    upper_lip_names = ("LipTop", "LipInnerTop", "LipOuterTop")
    lower_lip_names = ("LipBottom", "LipInnerBottom", "LipOuterBottom")
    upper_lip_cells = np.flatnonzero(
        np.isin(skin_group, [skin_names.index(name) for name in upper_lip_names])
    )
    lower_lip_cells = np.flatnonzero(
        np.isin(skin_group, [skin_names.index(name) for name in lower_lip_names])
    )
    shared_lip_points = np.intersect1d(
        np.unique(faces[upper_lip_cells]), np.unique(faces[lower_lip_cells])
    )
    upper_lip_cells = upper_lip_cells[
        ~np.any(np.isin(faces[upper_lip_cells], shared_lip_points), axis=1)
    ]
    lower_lip_cells = lower_lip_cells[
        ~np.any(np.isin(faces[lower_lip_cells], shared_lip_points), axis=1)
    ]
    upper_lip = (
        template_skin.extract_cells(upper_lip_cells)
        .extract_surface(algorithm=None)
        .triangulate()
    )
    lower_lip = (
        template_skin.extract_cells(lower_lip_cells)
        .extract_surface(algorithm=None)
        .triangulate()
    )
    source_lip_pairs, source_lip_midpoints, source_lip_lengths = collision_details(
        upper_lip, lower_lip
    )
    source_lip_pairs = np.column_stack(
        (
            np.asarray(upper_lip.cell_data["SourceCellId"], dtype=np.int64)[
                source_lip_pairs[:, 0]
            ],
            np.asarray(lower_lip.cell_data["SourceCellId"], dtype=np.int64)[
                source_lip_pairs[:, 1]
            ],
        )
    )

    def mapping_record(part: pv.PolyData) -> dict[str, Any]:
        distance, nearest = cKDTree(np.asarray(volume.points)).query(
            np.asarray(part.points), k=1
        )
        return {
            "vertices": int(part.n_points),
            "exact_vertices_le_1e-12m": int(np.count_nonzero(distance <= 1e-12)),
            "exact_fraction": float(np.mean(distance <= 1e-12)),
            "unique_nearest_fem_nodes": int(np.unique(nearest).size),
            "nearest_distance_m": distribution(distance),
        }

    boundary = volume.extract_surface(algorithm=None).triangulate()
    boundary.cell_data["BoundaryCellId"] = np.arange(boundary.n_cells, dtype=np.int64)
    original = np.asarray(boundary.point_data["vtkOriginalPointIds"], dtype=np.int64)
    point_group = np.asarray(volume.point_data["GroupId"], dtype=np.int32)[original]
    volume_names = [
        str(value) for value in np.asarray(volume.field_data["GroupName"]).reshape(-1)
    ]
    boundary_faces = np.asarray(boundary.faces).reshape(-1, 4)[:, 1:]
    upper_ids = [volume_names.index(name) for name in upper_lip_names]
    lower_ids = [volume_names.index(name) for name in lower_lip_names]
    upper_mask = np.any(np.isin(point_group[boundary_faces], upper_ids), axis=1)
    lower_mask = np.any(np.isin(point_group[boundary_faces], lower_ids), axis=1)
    ambiguous = upper_mask & lower_mask
    upper_mask &= ~ambiguous
    lower_mask &= ~ambiguous
    fem_upper = (
        boundary.extract_cells(np.flatnonzero(upper_mask))
        .extract_surface(algorithm=None)
        .triangulate()
    )
    fem_lower = (
        boundary.extract_cells(np.flatnonzero(lower_mask))
        .extract_surface(algorithm=None)
        .triangulate()
    )
    fem_lip_pairs, fem_lip_midpoints, fem_lip_lengths = collision_details(
        fem_upper, fem_lower
    )
    fem_lip_pairs = np.column_stack(
        (
            np.asarray(fem_upper.cell_data["BoundaryCellId"], dtype=np.int64)[
                fem_lip_pairs[:, 0]
            ],
            np.asarray(fem_lower.cell_data["BoundaryCellId"], dtype=np.int64)[
                fem_lip_pairs[:, 1]
            ],
        )
    )

    initial_pose = np.r_[frame[:, 0] * np.deg2rad(1.0), np.zeros(3)]
    pose_delta = np.r_[np.full(3, np.deg2rad(0.05)), np.full(3, 0.00002)]
    lower_pose, upper_pose = initial_pose - pose_delta, initial_pose + pose_delta
    corner_records = []
    corner_posterior_midpoints = []
    for bits in product((0, 1), repeat=6):
        pose = np.where(bits, upper_pose, lower_pose)
        moved = mandible.copy(deep=True)
        moved.points = apply_pose(np.asarray(mandible.points), pivot, pose)
        bone_pairs, bone_midpoints = collision_pairs(moved, cranium)
        posterior = bone_midpoints[:, 2] < 0.02
        upper_pairs, _ = collision_pairs(moved, upper_oral)
        lower_pairs, _ = collision_pairs(moved, lower_oral)
        nonlower_pairs, _ = collision_pairs(moved, nonlower)
        corner_posterior_midpoints.append(bone_midpoints[posterior])
        corner_records.append(
            {
                "skull_mandible_contact_pairs": len(bone_pairs),
                "posterior_joint_contact_pairs": int(posterior.sum()),
                "contacts_outside_posterior_region": int((~posterior).sum()),
                "upper_oral_contact_pairs": len(upper_pairs),
                "lower_oral_contact_pairs": len(lower_pairs),
                "nonlower_skin_contact_pairs": len(nonlower_pairs),
            }
        )
    corner_midpoints = np.concatenate(corner_posterior_midpoints)
    posterior_boxes = [
        [
            corner_midpoints[side].min(axis=0).tolist(),
            corner_midpoints[side].max(axis=0).tolist(),
        ]
        for side in (
            corner_midpoints[:, 0] < pivot[0],
            corner_midpoints[:, 0] >= pivot[0],
        )
    ]
    candidate_valid = all(
        record["contacts_outside_posterior_region"] == 0
        and record["upper_oral_contact_pairs"] == 0
        and record["nonlower_skin_contact_pairs"] == 0
        for record in corner_records
    )
    limits = {
        name: {
            "min": min(record[name] for record in corner_records),
            "max": max(record[name] for record in corner_records),
        }
        for name in corner_records[0]
    }
    oral = {
        "rest_skull_mandible": {
            "contact_pairs": len(rest_pairs),
            "unique_mandible_cells": int(np.unique(rest_pairs[:, 0]).size),
            "contact_cell_centroid_bounds_m": [
                rest_midpoints.min(axis=0).tolist(),
                rest_midpoints.max(axis=0).tolist(),
            ],
            "posterior_contact_pairs_z_lt_0p02m": int(rest_posterior.sum()),
            "posterior_cell_pairs": rest_pairs[rest_posterior].tolist(),
            "anterior_contact_pairs_z_gt_0p05m": int(
                np.count_nonzero(rest_midpoints[:, 2] > 0.05)
            ),
            "localization": "five clusters: bilateral posterior clusters near pivot landmarks and three anterior/dental-region clusters",
        },
        "template_skin": {
            "points": int(template_skin.n_points),
            "triangles": int(template_skin.n_cells),
            "open_edges": int(template_skin.n_open_edges),
            "is_manifold": bool(template_skin.is_manifold),
            "shared_vertices": {
                "LipInnerTop_LipInnerBottom": int(
                    np.intersect1d(
                        group_points("LipInnerTop"), group_points("LipInnerBottom")
                    ).size
                ),
                "LipTop_LipBottom": int(
                    np.intersect1d(
                        group_points("LipTop"), group_points("LipBottom")
                    ).size
                ),
                "MouthSocketTop_MouthSocketBottom": int(
                    np.intersect1d(
                        group_points("MouthSocketTop"),
                        group_points("MouthSocketBottom"),
                    ).size
                ),
            },
            "interpretation": "QA flags requiring localization; not proof of pathological fusion",
        },
        "inherited_lip_intersections": {
            "source_contact_pairs_after_shared_seam_removal": len(source_lip_pairs),
            "source_upper_lower_cell_pairs": source_lip_pairs.tolist(),
            "source_contact_midpoints_m": source_lip_midpoints.tolist(),
            "source_contact_segment_lengths_m": source_lip_lengths.tolist(),
            "source_contact_segment_length_sum_m": float(source_lip_lengths.sum()),
            "source_contact_midpoint_bounds_m": [
                source_lip_midpoints.min(axis=0).tolist(),
                source_lip_midpoints.max(axis=0).tolist(),
            ],
            "shared_seam_vertices": int(shared_lip_points.size),
            "source_to_fem_nearest_mapping": {
                "upper": mapping_record(upper_lip),
                "lower": mapping_record(lower_lip),
                "confidence": "insufficient for inherited-pair tracking on deformed FEM: mapping is non-bijective and mostly not exact",
            },
            "exact_fem_node_classifier": {
                "definition": "fixture boundary triangle has any upper/lower lip GroupId vertex; 133 mixed upper/lower triangles omitted",
                "upper_triangles": int(upper_mask.sum()),
                "lower_triangles": int(lower_mask.sum()),
                "ambiguous_triangles_omitted": int(ambiguous.sum()),
                "neutral_contact_pairs": len(fem_lip_pairs),
                "neutral_boundary_cell_pairs": fem_lip_pairs.tolist(),
                "neutral_contact_midpoints_m": fem_lip_midpoints.tolist(),
                "neutral_contact_segment_lengths_m": fem_lip_lengths.tolist(),
                "neutral_contact_segment_length_sum_m": float(fem_lip_lengths.sum()),
                "limit": "exact FEM-node mapping but conservative labels do not reproduce the source-template contact set",
            },
        },
        "hinge_screen": {
            "samples": hinge_samples,
            "restricted_pilot_envelope": {
                "rotation_local_x_degrees": [0.0, 2.0],
                "rotation_local_yz_degrees": [0.0, 0.0],
                "translation_local_m": [0.0, 0.0, 0.0],
                "status": "geometry-screened one-DOF pilot only",
                "evidence": "0-2 degree positive hinge samples add no upper-oral or non-lower-skin intersection; anterior skull contacts vanish by 0.5 degree; posterior and lower-oral pre-existing contacts remain",
                "exclusions": "posterior skull-mandible and lower-oral intersections are pre-existing geometry, not validated contact exclusions",
                "full_six_dof_gate": "blocked pending contact/exclusion semantics and multidirectional swept-volume checks",
            },
        },
    }
    contract = {
        "status": "candidate box screened, anatomy gate blocked",
        "pose_convention": "six world-frame values: rotation vector xyz in radians, then translation xyz in metres; rotation about mandible_pivot_m",
        "initial_pose_rad_m": initial_pose.tolist(),
        "lower_pose_rad_m": lower_pose.tolist(),
        "upper_pose_rad_m": upper_pose.tolist(),
        "screened_corner_count": len(corner_records),
        "corner_screen_passed": candidate_valid,
        "screened_limits": limits,
        "posterior_joint_exclusion_region": {
            "world_aabbs_m": posterior_boxes,
            "definition": "union of bilateral posterior contact midpoint bounds over all 64 candidate-box corners",
            "anatomical_status": "provisional TMJ inference from bilateral posterior location and unnamed landmark pair; not a validated contact law",
        },
        "runtime_geometry_checks": [
            "reject pose outside lower_pose_rad_m/upper_pose_rad_m",
            "reject solved mandible support nodes more than 1e-8 m from the rigid transform defined by pose_rad_m",
            "at every proposed and solved pose, reject source mandible-cranium contact midpoint outside either posterior AABB",
            "reject any source mandible intersection with upper oral or any non-lower template-skin triangle",
            "reject lower-oral source contact count above the screened corner maximum and record the count",
            "on deformed FEM nodes, reject every upper/lower boundary-cell pair absent from the neutral conservative classifier",
            "on deformed FEM nodes, reject inherited lip or mandible-oral pairs whose intersection-segment length grows by more than 1e-8 m",
        ],
        "limitations": [
            "the source template has 17 inherited lip intersections even after all triangles incident to 58 shared seam vertices are removed",
            "source lip vertices do not map bijectively or mostly exactly to FEM nodes, so source contact pairs cannot be tracked through FEM deformation",
            "the exact-node FEM classifier omits 133 ambiguous mixed-label triangles and does not reproduce the source-template contact set",
            "posterior skull-mandible and lower-oral contacts have no validated exclusion semantics or contact law",
            "the fixture has no separate lower-teeth surface or verified source-teeth to rigid-mandible/FEM correspondence; tooth contact cannot be certified",
            "corner screening does not certify every interior six-DoF pose; the runtime audit must check each proposal",
        ],
        "admission": "never pass the full six-DoF anatomy gate from this candidate contract",
    }
    return oral, contract


def observations(
    volume: pv.UnstructuredGrid, skin: pv.PolyData
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    names = TRAINING + RESERVED
    ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    if ids.shape != (skin.n_points,) or np.unique(ids).size != skin.n_points:
        raise ValueError("skin GlobalPointId is not one-to-one")
    if not np.array_equal(np.asarray(skin.points), np.asarray(volume.points)[ids]):
        raise ValueError("skin points do not exactly map to volume")
    triangles = np.asarray(skin.faces).reshape(-1, 4)[:, 1:]
    xyz = np.asarray(skin.points)[triangles]
    area = (
        np.linalg.norm(np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1)
        / 2
    )
    point_area = np.zeros(skin.n_points, dtype=np.float64)
    np.add.at(point_area, triangles.reshape(-1), np.repeat(area / 3, 3))
    targets = np.stack(
        [np.asarray(volume.point_data[name], dtype=np.float64)[ids] for name in names]
    )
    valid = np.isfinite(targets).all(axis=(0, 2)) & (point_area > 0)
    if not np.all(valid):
        ids, point_area, targets = ids[valid], point_area[valid], targets[:, valid]
    normalized = point_area / point_area.sum()
    # Stable, spatially interleaved split based only on frozen source point IDs.
    key = ids.astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15)
    key = (key ^ (key >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    key = (key ^ (key >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    key ^= key >> np.uint64(31)
    adapt = (key & np.uint64(1)) == 0
    score = ~adapt
    if not adapt.any() or not score.any():
        raise ValueError("reserved observation split is empty")
    qa = []
    for idx, name in enumerate(names):
        norm = np.linalg.norm(targets[idx], axis=1)
        qa.append(
            {
                "name": name,
                "role": "training" if name in TRAINING else "reserved",
                "valid_observations": len(ids),
                "area_weighted_rms_m": float(np.sqrt(np.sum(normalized * norm**2))),
                "max_displacement_m": float(norm.max()),
                "jaw_pose_evidence": "none; transferred skin displacement contains no measured rigid bone transform",
            }
        )
    return {
        "observation_node_ids": ids,
        "observation_area_weights_m2": point_area,
        "observation_weight_normalized": normalized,
        "target_displacement_m": targets,
        "training_target_indices": np.arange(len(TRAINING), dtype=np.int32),
        "reserved_target_indices": np.arange(len(TRAINING), len(names), dtype=np.int32),
        "reserved_adapt_mask": adapt.astype(np.bool_),
        "reserved_score_mask": score.astype(np.bool_),
    }, {
        "names": list(names),
        "training": list(TRAINING),
        "reserved": list(RESERVED),
        "common_valid_observations": len(ids),
        "surface_area_m2": float(point_area.sum()),
        "area_weighting": "one third of each frozen neutral skin triangle area per incident vertex",
        "reserved_split": {
            "definition": "SplitMix64 parity of fixture-local observation node ID",
            "adapt_count": int(adapt.sum()),
            "score_count": int(score.sum()),
            "disjoint": True,
        },
        "targets": qa,
        "selection_rationale": "smile, lip pucker, asymmetric brow, and modest jaw-sensitive training motions; closed-smile and lateral-jaw reserves",
        "interpretation": "all are transferred Faceform skin fields, not independent measurements",
    }


def manifest_arrays(arrays: dict[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "shape": list(value.shape),
            "dtype": value.dtype.str,
            "sha256": array_sha256(value),
        }
        for name, value in sorted(arrays.items())
    }


def main(cfg: Config) -> None:
    paths = {
        "volume": cfg.volume,
        "skin": cfg.skin,
        "melon_volume": cfg.melon_volume,
        "mandible_surface": cfg.mandible_surface,
        "cranium_surface": cfg.cranium_surface,
        "mandible_landmarks": cfg.mandible_landmarks,
        "template_skin": cfg.template_skin,
    }
    sources = require_identity(paths)
    volume = pv.read(cfg.volume)
    skin = pv.read(cfg.skin)
    melon = pv.read(cfg.melon_volume)
    mandible_surface = pv.read(cfg.mandible_surface)
    cranium_surface = pv.read(cfg.cranium_surface)
    template_skin = pv.read(cfg.template_skin)
    if not isinstance(volume, pv.UnstructuredGrid) or not isinstance(
        melon, pv.UnstructuredGrid
    ):
        raise TypeError("volume inputs must be unstructured grids")
    if (volume.n_points, volume.n_cells) != (228_660, 1_146_517):
        raise ValueError("historical fixture topology changed")
    points = np.asarray(volume.points, dtype=np.float64)
    tets = tetrahedra(volume)
    muscle_fraction = np.asarray(volume.cell_data["MuscleFraction"], dtype=np.float64)
    muscle_id = np.asarray(volume.cell_data["MuscleId"], dtype=np.int32)
    active_mask = np.asarray(volume.cell_data["ActivationMask"], dtype=bool)
    if not np.array_equal(active_mask, muscle_fraction > 1e-6):
        raise ValueError("historical activation mask predicate changed")
    active_ids = np.flatnonzero(active_mask).astype(np.int64)
    if active_ids.size != 288_235 or np.unique(muscle_id[active_ids]).size != 103:
        raise ValueError("historical full-active domain changed")
    if np.any(muscle_id[active_ids] < 0):
        raise ValueError("active cell lacks a dominant MuscleId")
    volume_m3 = np.asarray(volume.cell_data["Volume"], dtype=np.float64)
    graph_i, graph_j, conductance, graph_stats = build_graph(
        points, tets, active_ids, muscle_id, muscle_fraction
    )
    supports, support_stats = support_mapping(volume, melon)
    pivot, frame, frame_stats = jaw_frame(
        cfg.mandible_landmarks, points[supports["mandible"]]
    )
    observation_arrays, cohort = observations(volume, skin)
    arrays = {
        "active_cell_ids": active_ids,
        "active_muscle_ids": muscle_id[active_ids].astype(np.int32),
        "active_muscle_fraction": muscle_fraction[active_ids].astype(np.float64),
        "active_effective_volume_m3": (
            volume_m3[active_ids] * muscle_fraction[active_ids]
        ).astype(np.float64),
        "graph_i": graph_i,
        "graph_j": graph_j,
        "graph_conductance_m": conductance,
        "mandible_node_ids": supports["mandible"],
        "cranium_node_ids": supports["cranium"],
        "historical_fixed_node_ids": supports["historical"],
        "historical_fixed_outside_support_node_ids": supports["outside"],
        "support_not_historical_fixed_node_ids": supports["newly_classified"],
        "mandible_pivot_m": pivot,
        "mandible_frame_world": frame,
        **observation_arrays,
    }
    cfg.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cfg.output_npz, **arrays)
    artifact_hash = sha256(cfg.output_npz)
    oral, pilot_contract = oral_audit(
        mandible_surface,
        cranium_surface,
        template_skin,
        volume,
        pivot,
        frame,
    )
    manifest = {
        "schema_version": 1,
        "purpose": "compact frozen inputs for joint activation/material/mandible inverse physics",
        "sources": sources,
        "fixture": {
            "volume_path": str(cfg.volume.resolve()),
            "skin_path": str(cfg.skin.resolve()),
            "points": int(volume.n_points),
            "tetrahedra": int(volume.n_cells),
            "active_predicate": "source ActivationMask exactly equals MuscleFraction > 1e-6",
            "active_cells": int(active_ids.size),
            "active_muscle_labels": int(np.unique(muscle_id[active_ids]).size),
            "historical_full_active": True,
            "later_named_face_subset_used": False,
        },
        "cohort": cohort,
        "support": {**support_stats, "reference_frame": frame_stats},
        "oral_contact_qa": oral,
        "joint_pilot_contract": pilot_contract,
        "activation_graph": graph_stats,
        "arrays": manifest_arrays(arrays),
        "artifact": {
            "path": str(cfg.output_npz.resolve()),
            "sha256": artifact_hash,
            "bytes": cfg.output_npz.stat().st_size,
        },
        "gates": {
            "input_identity": "pass",
            "full_active_domain": "pass",
            "support_mapping": "pass",
            "target_observations": "pass",
            "activation_graph": "pass",
            "full_six_dof_jaw": "blocked by unresolved pre-existing contacts and missing contact/exclusion semantics",
            "restricted_six_dof_candidate": "64 corners around a 1 degree opening pass source rigid-geometry screens; anatomy gate remains blocked",
        },
    }
    write_json(cfg.output_manifest, manifest)
    cherries.log_metrics(
        {
            "active/cells": int(active_ids.size),
            "active/graph_edges": len(graph_i),
            "active/graph_components": graph_stats["components"],
            "support/mandible_nodes": len(supports["mandible"]),
            "support/cranium_nodes": len(supports["cranium"]),
            "target/observations": len(arrays["observation_node_ids"]),
            "oral/rest_skull_mandible_contacts": oral["rest_skull_mandible"][
                "contact_pairs"
            ],
            "oral/source_lip_contact_pairs": oral["inherited_lip_intersections"][
                "source_contact_pairs_after_shared_seam_removal"
            ],
            "oral/candidate_corners": pilot_contract["screened_corner_count"],
        }
    )
    logger.info("Wrote %s (%d bytes)", cfg.output_npz, cfg.output_npz.stat().st_size)
    logger.info("Wrote %s", cfg.output_manifest)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
