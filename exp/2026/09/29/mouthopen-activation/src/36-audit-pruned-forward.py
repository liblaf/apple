# Copyright (c) 2026 liblaf
"""Audit the lowest-J cells at a terminal pruned MouthOpen forward checkpoint."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]
ROOT = GROUP.parents[4]
sys.path.insert(0, str(ROOT / "exp/2026/09/21/stress-activation-loss/src"))

from experiment import Profile  # noqa: E402

FACE_PATTERN = np.asarray(((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)))


class Config(cherries.BaseConfig):
    source: Path = Path("35-forward-pruned-002")
    output: Path = Path("36-pruned-audit")
    lowest_count: int = 20


def receipt(path: Path) -> dict[str, str | int]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": str(path.resolve()),
        "sha256": digest.hexdigest(),
        "bytes": path.stat().st_size,
    }


def main(cfg: Config) -> None:  # noqa: PLR0915
    assert cfg.lowest_count > 0
    source = GROUP / "data" / cfg.source
    checkpoint = cherries.input(source / "final.npz")
    summary_path = cherries.input(source / "summary.json")
    volume_path = cherries.input(GROUP / "data/30-pruned-fixture/volume.vtu")
    skin_path = cherries.input(GROUP / "data/30-pruned-fixture/skin.vtp")
    mapping_path = cherries.input(GROUP / "data/30-pruned-fixture/mapping.npz")
    prepared_path = cherries.input(GROUP / "data/10-mandible/prepared.npz")
    harmonic_path = cherries.input(source / "harmonic-weight.npz")
    output_path = cherries.output(cfg.output / "audit.json", mkdir=True)
    assert not output_path.exists(), output_path
    summary = json.loads(summary_path.read_text())
    assert summary["status"] == "blocked_at_minimum_pose_step"
    assert summary["final_checkpoint"]["sha256"] == receipt(checkpoint)["sha256"]
    with np.load(checkpoint, allow_pickle=False) as data:
        displacement = np.asarray(data["displacement"], dtype=np.float64)
        fraction = float(data["fraction"])
        pose = np.asarray(data["pose"], dtype=np.float64)
    assert fraction == summary["completed_pose_fraction"]
    mesh = pv.read(volume_path)
    skin = pv.read(skin_path)
    assert isinstance(mesh, pv.UnstructuredGrid)
    assert isinstance(skin, pv.PolyData)
    rest = np.asarray(mesh.points, dtype=np.float64)
    cells = np.asarray(mesh.cells).reshape(-1, 5)[:, 1:].astype(np.int64)
    assert displacement.shape == rest.shape
    with np.load(mapping_path, allow_pickle=False) as mapping:
        old_points = np.asarray(mapping["new_to_original_point"], dtype=np.int64)
        old_cells = np.asarray(mapping["new_to_original_cell"], dtype=np.int64)
        removed_cells = np.asarray(mapping["removed_original_cell_ids"], dtype=np.int64)
        original_to_new = np.asarray(mapping["original_to_new_point"], dtype=np.int64)
    with np.load(prepared_path, allow_pickle=False) as prepared:
        original_tets = np.asarray(prepared["tets"], dtype=np.int64)
        original_rest = np.asarray(prepared["X"], dtype=np.float64)
        original_skin_ids = np.asarray(prepared["skin_ids"], dtype=np.int64)
        triangles = np.asarray(prepared["triangles"], dtype=np.int64)
        target_skin = np.asarray(prepared["target_skin"], dtype=np.float64)
        patch_ids = np.asarray(prepared["patch_ids"], dtype=np.int64)
        pivot = np.asarray(prepared["pivot"], dtype=np.float64)
        full_pose = np.asarray(prepared["pose"], dtype=np.float64)
    with np.load(harmonic_path, allow_pickle=False) as data:
        harmonic_weight = np.asarray(data["weight"], dtype=np.float64)
    np.testing.assert_array_equal(rest, original_rest[old_points])
    np.testing.assert_array_equal(old_points[cells], original_tets[old_cells])
    fixed = np.asarray(mesh.point_data["IsFixed"], dtype=bool)
    group = np.asarray(mesh.point_data["GroupId"], dtype=int)
    group_names = [str(value) for value in mesh.field_data["GroupName"]]
    jaw = fixed & (group == group_names.index("Mandible"))
    assert not fixed[cells].all(axis=1).any()
    deformed = rest + displacement
    skin_ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    np.testing.assert_array_equal(skin_ids, original_to_new[original_skin_ids])
    np.testing.assert_array_equal(
        triangles, np.asarray(skin.faces).reshape(-1, 4)[:, 1:]
    )
    np.testing.assert_array_equal(skin.points, rest[skin_ids])
    skin_rest = rest[skin_ids]
    skin_fit = deformed[skin_ids]
    rest_tri = skin_rest[triangles]
    rest_cross = np.cross(
        rest_tri[:, 1] - rest_tri[:, 0], rest_tri[:, 2] - rest_tri[:, 0]
    )
    reference_area = np.linalg.norm(rest_cross, axis=1) / 2
    assert np.all(reference_area > 0)
    vertex_area = np.zeros(len(skin_ids))
    np.add.at(vertex_area, triangles.ravel(), np.repeat(reference_area / 3, 3))
    fit_delta = skin_fit - target_skin
    all_face_fit_rms_mm = float(
        1000
        * np.sqrt(
            np.sum(vertex_area * np.sum(fit_delta**2, axis=1)) / vertex_area.sum()
        )
    )
    chin_patch_fit_rms_mm = float(
        1000
        * np.sqrt(
            np.sum(vertex_area[patch_ids] * np.sum(fit_delta[patch_ids] ** 2, axis=1))
            / vertex_area[patch_ids].sum()
        )
    )
    fit_tri = skin_fit[triangles]
    target_tri = target_skin[triangles]
    fit_cross = np.cross(fit_tri[:, 1] - fit_tri[:, 0], fit_tri[:, 2] - fit_tri[:, 0])
    target_cross = np.cross(
        target_tri[:, 1] - target_tri[:, 0], target_tri[:, 2] - target_tri[:, 0]
    )
    fit_norm = np.linalg.norm(fit_cross, axis=1)
    target_norm = np.linalg.norm(target_cross, axis=1)
    assert np.all(fit_norm > 0)
    assert np.all(target_norm > 0)
    cosine = np.sum(fit_cross * target_cross, axis=1) / (fit_norm * target_norm)
    angle = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    normal_rms_deg = float(
        np.sqrt(np.sum(reference_area * angle**2) / reference_area.sum())
    )
    np.testing.assert_allclose(
        all_face_fit_rms_mm, summary["final"]["fit_rms_mm"], rtol=1e-10
    )
    old_tet_points = rest[cells]
    new_tet_points = deformed[cells]
    old_det = np.linalg.det(
        (old_tet_points[:, 1:] - old_tet_points[:, :1]).transpose(0, 2, 1)
    )
    new_det = np.linalg.det(
        (new_tet_points[:, 1:] - new_tet_points[:, :1]).transpose(0, 2, 1)
    )
    assert np.all(old_det > 0)
    jacobian = new_det / old_det
    assert np.isfinite(jacobian).all()
    assert np.count_nonzero(jacobian <= 0) == summary["final"]["inverted_cells"]
    np.testing.assert_allclose(
        jacobian.min(), summary["final"]["minimum_J"], rtol=1e-8, atol=1e-12
    )

    deleted_tets = original_tets[removed_cells]
    deleted_faces = set(
        map(tuple, np.sort(deleted_tets[:, FACE_PATTERN].reshape(-1, 3), axis=1))
    )
    deleted_vertices = set(deleted_tets.ravel().tolist())
    deleted_vertex_tree = cKDTree(
        original_rest[np.fromiter(deleted_vertices, dtype=np.int64)]
    )
    deleted_centroids = original_rest[deleted_tets].mean(axis=1)
    centroid_tree = cKDTree(deleted_centroids)
    order = np.argsort(jacobian)[: cfg.lowest_count]
    lowest = []
    for new_cell in order:
        vertex_ids = cells[new_cell]
        original_ids = old_points[vertex_ids]
        old_id = int(old_cells[new_cell])
        original_faces = np.sort(original_ids[FACE_PATTERN], axis=1)
        centroid = rest[vertex_ids].mean(axis=0)
        _, closest_index = centroid_tree.query(centroid)
        nearest_removed_cell = int(removed_cells[closest_index])
        nearest_vertex_distance, _ = deleted_vertex_tree.query(rest[vertex_ids])
        lowest.append(
            {
                "new_cell_id": int(new_cell),
                "original_cell_id": old_id,
                "J": float(jacobian[new_cell]),
                "rest_signed_volume_m3": float(old_det[new_cell] / 6),
                "deformed_signed_volume_m3": float(new_det[new_cell] / 6),
                "fixed_corner_count": int(fixed[vertex_ids].sum()),
                "jaw_routed_corner_count": int(jaw[vertex_ids].sum()),
                "active": bool(mesh.cell_data["ActivationMask"][new_cell]),
                "material_fractions": {
                    label: float(mesh.cell_data[field][new_cell])
                    for label, field in (
                        ("fat", "FatFraction"),
                        ("aponeurosis", "AponeurosisFraction"),
                        ("muscle", "MuscleFraction"),
                        ("smas", "SMASFraction"),
                    )
                },
                "corners": [
                    {
                        "new_point_id": int(new_point),
                        "original_point_id": int(old_point),
                        "group": group_names[group[new_point]]
                        if group[new_point] >= 0
                        else "unlabelled",
                        "is_fixed": bool(fixed[new_point]),
                        "jaw_routed": bool(jaw[new_point]),
                    }
                    for new_point, old_point in zip(
                        vertex_ids, original_ids, strict=True
                    )
                ],
                "deleted_cell_adjacency": {
                    "shared_face_count": sum(
                        tuple(face) in deleted_faces for face in original_faces
                    ),
                    "shared_vertex_count": sum(
                        int(vertex in deleted_vertices) for vertex in original_ids
                    ),
                    "minimum_vertex_distance_to_deleted_vertex_mm": float(
                        nearest_vertex_distance.min() * 1000
                    ),
                    "nearest_deleted_cell_original_id_by_rest_centroid": nearest_removed_cell,
                    "nearest_deleted_cell_rest_centroid_distance_mm": float(
                        np.linalg.norm(centroid - deleted_centroids[closest_index])
                        * 1000
                    ),
                },
            }
        )
    near = jacobian < 0.1
    near_fixed = Counter(str(int(value)) for value in fixed[cells[near]].sum(axis=1))
    last_attempt = summary["attempts"][-1]
    assert last_attempt["status"] == "rejected_carry_volume_bound"
    target_pose = float(last_attempt["target_fraction"]) * full_pose
    old_rotation = Rotation.from_rotvec(pose[:3]).as_matrix()
    new_rotation = Rotation.from_rotvec(target_pose[:3]).as_matrix()
    carried = (
        (deformed - pivot - pose[3:]) @ old_rotation @ new_rotation.T
        + pivot
        + target_pose[3:]
    )
    candidate = displacement + harmonic_weight[:, None] * (carried - deformed)
    candidate[jaw] = (
        (rest[jaw] - pivot) @ new_rotation.T + pivot + target_pose[3:] - rest[jaw]
    )
    candidate[fixed & ~jaw] = 0.0
    direction = candidate - displacement
    maximum_norm = -np.inf
    limiting_cell = -1
    for start in range(0, len(cells), 100_000):
        chunk = cells[start : start + 100_000]
        x0 = rest[chunk]
        x = deformed[chunk]
        du = direction[chunk]
        dm = (x0[:, 1:] - x0[:, :1]).transpose(0, 2, 1)
        f = (x[:, 1:] - x[:, :1]).transpose(0, 2, 1) @ np.linalg.inv(dm)
        df = (du[:, 1:] - du[:, :1]).transpose(0, 2, 1) @ np.linalg.inv(dm)
        a = np.linalg.solve(f, df)
        norms = np.linalg.norm(a, axis=(1, 2))
        local = int(np.argmax(norms))
        if norms[local] > maximum_norm:
            maximum_norm = float(norms[local])
            limiting_cell = start + local
    measured_bound = min(1.0, summary["config"]["volume_safety"] / maximum_norm)
    np.testing.assert_allclose(
        measured_bound, last_attempt["carry_volume_fraction"], rtol=5e-6, atol=1e-8
    )
    audit = {
        "schema": "pruned-mouthopen-terminal-cell-audit-v1",
        "status": "observed_near_collapse_on_this_pose_path",
        "scope": "CPU analysis of saved terminal displacement; no new forward solve or mesh edit",
        "terminal_forward_status": summary["status"],
        "completed_pose_fraction": fraction,
        "pose_rad_m": pose.tolist(),
        "full_pose_fraction": fraction,
        "pose_rotation_magnitude_deg": float(np.degrees(np.linalg.norm(pose[:3]))),
        "pose_translation_magnitude_mm": float(1000 * np.linalg.norm(pose[3:])),
        "fit": {
            "all_face_area_weighted_position_rms_mm": all_face_fit_rms_mm,
            "chin_27_vertex_area_weighted_position_rms_mm": chin_patch_fit_rms_mm,
            "all_fitting_triangles_reference_area_weighted_normal_angle_rms_deg": normal_rms_deg,
        },
        "cell_count": len(cells),
        "minimum_J": float(jacobian.min()),
        "inverted_cells": int(np.count_nonzero(jacobian <= 0)),
        "J_below": {
            str(threshold): int(np.count_nonzero(jacobian < threshold))
            for threshold in (0.001, 0.01, 0.1, 0.5)
        },
        "fixed_corner_counts_for_J_below_0p1": dict(near_fixed),
        "lowest_cells": lowest,
        "deleted_cell_count": len(removed_cells),
        "last_rejected_carry": {
            "attempt_index": last_attempt["index"],
            "target_fraction": last_attempt["target_fraction"],
            "reported_volume_fraction": last_attempt["carry_volume_fraction"],
            "recomputed_volume_fraction": measured_bound,
            "maximum_F_inverse_dF_frobenius": maximum_norm,
            "limiting_new_cell_id": limiting_cell,
            "limiting_original_cell_id": int(old_cells[limiting_cell]),
            "limiting_cell_is_current_minimum_J_cell": bool(
                limiting_cell == int(order[0])
            ),
        },
        "interpretation_limit": "A near-collapsed retained cell blocks this saved path; this is no proof that every pose path or changed physical model is impossible.",
        "sources": {
            "checkpoint": receipt(checkpoint),
            "forward_summary": receipt(summary_path),
            "volume": receipt(volume_path),
            "skin": receipt(skin_path),
            "mapping": receipt(mapping_path),
            "original_preparation": receipt(prepared_path),
            "harmonic_weight": receipt(harmonic_path),
            "script": receipt(Path(__file__)),
        },
    }
    output_path.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    cherries.log_metrics(
        {
            "terminal/fraction": fraction,
            "terminal/min_J": float(jacobian.min()),
            "terminal/inverted_cells": int(np.count_nonzero(jacobian <= 0)),
            "terminal/J_below_0p1": int(near.sum()),
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
