# Copyright (c) 2026 liblaf
"""Fit MouthOpen chin motion and audit the historical fixed boundary on CPU."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]
ROOT = GROUP.parents[4]
HISTORICAL = ROOT / "exp/2026/09/07/face-actuation-diagnosis/data/12-historical-fixture"
TRANSFER = ROOT / "exp/2026/09/23/new-neutral/data/blendshapes-005/blendshapes.npz"
CHIN = ROOT / "exp/2026/09/23/new-neutral/data/chin-rigid-pose-001/estimate.json"
HINGE = (
    ROOT
    / "exp/2026/09/21/joint-activation-material-mandible/data/expression-fitting-007/protocol.json"
)
CHIN_HELPER = ROOT / "exp/2026/09/23/new-neutral/src/chin_rigid_pose.py"
HISTORICAL_SRC = ROOT / "exp/2026/09/21/stress-activation-loss/src"
sys.path[:0] = [str(CHIN_HELPER.parent), str(HISTORICAL_SRC)]

from chin_rigid_pose import estimate_rigid_chin_pose  # noqa: E402
from experiment import Profile  # noqa: E402

FRACTIONS = (0.0, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0)


class Config(cherries.BaseConfig):
    output: Path = Path("10-mandible")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, str | int]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def rigid_points(points: np.ndarray, pivot: np.ndarray, pose: np.ndarray) -> np.ndarray:
    rotation = Rotation.from_rotvec(pose[:3]).as_matrix()
    return (points - pivot) @ rotation.T + pivot + pose[3:]


def scaled_pose(pose: np.ndarray, fraction: float) -> np.ndarray:
    """Interpolate from identity by scaling rotation vector and translation."""
    return fraction * pose


def fixed_tet_jacobian(
    vertices: np.ndarray,
    local_tets: np.ndarray,
    jaw_local: np.ndarray,
    pivot: np.ndarray,
    pose: np.ndarray,
    rest_det: np.ndarray,
    fraction: float,
) -> np.ndarray:
    moved = vertices.copy()
    moved[jaw_local] = rigid_points(
        moved[jaw_local], pivot, scaled_pose(pose, fraction)
    )
    cells = moved[local_tets]
    edges = (cells[:, 1:] - cells[:, :1]).transpose(0, 2, 1)
    return np.linalg.det(edges) / rest_det


def pose_path_audit(
    rest: np.ndarray,
    fixed_tets: np.ndarray,
    jaw: np.ndarray,
    pivot: np.ndarray,
    pose: np.ndarray,
) -> tuple[list[dict], float | None, np.ndarray]:
    cells = rest[fixed_tets]
    rest_det = np.linalg.det((cells[:, 1:] - cells[:, :1]).transpose(0, 2, 1))
    assert np.all(rest_det > 0)
    vertex_ids, inverse = np.unique(fixed_tets, return_inverse=True)
    vertices = rest[vertex_ids]
    local_tets = inverse.reshape(-1, 4)
    jaw_local = jaw[vertex_ids]

    def jacobian(fraction: float) -> np.ndarray:
        return fixed_tet_jacobian(
            vertices, local_tets, jaw_local, pivot, pose, rest_det, fraction
        )

    samples = []
    for fraction in FRACTIONS:
        value = jacobian(fraction)
        samples.append(
            {
                "fraction": fraction,
                "angle_deg": float(np.degrees(np.linalg.norm(pose[:3])) * fraction),
                "translation_mm": float(1000 * np.linalg.norm(pose[3:]) * fraction),
                "minimum_J": float(value.min()),
                "inverted_allfixed_tetrahedra": int(np.count_nonzero(value <= 0)),
            }
        )
    # A dense scalar scan finds the earliest sampled crossing; bisection then
    # locates that root. The certificate is specific to this rigid path.
    previous_fraction = 0.0
    previous_minimum = 1.0
    bracket = None
    for fraction in np.linspace(0.001, 1.0, 1000):
        minimum = float(jacobian(float(fraction)).min())
        if previous_minimum > 0 >= minimum:
            bracket = (previous_fraction, float(fraction))
            break
        previous_fraction, previous_minimum = float(fraction), minimum
    threshold = None
    if bracket is not None:
        low, high = bracket
        for _ in range(50):
            middle = (low + high) / 2
            if float(jacobian(middle).min()) > 0:
                low = middle
            else:
                high = middle
        threshold = high
    return samples, threshold, jacobian(1.0)


def main(cfg: Config) -> None:  # noqa: PLR0915
    inputs = {
        "historical_volume": cherries.input(HISTORICAL / "volume.vtu"),
        "historical_skin": cherries.input(HISTORICAL / "skin.vtp"),
        "blendshapes": cherries.input(TRANSFER),
        "chin_patch_receipt": cherries.input(CHIN),
        "hinge_protocol": cherries.input(HINGE),
    }
    out = cherries.output(cfg.output / "pose.json", mkdir=True).parent
    assert not (out / "pose.json").exists(), out
    mesh = pv.read(inputs["historical_volume"])
    skin = pv.read(inputs["historical_skin"])
    assert isinstance(mesh, pv.UnstructuredGrid)
    assert isinstance(skin, pv.PolyData)
    rest = np.asarray(mesh.points, dtype=np.float64)
    tets = np.asarray(mesh.cells).reshape(-1, 5)[:, 1:].astype(np.int64)
    with np.load(inputs["blendshapes"], allow_pickle=False) as bundle:
        names = [str(name) for name in bundle["expression_names"]]
        expression_index = names.index("MouthOpen")
        skin_ids = np.asarray(bundle["skin_global_ids"], dtype=np.int64).copy()
        triangles = np.asarray(bundle["skin_triangles"], dtype=np.int64).copy()
        source_neutral = np.asarray(bundle["source_neutral_points_m"]).copy()
        transferred = np.asarray(
            bundle["expression_displacement_m"][expression_index]
        ).copy()
    np.testing.assert_array_equal(skin_ids, skin.point_data["GlobalPointId"])
    np.testing.assert_array_equal(
        triangles, np.asarray(skin.faces).reshape(-1, 4)[:, 1:]
    )
    np.testing.assert_array_equal(rest[skin_ids], np.asarray(skin.points))
    np.testing.assert_array_equal(rest[skin_ids], source_neutral)
    np.testing.assert_array_equal(
        transferred, np.asarray(mesh.point_data["MouthOpen"])[skin_ids]
    )
    target = rest[skin_ids] + transferred
    chin_receipt = json.loads(inputs["chin_patch_receipt"].read_text())
    assert chin_receipt["schema"] == "chin-rigid-pose-estimate-v1"
    patch = np.asarray(chin_receipt["patch_local_ids"], dtype=np.int64)
    assert len(patch) == 27
    hinge_contract = json.loads(inputs["hinge_protocol"].read_text())["mandible_pose"]
    pivot = np.asarray(hinge_contract["pivot_m"], dtype=np.float64)
    axis = np.asarray(hinge_contract["axis_world"], dtype=np.float64)
    np.testing.assert_array_equal(pivot, chin_receipt["mandible_pivot_m"])
    assert abs(np.linalg.norm(axis) - 1.0) < 1e-12
    full = estimate_rigid_chin_pose(rest[skin_ids], target, triangles, patch, pivot)
    pose = np.asarray(full["pose_rad_m"], dtype=np.float64)
    assert full["det_rotation"] > 0.999999999999

    # Fit the documented one-DOF hinge independently as a constrained seed.
    tri = rest[skin_ids][triangles]
    area = (
        np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        / 2
    )
    area_weights = np.zeros(len(skin_ids))
    np.add.at(area_weights, triangles.ravel(), np.repeat(area / 3, 3))
    weights = area_weights[patch] / area_weights[patch].sum()
    x, y = rest[skin_ids][patch], target[patch]

    def hinge_error(angle: float) -> float:
        candidate = np.concatenate((axis * angle, np.zeros(3)))
        delta = rigid_points(x, pivot, candidate) - y
        return float(np.sum(weights * np.sum(delta * delta, axis=1)))

    bounds = np.radians(hinge_contract["angle_bounds_deg"])
    hinge_fit = minimize_scalar(
        hinge_error, bounds=tuple(bounds), method="bounded", options={"xatol": 1e-13}
    )
    assert hinge_fit.success
    hinge_pose = np.concatenate((axis * hinge_fit.x, np.zeros(3)))

    fixed = np.asarray(mesh.point_data["IsFixed"], dtype=bool)
    fixed_mask = np.asarray(mesh.point_data["FixedMask"], dtype=bool)
    assert fixed_mask.shape == (len(rest), 3)
    np.testing.assert_array_equal(fixed_mask, np.repeat(fixed[:, None], 3, axis=1))
    group_names = [str(name) for name in mesh.field_data["GroupName"]]
    assert group_names.count("Mandible") == 1
    group = np.asarray(mesh.point_data["GroupId"], dtype=int)
    mandible_group_id = group_names.index("Mandible")
    assert mandible_group_id == 28
    assert np.count_nonzero(group == mandible_group_id) == 7510
    jaw = fixed & (group == mandible_group_id)
    assert np.count_nonzero(jaw) == 6145
    allfixed_ids = np.flatnonzero(fixed[tets].all(axis=1))
    fixed_tets = tets[allfixed_ids]
    samples, threshold, full_j = pose_path_audit(rest, fixed_tets, jaw, pivot, pose)
    inverted = allfixed_ids[full_j <= 0]
    boundary_u = np.zeros_like(rest)
    boundary_u[jaw] = rigid_points(rest[jaw], pivot, pose) - rest[jaw]
    material_names = ("fat", "aponeurosis", "muscle", "smas")
    fraction_fields = (
        "FatFraction",
        "AponeurosisFraction",
        "MuscleFraction",
        "SMASFraction",
    )
    by_cell = []
    for cell_id in inverted:
        row = int(np.searchsorted(allfixed_ids, cell_id))
        ids = tets[cell_id]
        fractions = {
            name: float(mesh.cell_data[field][cell_id])
            for name, field in zip(material_names, fraction_fields, strict=True)
        }
        by_cell.append(
            {
                "cell_id": int(cell_id),
                "J": float(full_j[row]),
                "vertex_ids": ids.tolist(),
                "vertex_groups": [
                    group_names[group[i]] if group[i] >= 0 else "unlabelled"
                    for i in ids
                ],
                "jaw_routed_vertices": int(jaw[ids].sum()),
                "material_fractions": fractions,
                "activation_mask": bool(mesh.cell_data["ActivationMask"][cell_id]),
            }
        )
    prepared_path = cherries.output(cfg.output / "prepared.npz", mkdir=True)
    np.savez_compressed(
        prepared_path,
        X=rest,
        tets=tets,
        skin_ids=skin_ids,
        triangles=triangles,
        target_skin=target,
        patch_ids=patch,
        pivot=pivot,
        pose=pose,
        hinge_pose=hinge_pose,
        boundary_u=boundary_u,
        fixed_mask=fixed_mask,
        jaw_mask=jaw,
        allfixed_cell_ids=allfixed_ids,
        inverted_allfixed_cell_ids=inverted,
    )
    sources = {key: record(path) for key, path in inputs.items()}
    sources.update(
        {
            "chin_helper": record(CHIN_HELPER),
            "preparation_script": record(Path(__file__)),
            "neutral_mandible": record(
                ROOT.parent / "melon/exp/2026/05/27/head/data/13-mandible.ply"
            ),
            "mandible_landmarks": record(
                ROOT.parent
                / "melon/exp/2026/05/27/head/data/11-mandible.landmarks.json"
            ),
        }
    )
    pose_receipt = {
        "schema": "historical-mouthopen-mandible-pose-v1",
        "status": "blocked_by_prescribed_cell_inversions"
        if len(inverted)
        else "geometry_seed_only",
        "target": "MouthOpen transferred displacement matches original historical volume point data exactly on every skin vertex",
        "full_rigid_chin_fit": full,
        "hinge_only_fit": {
            "pose_rad_m": hinge_pose.tolist(),
            "angle_deg": float(np.degrees(hinge_fit.x)),
            "weighted_patch_rms_mm": float(1000 * np.sqrt(hinge_fit.fun)),
            "axis_world": axis.tolist(),
            "translation_fixed_zero": True,
        },
        "limitations": "Skin patch fit is an estimated jaw seed, not measured bone motion or a validated FEM equilibrium.",
        "sources": sources,
    }
    write_json(out / "pose.json", pose_receipt)
    audit = {
        "schema": "historical-mouthopen-isfixed-kinematic-audit-v1",
        "status": "blocked_by_prescribed_cell_inversions"
        if len(inverted)
        else "no_forced_inversion_found",
        "scope": "CPU kinematic audit of prescribed IsFixed nodes; no forward solve or contact evaluation",
        "boundary_policy": "IsFixed is the sole original FEM clamp; only IsFixed intersect Mandible gets rigid jaw displacement",
        "mandible_group_id_from_GroupName": mandible_group_id,
        "mandible_group_vertices": int(np.count_nonzero(group == mandible_group_id)),
        "fixed_mask_equals_three_component_IsFixed": True,
        "fixed_nodes": int(fixed.sum()),
        "jaw_routed_fixed_nodes": int(jaw.sum()),
        "allfixed_tetrahedra": len(allfixed_ids),
        "full_pose_inverted_allfixed_tetrahedra": len(inverted),
        "full_pose_min_J_allfixed": float(full_j.min()),
        "sampled_pose_path": samples,
        "first_sampled_positive_to_inverted_threshold_fraction_bisected": threshold,
        "threshold_scope": "first crossing bracket found on a 0.001-spaced scalar scan, then bisected; diagnostic for this one SE3 path only",
        "inverted_cells": by_cell,
        "inverted_cell_jaw_vertex_count_histogram": dict(
            Counter(str(row["jaw_routed_vertices"]) for row in by_cell)
        ),
        "prepared": record(prepared_path),
        "sources": sources,
    }
    audit_path = cherries.output(cfg.output / "audit.json", mkdir=True)
    write_json(audit_path, audit)
    cherries.log_metrics(
        {
            "chin/full_rms_mm": full["weighted_rms_after_m"] * 1000,
            "chin/hinge_rms_mm": float(np.sqrt(hinge_fit.fun) * 1000),
            "boundary/fixed_nodes": int(fixed.sum()),
            "boundary/jaw_nodes": int(jaw.sum()),
            "boundary/inverted_allfixed_tetrahedra": len(inverted),
            "boundary/min_J": float(full_j.min()),
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
