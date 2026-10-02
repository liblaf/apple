# ruff: noqa: CPY001, PLR0915
"""Audit the step-205 inversion using only frozen CPU-side geometry."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
import scipy
import torch
from scipy.spatial import cKDTree

GROUP = Path(__file__).resolve().parents[1]
FIXTURE = GROUP.parents[1] / "07/face-actuation-diagnosis/data/12-historical-fixture"
CANONICAL_200 = (
    Path(os.environ["APPLE_HISTORICAL_WORKTREE"])
    / "exp/2026/09/08/physical-volume-baseline/data/20-baseline/optimizer-latest.pt"
)
RUN = GROUP / "data/20-fit300"
ACCEPTED_204 = RUN / "last.npz"
REJECTED_205 = RUN / "rejected-step-0205.npz"
REG_RUN = GROUP / "data/22-reg300"
REG_ACCEPTED_204 = REG_RUN / "last.npz"
REG_REJECTED_205 = REG_RUN / "rejected-step-0205.npz"
OUTPUT = GROUP / "data/11-inversion-audit"

KNOWN_CELL = 573586
NEW_CELL = 620845
TARGET_CELLS = (KNOWN_CELL, NEW_CELL)

EXPECTED_INPUT_HASHES = {
    CANONICAL_200: "c2f77e4bdb09e8e0be50d0f8ec9a9926ea7a4bc08db5a368ec83dcb768707b58",
    ACCEPTED_204: "82d7e68249a3c2e26612158ce83f2d53d6215c83810e6309605180b45974c757",
    REJECTED_205: "de05719f0e59b9840ed2841b8bb8966e85799325266fafebab21e017eae20f95",
    RUN
    / "rejected-geometry.json": "094d0d2eed62d4f04e56f68fb3c2a2cd70e9ec2fcd66dc4d479f3082a74ee3af",
    RUN
    / "failure.json": "a1b529830322d0ecc6f6f4b6d42287473f9a5fad56503fb4fbe0e8cac1fdad6a",
    RUN
    / "solver-receipts.jsonl": "6d1953659bfeda37b5054ef701eb30b1c75c222bdc80b73972b6c2ca0885abf8",
    RUN
    / "provenance.json": "62bab2da51c7aeedac6514fbb1633ce85651cc4c34215753b7ca02d16b93d233",
    FIXTURE
    / "volume.vtu": "238962d0d27a2d35b6211a7a60204d362187b4190dfc7591756bc73ea26ff3b6",
    FIXTURE
    / "skin.vtp": "79eed2a5e2b5f23e84287fe729989ab356cb2780e3342b722b074a9231297833",
    REG_ACCEPTED_204: "063fe95418ebb97ec8c631622d1f7fe49493f9521432c9835c9f3869df8f989c",
    REG_REJECTED_205: "7ec7862590d11682df8a603fe57dd71ac187c1355ab4defd84dde95a1e178cdd",
    REG_RUN
    / "rejected-geometry.json": "52399517efb6d9ecdd557d14aabc3f12aeabe9962ba8dc7ad9f9c013c44fb7c7",
    REG_RUN
    / "failure.json": "a1b529830322d0ecc6f6f4b6d42287473f9a5fad56503fb4fbe0e8cac1fdad6a",
    REG_RUN
    / "solver-receipts.jsonl": "4ba0cfc5c8838add15eef23fc8aea8986d3ad6d44e870794734b0b90db3fe5ed",
    REG_RUN
    / "provenance.json": "7c40f37ce47d60cec0b210c655c89ef1ad0bb6f517a46e3236a5a8917ed5892d",
}

SOURCE_PATHS = {
    "audit": Path(__file__).resolve(),
    "continuation_runner": GROUP / "src/20-continue.py",
    "geometry_reference": GROUP / "src/baseline_physics.py",
    "regularized_runner": GROUP / "src/22-regularized-branch.py",
}
EXPECTED_SUPPORTING_SOURCE_HASHES = {
    "continuation_runner": "1088d812da1d6249979f4da98714ec647f9c1056bea636faf9f21b6dda7f0b9e",
    "geometry_reference": "18de2c57ef463bcb137abc7846fef68166481aa619d376067901e67289045c8f",
    "regularized_runner": "d536fc7fd5cf9e0e7d74d10b2b7a42197b9861a828f777270f129f05f0d743b3",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_record(value: np.ndarray) -> dict[str, Any]:
    array = np.ascontiguousarray(value)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256_c_order_bytes": hashlib.sha256(array.tobytes()).hexdigest(),
        "finite": bool(np.isfinite(array).all()),
    }


def json_write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def signed_six_volume(vertices: np.ndarray) -> np.ndarray:
    return np.einsum(
        "...i,...i->...",
        vertices[..., 1, :] - vertices[..., 0, :],
        np.cross(
            vertices[..., 2, :] - vertices[..., 0, :],
            vertices[..., 3, :] - vertices[..., 0, :],
        ),
    )


def edge_lengths(vertices: np.ndarray) -> np.ndarray:
    return np.array(
        [
            np.linalg.norm(vertices[i] - vertices[j])
            for i in range(4)
            for j in range(i + 1, 4)
        ]
    )


def minimum_altitude(vertices: np.ndarray) -> float:
    volume = abs(float(signed_six_volume(vertices))) / 6
    areas = []
    for excluded in range(4):
        face = vertices[np.arange(4) != excluded]
        areas.append(np.linalg.norm(np.cross(face[1] - face[0], face[2] - face[0])) / 2)
    assert min(areas) > 0
    return float(min(3 * volume / np.asarray(areas)))


def mean_ratio_quality(vertices: np.ndarray) -> float:
    volume = abs(float(signed_six_volume(vertices))) / 6
    lengths = edge_lengths(vertices)
    return float(12 * (3 * volume) ** (2 / 3) / np.sum(lengths**2))


def point_triangle_distance(point: np.ndarray, triangles: np.ndarray) -> float:
    """Exact point-to-triangle distance via face projection and three edges."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac, ap = b - a, c - a, point - a
    normal = np.cross(ab, ac)
    normal_sq = np.einsum("ij,ij->i", normal, normal)
    assert np.all(normal_sq > 0)
    signed_numerator = np.einsum("ij,ij->i", ap, normal)
    projection = point - signed_numerator[:, None] * normal / normal_sq[:, None]
    v0, v1, v2 = ab, ac, projection - a
    d00 = np.einsum("ij,ij->i", v0, v0)
    d01 = np.einsum("ij,ij->i", v0, v1)
    d11 = np.einsum("ij,ij->i", v1, v1)
    d20 = np.einsum("ij,ij->i", v2, v0)
    d21 = np.einsum("ij,ij->i", v2, v1)
    denominator = d00 * d11 - d01 * d01
    assert np.all(denominator > 0)
    bary_v = (d11 * d20 - d01 * d21) / denominator
    bary_w = (d00 * d21 - d01 * d20) / denominator
    bary_u = 1 - bary_v - bary_w
    inside = (bary_u >= 0) & (bary_v >= 0) & (bary_w >= 0)
    plane_distance = np.abs(signed_numerator) / np.sqrt(normal_sq)

    def segment_distance(start: np.ndarray, end: np.ndarray) -> np.ndarray:
        direction = end - start
        fraction = np.einsum("ij,ij->i", point - start, direction) / np.einsum(
            "ij,ij->i", direction, direction
        )
        fraction = np.clip(fraction, 0, 1)
        closest = start + fraction[:, None] * direction
        return np.linalg.norm(point - closest, axis=1)

    boundary_distance = np.minimum.reduce(
        (segment_distance(a, b), segment_distance(b, c), segment_distance(c, a))
    )
    return float(np.min(np.where(inside, plane_distance, boundary_distance)))


def state_geometry(
    *,
    label: str,
    step: int,
    q: np.ndarray,
    u: np.ndarray,
    rest: np.ndarray,
    tets: np.ndarray,
    rest_edge_matrix: np.ndarray,
    rest_signed_six: np.ndarray,
    active_ids: np.ndarray,
    face_ids: np.ndarray,
    skin_triangles: np.ndarray,
    expected_inverted: list[int],
    saved_fields: dict[str, Any],
) -> dict[str, Any]:
    current = rest + u
    vertices = current[tets]

    # Independent route 1: signed tetrahedral-volume ratio.
    j_scalar = signed_six_volume(vertices) / rest_signed_six
    # Independent route 2: construct each full deformation gradient.
    current_edges = np.transpose(vertices[:, 1:, :] - vertices[:, :1, :], (0, 2, 1))
    deformation_gradient = current_edges @ np.linalg.inv(rest_edge_matrix)
    j_full = np.linalg.det(deformation_gradient)
    scalar_ids = np.flatnonzero(j_scalar <= 0).tolist()
    full_ids = np.flatnonzero(j_full <= 0).tolist()
    assert scalar_ids == full_ids == expected_inverted

    active_centroids = current[tets[active_ids]].mean(axis=1)
    active_tree = cKDTree(active_centroids)
    face_tree = cKDTree(current[face_ids])
    deformed_skin = current[skin_triangles]
    cells: dict[str, Any] = {}
    for cell_id in TARGET_CELLS:
        cell_vertices = current[tets[cell_id]]
        centroid = cell_vertices.mean(axis=0)
        active_distance, active_local_id = active_tree.query(centroid)
        face_distance, face_local_id = face_tree.query(centroid)
        singular_values = np.linalg.svd(deformation_gradient[cell_id], compute_uv=False)
        cells[str(cell_id)] = {
            "j_scalar_volume_ratio": float(j_scalar[cell_id]),
            "j_full_det_F": float(j_full[cell_id]),
            "signed_volume_mm3": float(signed_six_volume(cell_vertices) / 6 * 1e9),
            "deformation_gradient": deformation_gradient[cell_id].tolist(),
            "singular_values": singular_values.tolist(),
            "singular_value_condition_number": float(
                singular_values.max() / singular_values.min()
            ),
            "edge_length_min_mm": float(edge_lengths(cell_vertices).min() * 1e3),
            "edge_length_max_mm": float(edge_lengths(cell_vertices).max() * 1e3),
            "minimum_altitude_um": minimum_altitude(cell_vertices) * 1e6,
            "mean_ratio_quality_absolute_volume": mean_ratio_quality(cell_vertices),
            "centroid_m": centroid.tolist(),
            "centroid_to_nearest_IsFace_vertex_mm": float(face_distance * 1e3),
            "nearest_IsFace_vertex_id": int(face_ids[int(face_local_id)]),
            "centroid_to_skin_triangle_mm": point_triangle_distance(
                centroid, deformed_skin
            )
            * 1e3,
            "centroid_to_nearest_active_cell_mm": float(active_distance * 1e3),
            "nearest_active_cell_id": int(active_ids[int(active_local_id)]),
        }
    cells[str(NEW_CELL)]["centroid_to_known_cell_573586_mm"] = float(
        np.linalg.norm(
            current[tets[NEW_CELL]].mean(axis=0)
            - current[tets[KNOWN_CELL]].mean(axis=0)
        )
        * 1e3
    )
    cells[str(NEW_CELL)]["shared_vertices_with_known_cell_573586"] = sorted(
        set(map(int, tets[NEW_CELL])) & set(map(int, tets[KNOWN_CELL]))
    )

    fixed_mask = saved_fields.pop("fixed_mask")
    new_vertices = tets[NEW_CELL]
    fixed_local = np.flatnonzero(fixed_mask[new_vertices])
    free_local = np.flatnonzero(~fixed_mask[new_vertices])
    assert fixed_local.tolist() == [1, 2, 3] and free_local.tolist() == [0]
    fixed_face = current[new_vertices[fixed_local]]
    free_vertex = current[new_vertices[free_local[0]]]
    normal = np.cross(fixed_face[1] - fixed_face[0], fixed_face[2] - fixed_face[0])
    cells[str(NEW_CELL)]["free_vertex_id"] = int(new_vertices[free_local[0]])
    cells[str(NEW_CELL)]["fixed_opposite_face_vertex_ids"] = list(
        map(int, new_vertices[fixed_local])
    )
    cells[str(NEW_CELL)]["free_vertex_signed_altitude_to_fixed_face_um"] = float(
        np.dot(free_vertex - fixed_face[0], normal) / np.linalg.norm(normal) * 1e6
    )

    return {
        "label": label,
        "step": step,
        "saved_fields": saved_fields,
        "arrays": {"q": array_record(q), "u": array_record(u)},
        "determinant_audit": {
            "scalar_definition": "signed current six-volume / signed rest six-volume",
            "full_definition": "det(Ds @ inv(Dm))",
            "max_abs_scalar_minus_full": float(np.max(np.abs(j_scalar - j_full))),
            "scalar_inverted_cell_ids": scalar_ids,
            "full_inverted_cell_ids": full_ids,
            "inverted_id_sets_match": True,
            "all_finite": bool(
                np.isfinite(j_scalar).all() and np.isfinite(j_full).all()
            ),
            "minimum_j": float(j_scalar.min()),
            "maximum_j": float(j_scalar.max()),
        },
        "cells": cells,
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    assert not any(OUTPUT.iterdir()), f"Output directory must be empty: {OUTPUT}"

    before_hashes = {path: sha256(path) for path in EXPECTED_INPUT_HASHES}
    assert before_hashes == EXPECTED_INPUT_HASHES
    for name, expected in EXPECTED_SUPPORTING_SOURCE_HASHES.items():
        assert sha256(SOURCE_PATHS[name]) == expected
    assert sha256(SOURCE_PATHS["continuation_runner"]) == sha256(
        RUN / "sources/experiment/20-continue.py"
    )
    assert sha256(SOURCE_PATHS["regularized_runner"]) == sha256(
        REG_RUN / "sources/experiment/22-regularized-branch.py"
    )
    regularized_provenance = json.loads((REG_RUN / "provenance.json").read_text())
    assert regularized_provenance["sources"]["__main__"]["sha256"] == sha256(
        SOURCE_PATHS["regularized_runner"]
    )

    volume = pv.read(FIXTURE / "volume.vtu")
    skin = pv.read(FIXTURE / "skin.vtp")
    rest = np.asarray(volume.points, dtype=np.float64)
    tets = np.asarray(volume.cells, dtype=np.int64).reshape(-1, 5)[:, 1:]
    assert rest.shape == (228660, 3) and tets.shape == (1146517, 4)
    rest_vertices = rest[tets]
    rest_edge_matrix = np.transpose(
        rest_vertices[:, 1:, :] - rest_vertices[:, :1, :], (0, 2, 1)
    )
    rest_signed_six = signed_six_volume(rest_vertices)
    assert np.all(rest_signed_six > 0)
    active_ids = np.flatnonzero(np.asarray(volume.cell_data["ActivationMask"], bool))
    assert active_ids.shape == (288235,)
    face_ids = np.flatnonzero(np.asarray(volume.point_data["IsFace"], bool))
    skin_global = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    skin_triangles = skin_global[
        np.asarray(skin.faces, dtype=np.int64).reshape(-1, 4)[:, 1:]
    ]
    fixed_mask = np.asarray(volume.point_data["IsFixed"], bool)

    checkpoint = torch.load(CANONICAL_200, map_location="cpu", weights_only=False)
    assert checkpoint["step"] == 200
    q200 = checkpoint["q"].detach().cpu().numpy()
    u200 = np.asarray(checkpoint["u"])

    with np.load(ACCEPTED_204, allow_pickle=False) as accepted:
        assert int(accepted["step"]) == 204 and bool(accepted["solver_valid"])
        assert np.array_equal(accepted["rest_points"], rest)
        assert np.array_equal(accepted["active_ids"], active_ids)
        q204, u204 = accepted["q"].copy(), accepted["u"].copy()
        accepted_saved = {
            "solver_valid": bool(accepted["solver_valid"]),
            "physical_volume_energy": bool(accepted["physical_volume_energy"]),
            "accepted": True,
            "forward_success": True,
            "adjoint_evaluated": True,
            "physically_valid": False,
        }
    with np.load(REJECTED_205, allow_pickle=False) as rejected:
        assert int(rejected["step"]) == 205 and bool(rejected["solver_valid"])
        assert np.array_equal(rejected["rest_points"], rest)
        assert np.array_equal(rejected["active_ids"], active_ids)
        q205, u205 = rejected["q"].copy(), rejected["u"].copy()
        rejected_saved = {
            "solver_valid_original": bool(rejected["solver_valid"]),
            "physical_volume_energy": bool(rejected["physical_volume_energy"]),
            "accepted": False,
            "forward_success": True,
            "adjoint_evaluated": False,
            "physically_valid": False,
        }
    with np.load(REG_ACCEPTED_204, allow_pickle=False) as regularized_accepted:
        assert int(regularized_accepted["step"]) == 204
        assert bool(regularized_accepted["solver_valid"])
        assert np.array_equal(regularized_accepted["rest_points"], rest)
        assert np.array_equal(regularized_accepted["active_ids"], active_ids)
        q_reg204 = regularized_accepted["q"].copy()
        u_reg204 = regularized_accepted["u"].copy()
    with np.load(REG_REJECTED_205, allow_pickle=False) as regularized_rejected:
        assert int(regularized_rejected["step"]) == 205
        assert not bool(regularized_rejected["solver_valid"])
        assert np.array_equal(regularized_rejected["rest_points"], rest)
        assert np.array_equal(regularized_rejected["active_ids"], active_ids)
        q_reg205 = regularized_rejected["q"].copy()
        u_reg205 = regularized_rejected["u"].copy()
    rejection_record = json.loads((RUN / "rejected-geometry.json").read_text())
    failure_record = json.loads((RUN / "failure.json").read_text())
    assert rejection_record["step"] == failure_record["step"] == 205
    assert rejection_record["new_ids"] == [str(NEW_CELL)]
    assert rejection_record["forward"]["success"] is True
    assert rejection_record["gradient_evaluated"] is False
    regularized_rejection = json.loads((REG_RUN / "rejected-geometry.json").read_text())
    assert regularized_rejection["step"] == 205
    assert regularized_rejection["new_ids"] == [str(NEW_CELL)]
    assert regularized_rejection["forward"]["success"] is True
    assert regularized_rejection["gradient_evaluated"] is False

    common = {"fixed_mask": fixed_mask.copy()}
    states = {
        "canonical_step_200": state_geometry(
            label="canonical_step_200",
            step=200,
            q=q200,
            u=u200,
            rest=rest,
            tets=tets,
            rest_edge_matrix=rest_edge_matrix,
            rest_signed_six=rest_signed_six,
            active_ids=active_ids,
            face_ids=face_ids,
            skin_triangles=skin_triangles,
            expected_inverted=[KNOWN_CELL],
            saved_fields={
                **common,
                "accepted": True,
                "forward_success": True,
                "adjoint_evaluated": True,
                "physically_valid": False,
            },
        ),
        "accepted_step_204": state_geometry(
            label="accepted_step_204",
            step=204,
            q=q204,
            u=u204,
            rest=rest,
            tets=tets,
            rest_edge_matrix=rest_edge_matrix,
            rest_signed_six=rest_signed_six,
            active_ids=active_ids,
            face_ids=face_ids,
            skin_triangles=skin_triangles,
            expected_inverted=[KNOWN_CELL],
            saved_fields={**common, **accepted_saved},
        ),
        "rejected_step_205": state_geometry(
            label="rejected_step_205",
            step=205,
            q=q205,
            u=u205,
            rest=rest,
            tets=tets,
            rest_edge_matrix=rest_edge_matrix,
            rest_signed_six=rest_signed_six,
            active_ids=active_ids,
            face_ids=face_ids,
            skin_triangles=skin_triangles,
            expected_inverted=[KNOWN_CELL, NEW_CELL],
            saved_fields={**common, **rejected_saved},
        ),
        "regularized_accepted_step_204": state_geometry(
            label="regularized_accepted_step_204",
            step=204,
            q=q_reg204,
            u=u_reg204,
            rest=rest,
            tets=tets,
            rest_edge_matrix=rest_edge_matrix,
            rest_signed_six=rest_signed_six,
            active_ids=active_ids,
            face_ids=face_ids,
            skin_triangles=skin_triangles,
            expected_inverted=[KNOWN_CELL],
            saved_fields={
                **common,
                "solver_valid": True,
                "accepted": True,
                "forward_success": True,
                "adjoint_evaluated": True,
                "physically_valid": False,
            },
        ),
        "regularized_rejected_step_205": state_geometry(
            label="regularized_rejected_step_205",
            step=205,
            q=q_reg205,
            u=u_reg205,
            rest=rest,
            tets=tets,
            rest_edge_matrix=rest_edge_matrix,
            rest_signed_six=rest_signed_six,
            active_ids=active_ids,
            face_ids=face_ids,
            skin_triangles=skin_triangles,
            expected_inverted=[KNOWN_CELL, NEW_CELL],
            saved_fields={
                **common,
                "solver_valid": False,
                "accepted": False,
                "forward_success": True,
                "adjoint_evaluated": False,
                "physically_valid": False,
            },
        ),
    }

    static_cells = {}
    for cell_id in TARGET_CELLS:
        vertices = rest[tets[cell_id]]
        tissue = {
            key: float(np.asarray(volume.cell_data[key])[cell_id])
            for key in ("MuscleFraction", "FatFraction", "AponeurosisFraction")
        }
        cell_record = {
            "zero_based_cell_id": cell_id,
            "vertex_ids": list(map(int, tets[cell_id])),
            "fixed_vertex_flags": fixed_mask[tets[cell_id]].tolist(),
            "activation_mask": bool(volume.cell_data["ActivationMask"][cell_id]),
            "activation_control_id": int(
                volume.cell_data["ActivationControlId"][cell_id]
            ),
            "tissue_fractions": tissue,
            "rest_signed_volume_mm3": float(signed_six_volume(vertices) / 6 * 1e9),
            "rest_edge_length_min_mm": float(edge_lengths(vertices).min() * 1e3),
            "rest_edge_length_max_mm": float(edge_lengths(vertices).max() * 1e3),
            "rest_minimum_altitude_mm": minimum_altitude(vertices) * 1e3,
            "rest_mean_ratio_quality": mean_ratio_quality(vertices),
            "rest_edge_matrix_condition_number": float(
                np.linalg.cond(rest_edge_matrix[cell_id])
            ),
        }
        local_fixed = np.flatnonzero(fixed_mask[tets[cell_id]])
        local_free = np.flatnonzero(~fixed_mask[tets[cell_id]])
        if len(local_fixed) == 3 and len(local_free) == 1:
            fixed_face = vertices[local_fixed]
            normal = np.cross(
                fixed_face[1] - fixed_face[0], fixed_face[2] - fixed_face[0]
            )
            signed_altitude = float(
                np.dot(vertices[local_free[0]] - fixed_face[0], normal)
                / np.linalg.norm(normal)
            )
            cell_record.update(
                rest_free_vertex_id=int(tets[cell_id, local_free[0]]),
                rest_fixed_opposite_face_vertex_ids=list(
                    map(int, tets[cell_id, local_fixed])
                ),
                rest_free_vertex_signed_altitude_to_fixed_face_mm=(
                    signed_altitude * 1e3
                ),
                rest_free_vertex_absolute_altitude_to_fixed_face_mm=(
                    abs(signed_altitude) * 1e3
                ),
            )
        static_cells[str(cell_id)] = cell_record

    correction = {
        "schema": "immutable-saved-state-semantic-correction-v1",
        "status": "correction_recorded_without_modifying_original",
        "target": {
            "path": str(REJECTED_205),
            "sha256": before_hashes[REJECTED_205],
            "step": 205,
            "observed_solver_valid": True,
        },
        "reason": (
            "The frozen NPZ was written after a successful forward solve but before "
            "the geometry guard accepted the state. Its solver_valid=true field must "
            "not be interpreted as accepted or physically valid."
        ),
        "override_fields": {
            "solver_valid": False,
            "accepted": False,
            "forward_success": True,
            "adjoint_evaluated": False,
            "physically_valid": False,
        },
        "evidence": {
            "rejected_geometry_path": str(RUN / "rejected-geometry.json"),
            "rejected_geometry_sha256": before_hashes[RUN / "rejected-geometry.json"],
            "new_inverted_cell_ids": [NEW_CELL],
            "forward_success": rejection_record["forward"]["success"],
            "gradient_evaluated": rejection_record["gradient_evaluated"],
            "independent_inverted_cell_ids": states["rejected_step_205"][
                "determinant_audit"
            ]["scalar_inverted_cell_ids"],
        },
        "original_modified": False,
    }
    correction_path = OUTPUT / "correction-receipt.json"
    json_write(correction_path, correction)

    source_snapshots = {}
    source_dir = OUTPUT / "sources/experiment"
    source_dir.mkdir(parents=True)
    for name, live in SOURCE_PATHS.items():
        snapshot = source_dir / live.name
        shutil.copy2(live, snapshot)
        live_hash = sha256(live)
        snapshot_hash = sha256(snapshot)
        assert live_hash == snapshot_hash
        source_snapshots[name] = {
            "live_path": str(live),
            "live_sha256": live_hash,
            "snapshot_path": str(snapshot),
            "snapshot_sha256": snapshot_hash,
            "identical": True,
        }

    after_hashes = {path: sha256(path) for path in EXPECTED_INPUT_HASHES}
    assert after_hashes == before_hashes
    summary = {
        "schema": "physical-volume-inversion-audit-v1",
        "status": "passed",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "scope": (
            "CPU-only postprocess of frozen q200, accepted q204, and rejected q205; "
            "no forward solve, adjoint solve, optimizer update, or source/input edit"
        ),
        "conclusion": (
            "Cell 620845 was already nearly collapsed at canonical step 200, remained "
            "positive at accepted step 204, and became newly inverted at rejected step "
            "205 when its only free vertex crossed its three-fixed-vertex opposite face. "
            "The independently guarded weak-regularization branch also accepted only "
            "through step 204 and newly inverted the same cell at step 205."
        ),
        "cell_id_convention": "zero-based VTK tetrahedron order from volume.vtu",
        "determinant_methods": {
            "scalar": "signed current tetrahedron volume / signed rest tetrahedron volume",
            "full": "det(Ds @ inv(Dm))",
            "acceptance_rule": "J > 0; J <= 0 is inverted",
        },
        "inputs": {
            str(path.relative_to(GROUP)) if path.is_relative_to(GROUP) else str(path): {
                "path": str(path),
                "sha256_before": before_hashes[path],
                "sha256_after": after_hashes[path],
                "unchanged": True,
            }
            for path in EXPECTED_INPUT_HASHES
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pyvista": pv.__version__,
            "torch": torch.__version__,
            "torch_load_map_location": "cpu",
        },
        "source_snapshots": source_snapshots,
        "mesh": {
            "points": len(rest),
            "tetrahedra": len(tets),
            "active_tetrahedra": len(active_ids),
            "IsFace_vertices": len(face_ids),
            "skin_triangles": len(skin_triangles),
            "rest_all_positive_orientation": True,
        },
        "static_cells": static_cells,
        "states": states,
        "common_state_counterfactual": {
            "comparison_scope": "same canonical step 200; common accepted step 204 and independently rejected step 205",
            "baseline_new_cell_620845_j_step_204": states["accepted_step_204"]["cells"][
                str(NEW_CELL)
            ]["j_scalar_volume_ratio"],
            "regularized_new_cell_620845_j_step_204": states[
                "regularized_accepted_step_204"
            ]["cells"][str(NEW_CELL)]["j_scalar_volume_ratio"],
            "baseline_new_cell_620845_j_step_205": states["rejected_step_205"]["cells"][
                str(NEW_CELL)
            ]["j_scalar_volume_ratio"],
            "regularized_new_cell_620845_j_step_205": states[
                "regularized_rejected_step_205"
            ]["cells"][str(NEW_CELL)]["j_scalar_volume_ratio"],
            "same_new_inverted_id_at_step_205": True,
            "regularization_prevented_inversion_through_step_205": False,
        },
        "correction_receipt": {
            "path": str(correction_path),
            "sha256": sha256(correction_path),
            "override_fields": correction["override_fields"],
        },
        "limitations": [
            "This audit establishes the saved geometry and acceptance semantics only.",
            "It does not identify a unique constitutive or optimizer cause for the crossing.",
            "Physical validity remains false for every audited state because cell 573586 is inverted throughout.",
        ],
    }
    json_write(OUTPUT / "summary.json", summary)
    print(json.dumps({"status": summary["status"], "output": str(OUTPUT)}))


if __name__ == "__main__":
    main()
