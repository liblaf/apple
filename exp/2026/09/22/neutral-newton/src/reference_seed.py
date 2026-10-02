"""Geometric cold initialization from the unchanged constitutive reference.

No solved displacement or equilibrium checkpoint enters this module. Surface
projection and a graph-harmonic extension only construct a contact-feasible seed.
"""

from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from joint_common import sha256, write_json
from joint_data import _collision_geometry
from joint_frozen_neutral import load_script

LOG = logging.getLogger(__name__)
repair = load_script("84-repair-eye-initialization.py")


def _audit(surface: np.ndarray, geometry: Any, eyes: Any) -> dict[str, Any]:
    soft = repair.poly(surface, geometry.soft_faces)
    result = {}
    for name in ("cranium", "mandible", "eyes"):
        obstacle = (
            repair.poly(eyes.points_m, eyes.triangles)
            if name == "eyes"
            else repair.poly(
                getattr(geometry, f"{name}_points_m"),
                getattr(geometry, f"{name}_faces"),
            )
        )
        pairs, *_ = _collision_geometry(soft, obstacle)
        result[f"soft_{name}_intersection_pairs"] = len(pairs)
    distance = repair.signed_distance(
        surface, repair.poly(eyes.points_m, eyes.triangles)
    )
    result["minimum_eye_node_signed_distance_m"] = float(distance.min())
    result["inside_eye_nodes"] = int(np.count_nonzero(distance < 0))
    return result


def _harmonic_extension(
    reference: np.ndarray,
    tets: np.ndarray,
    soft_ids: np.ndarray,
    fixed_ids: np.ndarray,
    surface: np.ndarray,
) -> np.ndarray:
    edges = np.concatenate(
        [tets[:, (a, b)] for a, b in itertools.combinations(range(4), 2)]
    )
    dm = np.transpose(reference[tets[:, 1:]] - reference[tets[:, :1]], (0, 2, 1))
    volume = np.linalg.det(dm) / 6
    assert np.all(volume > 0)
    weights = np.tile(volume, 6) / np.sum(
        (reference[edges[:, 0]] - reference[edges[:, 1]]) ** 2, axis=1
    )
    adjacency = sp.coo_matrix(
        (
            np.r_[weights, weights],
            (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]]),
        ),
        shape=(len(reference), len(reference)),
    ).tocsr()
    lap = sp.diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency
    boundary = np.union1d(soft_ids, fixed_ids)
    interior = np.setdiff1d(np.arange(len(reference)), boundary)
    matrix = lap[interior][:, interior].tocsr()
    coupling = lap[interior][:, boundary].tocsr()
    displacement = np.zeros_like(reference)
    displacement[soft_ids] = surface - reference[soft_ids]
    preconditioner = sp.diags(1 / matrix.diagonal())
    for axis in range(3):
        value, info = spla.cg(
            matrix,
            -(coupling @ displacement[boundary, axis]),
            M=preconditioner,
            rtol=1e-9,
            atol=0,
            maxiter=4000,
        )
        assert info == 0, info
        displacement[interior, axis] = value
    return displacement


def _separate_contact_planes(
    surface: np.ndarray,
    faces: np.ndarray,
    pairs: np.ndarray,
    obstacle: Any,
    clearance: float,
) -> None:
    """Project intersected faces past their local contacted obstacle planes."""
    rigid_faces = np.asarray(obstacle.faces).reshape(-1, 4)[:, 1:]
    normals = np.asarray(obstacle.cell_normals)
    for soft_face, rigid_face in pairs:
        ids = faces[soft_face]
        normal = normals[rigid_face]
        base = obstacle.points[rigid_faces[rigid_face, 0]]
        amount = np.maximum(clearance - (surface[ids] - base) @ normal, 0)
        surface[ids] += amount[:, None] * normal


def prepare_reference_seed(
    physics: Any,
    output_dir: Path,
    *,
    clearance_m: float = 1e-5,
    max_iterations: int = 32,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a FEM-only seed and save its independent geometry audit receipt.

    ``physics`` supplies ``points``, ``tets``, ``full_skull.geometry`` and ``eyes``.
    It may be a geometry-only namespace; no model or forward runtime is accessed.
    """
    assert clearance_m > 1e-8
    output_dir.mkdir(parents=True, exist_ok=False)
    geometry, eyes = physics.full_skull.geometry, physics.eyes
    reference = np.asarray(geometry.fem_reference_points_m).copy()
    assert np.array_equal(reference, physics.points)
    soft_ids = np.asarray(geometry.soft_global_ids)
    fixed = np.asarray(geometry.fixed_global_ids)
    assert not np.intersect1d(soft_ids, fixed).size
    surface = reference[soft_ids].copy()
    initial = _audit(surface, geometry, eyes)
    write_json(output_dir / "reference-audit.json", initial)
    LOG.info("Cold reference geometry: %s", initial)
    # Bone overlap cannot be hidden by an eye-only projection.
    assert initial["soft_cranium_intersection_pairs"] == 0
    assert initial["soft_mandible_intersection_pairs"] == 0
    obstacle = repair.poly(eyes.points_m, eyes.triangles).compute_normals(
        cell_normals=True,
        point_normals=False,
        auto_orient_normals=True,
        consistent_normals=True,
        split_vertices=False,
    )
    needs_repair = (
        initial["soft_eyes_intersection_pairs"] != 0
        or initial["minimum_eye_node_signed_distance_m"] <= 1e-8
    )
    iterations = 0
    bones = {
        name: repair.poly(
            getattr(geometry, f"{name}_points_m"), getattr(geometry, f"{name}_faces")
        ).compute_normals(
            cell_normals=True,
            point_normals=False,
            auto_orient_normals=True,
            consistent_normals=True,
            split_vertices=False,
        )
        for name in ("cranium", "mandible")
    }
    if needs_repair:
        for iteration in range(max_iterations):
            pairs, *_ = _collision_geometry(
                repair.poly(surface, geometry.soft_faces), obstacle
            )
            signed = repair.signed_distance(surface, obstacle)
            local = np.flatnonzero(signed < clearance_m)
            if len(pairs):
                local = np.unique(
                    np.r_[local, geometry.soft_faces[np.unique(pairs[:, 0])].ravel()]
                )
            changed = repair.project(surface, obstacle, local, clearance_m)
            _separate_contact_planes(
                surface, geometry.soft_faces, pairs, obstacle, clearance_m
            )
            bone_pairs = 0
            for bone in bones.values():
                crossed, *_ = _collision_geometry(
                    repair.poly(surface, geometry.soft_faces), bone
                )
                _separate_contact_planes(
                    surface, geometry.soft_faces, crossed, bone, clearance_m
                )
                crossed, *_ = _collision_geometry(
                    repair.poly(surface, geometry.soft_faces), bone
                )
                bone_pairs += len(crossed)
            pairs, *_ = _collision_geometry(
                repair.poly(surface, geometry.soft_faces), obstacle
            )
            signed = repair.signed_distance(surface, obstacle)
            iterations = iteration + 1
            LOG.info(
                "Reference eye projection %d: pairs=%d min_signed=%g moved=%d",
                iterations,
                len(pairs),
                signed.min(),
                changed,
            )
            if (
                not bone_pairs
                and not len(pairs)
                and signed.min() >= clearance_m * (1 - 1e-10)
            ):
                break
        else:
            raise RuntimeError("Reference eye projection failed to clear intersections")
        displacement = _harmonic_extension(
            reference, physics.tets, soft_ids, fixed, surface
        )
    else:
        displacement = np.zeros_like(reference)
    candidate = reference + displacement
    assert not np.any(displacement[fixed])
    final = _audit(candidate[soft_ids], geometry, eyes)
    determinants = repair.determinants(candidate, reference, physics.tets)
    final.update(
        finite=bool(np.isfinite(candidate).all()),
        detF_min=float(determinants.min()),
        detF_max=float(determinants.max()),
        inverted_tetrahedra=int(np.count_nonzero(determinants <= 0)),
        fixed_nodes_changed=False,
        maximum_displacement_mm=float(
            np.linalg.norm(displacement, axis=1).max() * 1000
        ),
    )
    collision_free = all(
        final[f"soft_{name}_intersection_pairs"] == 0
        for name in ("cranium", "mandible", "eyes")
    )
    success = (
        collision_free
        and final["finite"]
        and final["minimum_eye_node_signed_distance_m"] > 1e-8
    )
    seed_path = output_dir / "seed.npz"
    np.savez_compressed(seed_path, displacement_m=displacement)
    receipt = {
        "schema": "reference-cold-geometric-seed-v1",
        "success": success,
        "success_scope": "finite fixed-boundary contact-feasible geometric initializer; inversion counts are diagnostics and equilibrium is not certified",
        "origin": "zero displacement from existing constitutive reference",
        "prior_equilibrium_used": False,
        "forward_solves": 0,
        "fem_reference_rebased": False,
        "rigid_coordinates_changed": False,
        "method": "eye surface projection and graph-harmonic volume extension"
        if needs_repair
        else "unchanged zero displacement",
        "clearance_target_m": clearance_m,
        "projection_iterations": iterations,
        "initial": initial,
        "final": final,
        "seed": {"path": str(seed_path.resolve()), "sha256": sha256(seed_path)},
        "source_module": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__)),
        },
        "audit_scope": "all pure-soft triangles versus complete source cranium, mandible, eyes; bonded mixed FEM faces omitted as in collision model",
    }
    write_json(output_dir / "summary.json", receipt)
    assert success, final
    return displacement, receipt
