"""Clear the adopted neutral from rigid source eyes without rebasing FEM."""

from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path

import numpy as np
import pyvista as pv
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from joint_common import GROUP, ProfileJoint, archive_sources, sha256, write_json
from joint_data import _collision_geometry
from vtkmodules.vtkFiltersCore import vtkImplicitPolyDataDistance

from liblaf import cherries

LOG = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    frozen_neutral: Path = GROUP / "data/frozen-neutral-004"
    geometry: Path = GROUP / "data/simple-skin-forward-inputs-001/geometry.npz"
    eyes: Path = GROUP / "data/rigid-eyes-001"
    volume: Path = GROUP / "data/simple-skin-forward-inputs-001/prepared/volume.vtu"
    clearance_m: float = 1e-4
    max_iterations: int = 32
    output_dir: Path = cherries.output("eye-initialization-003", mkdir=True)


def record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def poly(points: np.ndarray, faces: np.ndarray) -> pv.PolyData:
    return pv.PolyData(points, np.column_stack((np.full(len(faces), 3), faces)))


def signed_distance(points: np.ndarray, obstacle: pv.PolyData) -> np.ndarray:
    distance = vtkImplicitPolyDataDistance()
    distance.SetInput(obstacle)
    return np.asarray([distance.EvaluateFunction(point) for point in points])


def project(
    points: np.ndarray, obstacle: pv.PolyData, ids: np.ndarray, clearance: float
) -> int:
    """Move selected free vertices outward from an oriented closed obstacle."""
    distance = vtkImplicitPolyDataDistance()
    distance.SetInput(obstacle)
    moved = 0
    for index in np.unique(ids):
        closest = [0.0, 0.0, 0.0]
        value = float(
            distance.EvaluateFunctionAndGetClosestPoint(points[index], closest)
        )
        if value >= clearance:
            continue
        closest_a = np.asarray(closest)
        direction = points[index] - closest_a
        length = float(np.linalg.norm(direction))
        if length < 1e-14:
            # Point normals are consistently oriented by VTK for this closed eye mesh.
            cell = obstacle.find_closest_cell(points[index])
            direction = np.asarray(obstacle.cell_normals[cell])
        else:
            direction /= length
            if value < 0:
                direction *= -1
        points[index] += (clearance - value) * direction
        moved += 1
    return moved


def separate_faces(
    points: np.ndarray,
    soft_faces: np.ndarray,
    pairs: np.ndarray,
    rigid_points: np.ndarray,
    rigid_components: np.ndarray,
    triangle_components: np.ndarray,
    clearance: float,
) -> int:
    """Put each crossed soft face outside a radial source-eye support plane."""
    changed = 0
    for soft_cell, rigid_cell in pairs:
        ids = soft_faces[soft_cell]
        component = triangle_components[rigid_cell]
        eye_points = rigid_points[rigid_components == component]
        center = eye_points.mean(axis=0)
        normal = points[ids].mean(axis=0) - center
        normal /= np.linalg.norm(normal)
        target = float((eye_points @ normal).max() + clearance)
        amount = np.maximum(target - points[ids] @ normal, 0.0)
        points[ids] += amount[:, None] * normal
        changed += int(np.count_nonzero(amount))
    return changed


def determinants(
    points: np.ndarray, reference: np.ndarray, tets: np.ndarray
) -> np.ndarray:
    dm = np.transpose(reference[tets[:, 1:]] - reference[tets[:, :1]], (0, 2, 1))
    ds = np.transpose(points[tets[:, 1:]] - points[tets[:, :1]], (0, 2, 1))
    return np.linalg.det(ds @ np.linalg.inv(dm))


def main(cfg: Config) -> None:  # noqa: PLR0915
    assert cfg.clearance_m > 1e-8
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    archive_sources(cfg.output_dir)
    neutral_manifest = json.loads((cfg.frozen_neutral / "manifest.json").read_text())
    assert neutral_manifest["success"] is True
    assert (
        sha256(cfg.frozen_neutral / "state.npz")
        == neutral_manifest["artifacts"]["state.npz"]["sha256"]
    )
    eye_manifest = json.loads((cfg.eyes / "manifest.json").read_text())
    assert eye_manifest["success"] is True
    assert eye_manifest["source_coordinates_changed"] is False
    assert (
        sha256(cfg.eyes / "eyes.npz") == eye_manifest["artifacts"]["eyes.npz"]["sha256"]
    )
    eye_arrays = np.load(cfg.eyes / "eyes.npz")
    state = np.load(cfg.frozen_neutral / "state.npz")
    initial = np.asarray(state["neutral_displacement_m"], dtype=np.float64)
    with np.load(cfg.geometry) as source:
        geometry = {key: source[key] for key in source.files}
    reference = np.asarray(geometry["fem_reference_points_m"], dtype=np.float64)
    candidate = np.asarray(state["neutral_points_m"], dtype=np.float64)
    assert np.array_equal(candidate, reference + initial)
    soft_ids = np.asarray(geometry["soft_global_ids"], dtype=np.int64)
    soft_faces = np.asarray(geometry["soft_faces"], dtype=np.int64)
    fixed = np.asarray(geometry["fixed_global_ids"], dtype=np.int64)
    fixed_mask = np.zeros(len(reference), bool)
    fixed_mask[fixed] = True
    assert not np.any(fixed_mask[soft_ids])
    eyes = pv.read(cfg.eyes / "eyes.vtp").compute_normals(
        cell_normals=True,
        point_normals=False,
        auto_orient_normals=True,
        consistent_normals=True,
        split_vertices=False,
    )
    # Keep the original source-face order for collision geometry; normals only guide projection.
    assert np.array_equal(
        np.asarray(eyes.points), np.load(cfg.eyes / "eyes.npz")["points_m"]
    )
    surface_points = candidate[soft_ids].copy()
    surface = poly(surface_points, soft_faces)
    initial_pairs, *_ = _collision_geometry(surface, eyes)
    initial_signed = signed_distance(surface_points, eyes)
    initial_fixed_conflicts = int(
        np.intersect1d(
            soft_ids[np.flatnonzero(initial_signed < cfg.clearance_m)], fixed
        ).size
    )
    assert initial_fixed_conflicts == 0

    # Correct every intersecting triangle, then spread the correction through the
    # volume with the same graph-harmonic extension used for full-skull repair.
    for iteration in range(cfg.max_iterations):
        surface = poly(surface_points, soft_faces)
        pairs, *_ = _collision_geometry(surface, eyes)
        signed = signed_distance(surface_points, eyes)
        local = np.flatnonzero(signed < cfg.clearance_m)
        if len(pairs):
            local = np.unique(
                np.concatenate((local, soft_faces[np.unique(pairs[:, 0])].ravel()))
            )
        changed = project(surface_points, eyes, local, cfg.clearance_m)
        changed += separate_faces(
            surface_points,
            soft_faces,
            pairs,
            np.asarray(eye_arrays["points_m"]),
            np.asarray(eye_arrays["vertex_component_ids"]),
            np.asarray(eye_arrays["triangle_component_ids"]),
            cfg.clearance_m,
        )
        surface = poly(surface_points, soft_faces)
        pairs, *_ = _collision_geometry(surface, eyes)
        signed = signed_distance(surface_points, eyes)
        LOG.info(
            "eye projection %d: pairs=%d min_signed=%.9g moved=%d",
            iteration + 1,
            len(pairs),
            signed.min(),
            changed,
        )
        if len(pairs) == 0 and float(signed.min()) >= cfg.clearance_m * (1 - 1e-10):
            break
    else:
        message = "eye surface projection did not clear all source-eye intersections"
        raise RuntimeError(message)

    volume = pv.read(cfg.volume)
    tets = np.asarray(volume.cells).reshape(-1, 5)[:, 1:].astype(np.int64)
    dm = np.transpose(reference[tets[:, 1:]] - reference[tets[:, :1]], (0, 2, 1))
    weights = np.tile(np.linalg.det(dm) / 6, 6) / np.sum(
        (
            reference[
                np.concatenate(
                    [tets[:, (a, b)] for a, b in itertools.combinations(range(4), 2)]
                )[:, 0]
            ]
            - reference[
                np.concatenate(
                    [tets[:, (a, b)] for a, b in itertools.combinations(range(4), 2)]
                )[:, 1]
            ]
        )
        ** 2,
        axis=1,
    )
    edges = np.concatenate(
        [tets[:, (a, b)] for a, b in itertools.combinations(range(4), 2)]
    )
    adjacency = sp.coo_matrix(
        (
            np.r_[weights, weights],
            (np.r_[edges[:, 0], edges[:, 1]], np.r_[edges[:, 1], edges[:, 0]]),
        ),
        shape=(len(reference), len(reference)),
    ).tocsr()
    lap = sp.diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency
    boundary = np.union1d(soft_ids, fixed)
    interior = np.setdiff1d(np.arange(len(reference)), boundary)
    matrix = lap[interior][:, interior].tocsr()
    coupling = lap[interior][:, boundary].tocsr()
    correction = np.zeros_like(reference)
    correction[soft_ids] = surface_points - candidate[soft_ids]
    preconditioner = sp.diags(1 / matrix.diagonal())
    for axis in range(3):
        value, info = spla.cg(
            matrix,
            -(coupling @ correction[boundary, axis]),
            M=preconditioner,
            rtol=1e-9,
            atol=0,
            maxiter=4000,
        )
        assert info == 0, info
        correction[interior, axis] = value
    repaired = candidate + correction
    assert np.array_equal(repaired[fixed], candidate[fixed])
    repaired_surface = poly(repaired[soft_ids], soft_faces)
    final_pairs, *_ = _collision_geometry(repaired_surface, eyes)
    final_signed = signed_distance(repaired[soft_ids], eyes)
    # Whole-face projection is repeated after volume extension is unnecessary:
    # the extension leaves boundary values exactly as projected.
    assert len(final_pairs) == 0
    assert float(final_signed.min()) >= cfg.clearance_m * (1 - 1e-10)
    det = determinants(repaired, reference, tets)
    seed = cfg.output_dir / "seed.npz"
    np.savez_compressed(seed, displacement_m=repaired - reference)
    summary = {
        "schema": "joint-eye-initialization-v1",
        "success": True,
        "seed": record(seed),
        "frozen_neutral": record(cfg.frozen_neutral / "state.npz"),
        "geometry": record(cfg.geometry),
        "eyes": record(cfg.eyes / "eyes.npz"),
        "method": "free soft-surface closest-source-eye projection with graph-harmonic interior extension; FEM reference and rigid coordinates unchanged",
        "clearance_target_m": cfg.clearance_m,
        "initial": {
            "soft_eye_intersection_pairs": len(initial_pairs),
            "soft_eye_faces": len(np.unique(initial_pairs[:, 0])),
            "minimum_node_signed_distance_m": float(initial_signed.min()),
            "fixed_node_conflicts": initial_fixed_conflicts,
        },
        "final": {
            "soft_eye_intersection_pairs": len(final_pairs),
            "minimum_node_signed_distance_m": float(final_signed.min()),
            "fixed_nodes_changed": bool(np.any(correction[fixed] != 0)),
            "finite": bool(np.isfinite(repaired).all()),
            "detF_min": float(det.min()),
            "detF_max": float(det.max()),
            "inverted_tetrahedra": int(np.sum(det <= 0)),
            "maximum_repair_mm": float(np.linalg.norm(correction, axis=1).max() * 1000),
            "moved_soft_nodes": int(
                np.sum(np.linalg.norm(correction[soft_ids], axis=1) > 0)
            ),
        },
        "rigid_coordinates_changed": False,
        "fem_reference_rebased": False,
        "ipc_clearance_requirement_m": 1e-8,
        "ipc_clearance_satisfied": float(final_signed.min()) > 1e-8,
        "provenance": record(cfg.output_dir / "provenance.json"),
    }
    write_json(cfg.output_dir / "summary.json", summary)
    cherries.log_metrics(
        {
            "initial/pairs": len(initial_pairs),
            "final/pairs": len(final_pairs),
            "final/min_signed_um": final_signed.min() * 1e6,
            "final/inverted_tetrahedra": summary["final"]["inverted_tetrahedra"],
        }
    )
    cherries.log_output(cfg.output_dir)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
