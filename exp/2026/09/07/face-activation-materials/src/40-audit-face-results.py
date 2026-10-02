# ruff: noqa: EM101, EM102, TRY003
"""CPU-only geometric and surface-frequency audit for saved face equilibria."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ipctk
import numpy as np
import pydantic_settings as ps
import pyvista as pv
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from experiment_profile import ProfileCometNoCommit
from scipy.sparse import csgraph

from liblaf import cherries

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
REPO_ROOT = HERE.parents[5]
LOG = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    """Input result directories and Cherries-managed audit output."""

    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)

    fixture: Path = cherries.input("10-fixture")
    result_dirs: str = "data/20-pilot"
    endpoint_name: str = "final.vtu"
    output_dir: Path = cherries.output("40-audit-pilot", mkdir=True)
    scales_mm: str = "2,5,10"
    top_k: int = 25
    render: bool = True


@dataclass(frozen=True)
class SurfaceOperators:
    """Rest-surface geometry, FEM mass, cotangent stiffness, and masks."""

    points: np.ndarray
    triangles: np.ndarray
    global_ids: np.ndarray
    group_ids: np.ndarray
    group_names: tuple[str, ...]
    mass: np.ndarray
    stiffness: sp.csr_matrix
    normals: np.ndarray
    boundary: np.ndarray
    boundary_edges: int
    mouth: np.ndarray
    lip_seed: np.ndarray
    lip_distance: np.ndarray


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def triangles(poly: pv.PolyData) -> np.ndarray:
    encoded = np.asarray(poly.faces, dtype=np.int64).reshape(-1, 4)
    if encoded.shape[0] != poly.n_cells or np.any(encoded[:, 0] != 3):
        raise ValueError("expected an all-triangle PolyData")
    return encoded[:, 1:]


def tets(mesh: pv.UnstructuredGrid) -> np.ndarray:
    encoded = np.asarray(mesh.cells, dtype=np.int64).reshape(-1, 5)
    if encoded.shape[0] != mesh.n_cells or np.any(encoded[:, 0] != 4):
        raise ValueError("expected an all-tetrahedron UnstructuredGrid")
    return encoded[:, 1:]


def weighted_quantiles(
    values: np.ndarray,
    weights: np.ndarray,
    probabilities: tuple[float, ...] = (
        0,
        0.0001,
        0.001,
        0.01,
        0.05,
        0.5,
        0.95,
        0.99,
        0.999,
        0.9999,
        1,
    ),
) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not np.any(valid):
        raise ValueError("weighted quantile has no positive finite support")
    value = values[valid]
    weight = weights[valid]
    order = np.argsort(value)
    value = value[order]
    cumulative = np.cumsum(weight[order])
    cumulative = (cumulative - 0.5 * weight[order]) / cumulative[-1]
    result = np.interp(np.asarray(probabilities), cumulative, value)
    return {
        f"q{probability:g}": float(item)
        for probability, item in zip(probabilities, result, strict=True)
    }


def scalar_stats(
    values: np.ndarray, mass: np.ndarray, mask: np.ndarray
) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    selected_mass = np.where(mask, mass, 0.0)
    total = float(selected_mass.sum())
    if total <= 0.0:
        raise ValueError("surface statistic mask has zero area")
    rms = math.sqrt(float(np.sum(selected_mass * values**2) / total))
    mean = float(np.sum(selected_mass * values) / total)
    absolute = np.abs(values)
    quantiles = weighted_quantiles(absolute, selected_mass, (0.5, 0.95, 0.99, 1.0))
    return {
        "area_m2": total,
        "mean_mm": 1000.0 * mean,
        "rms_mm": 1000.0 * rms,
        "abs_median_mm": 1000.0 * quantiles["q0.5"],
        "abs_q95_mm": 1000.0 * quantiles["q0.95"],
        "abs_q99_mm": 1000.0 * quantiles["q0.99"],
        "abs_max_mm": 1000.0 * quantiles["q1"],
    }


def cotangent_operators(skin: pv.PolyData) -> SurfaceOperators:
    points = np.asarray(skin.points, dtype=np.float64)
    tri = triangles(skin)
    p0, p1, p2 = points[tri[:, 0]], points[tri[:, 1]], points[tri[:, 2]]
    cross = np.cross(p1 - p0, p2 - p0)
    double_area = np.linalg.norm(cross, axis=1)
    if np.any(double_area <= 0.0):
        raise ValueError("skin contains a degenerate rest triangle")
    area = 0.5 * double_area
    mass = np.zeros(skin.n_points, dtype=np.float64)
    np.add.at(mass, tri.ravel(), np.repeat(area / 3.0, 3))
    if np.any(mass <= 0.0):
        raise ValueError("skin contains an isolated or zero-area vertex")

    normals = np.zeros((skin.n_points, 3), dtype=np.float64)
    for column in range(3):
        np.add.at(normals, tri[:, column], cross)
    normal_length = np.linalg.norm(normals, axis=1)
    if np.any(normal_length <= 0.0):
        raise ValueError("skin vertex normal is undefined")
    normals /= normal_length[:, None]

    cot0 = np.einsum("ij,ij->i", p1 - p0, p2 - p0) / double_area
    cot1 = np.einsum("ij,ij->i", p2 - p1, p0 - p1) / double_area
    cot2 = np.einsum("ij,ij->i", p0 - p2, p1 - p2) / double_area
    edge_i = np.concatenate((tri[:, 1], tri[:, 2], tri[:, 0]))
    edge_j = np.concatenate((tri[:, 2], tri[:, 0], tri[:, 1]))
    edge_w = 0.5 * np.concatenate((cot0, cot1, cot2))
    rows = np.concatenate((edge_i, edge_j, edge_i, edge_j))
    cols = np.concatenate((edge_j, edge_i, edge_i, edge_j))
    data = np.concatenate((-edge_w, -edge_w, edge_w, edge_w))
    stiffness = sp.coo_matrix(
        (data, (rows, cols)), shape=(skin.n_points, skin.n_points)
    ).tocsr()
    if np.max(np.abs(np.asarray(stiffness.sum(axis=1)).reshape(-1))) > 1e-10:
        raise ValueError("cotangent stiffness does not preserve constants")

    raw_edges = np.sort(
        np.concatenate((tri[:, (0, 1)], tri[:, (1, 2)], tri[:, (2, 0)])), axis=1
    )
    unique_edges, counts = np.unique(raw_edges, axis=0, return_counts=True)
    if np.any((counts != 1) & (counts != 2)):
        raise ValueError("skin has a nonmanifold edge")
    boundary_ids = np.unique(unique_edges[counts == 1])
    boundary = np.zeros(skin.n_points, dtype=bool)
    boundary[boundary_ids] = True

    names = tuple(
        str(value) for value in np.asarray(skin.field_data["GroupName"]).reshape(-1)
    )
    group_ids = np.asarray(skin.point_data["GroupId"], dtype=np.int64)
    if np.any(group_ids < 0) or np.any(group_ids >= len(names)):
        raise ValueError("skin GroupId contains an invalid field-data index")
    lip_seed = np.asarray([names[index].startswith("Lip") for index in group_ids])
    if not np.any(lip_seed):
        raise ValueError("corrected skin has no Lip* seed vertices")
    edge_length = np.linalg.norm(
        points[unique_edges[:, 0]] - points[unique_edges[:, 1]], axis=1
    )
    graph = sp.coo_matrix(
        (
            np.concatenate((edge_length, edge_length)),
            (
                np.concatenate((unique_edges[:, 0], unique_edges[:, 1])),
                np.concatenate((unique_edges[:, 1], unique_edges[:, 0])),
            ),
        ),
        shape=(skin.n_points, skin.n_points),
    ).tocsr()
    lip_distance = np.asarray(
        csgraph.dijkstra(
            graph, directed=False, indices=np.flatnonzero(lip_seed), min_only=True
        )
    )
    mouth = lip_distance <= 0.010
    global_ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    return SurfaceOperators(
        points=points,
        triangles=tri,
        global_ids=global_ids,
        group_ids=group_ids,
        group_names=names,
        mass=mass,
        stiffness=stiffness,
        normals=normals,
        boundary=boundary,
        boundary_edges=int(np.count_nonzero(counts == 1)),
        mouth=mouth,
        lip_seed=lip_seed,
        lip_distance=lip_distance,
    )


def diffuse_scalar(
    scalar: np.ndarray, ops: SurfaceOperators, scale_m: float
) -> tuple[np.ndarray, np.ndarray]:
    # For the 2-D heat kernel, RMS radius sqrt(4t) equals the named scale.
    time = 0.25 * scale_m**2
    system = sp.diags(ops.mass) + time * ops.stiffness
    smooth = spla.spsolve(system.tocsc(), ops.mass * scalar)
    if np.any(~np.isfinite(smooth)):
        raise RuntimeError("cotangent diffusion returned a non-finite scalar")
    return smooth, scalar - smooth


def deformation_fields(
    rest: np.ndarray, final: np.ndarray, tetrahedra: np.ndarray, chunk: int = 50_000
) -> tuple[np.ndarray, np.ndarray]:
    count = tetrahedra.shape[0]
    detf = np.empty(count, dtype=np.float64)
    stretch = np.empty((count, 3), dtype=np.float64)
    for start in range(0, count, chunk):
        ids = tetrahedra[start : start + chunk]
        dm = np.transpose(rest[ids[:, 1:]] - rest[ids[:, :1]], (0, 2, 1))
        ds = np.transpose(final[ids[:, 1:]] - final[ids[:, :1]], (0, 2, 1))
        F = ds @ np.linalg.inv(dm)
        detf[start : start + len(ids)] = np.linalg.det(F)
        stretch[start : start + len(ids)] = np.sort(
            np.linalg.svd(F, compute_uv=False), axis=1
        )
    return detf, stretch


def activation_fields(result: pv.UnstructuredGrid) -> tuple[np.ndarray, np.ndarray]:
    packed = np.asarray(result.cell_data["ActivationInverseMatrix"], dtype=np.float64)
    if packed.shape != (result.n_cells, 9):
        raise ValueError("ActivationInverseMatrix must have shape (n_cells, 9)")
    matrices = packed.reshape(-1, 3, 3)
    symmetry_error = float(np.max(np.abs(matrices - matrices.transpose(0, 2, 1))))
    if symmetry_error > 1e-10:
        raise ValueError("ActivationInverseMatrix is not symmetric")
    eigenvalues = np.linalg.eigvalsh(matrices)
    determinant = np.linalg.det(matrices)
    return eigenvalues, determinant


def endpoint_file_consistency(
    fixture: pv.UnstructuredGrid,
    result: pv.UnstructuredGrid,
    result_dir: Path,
) -> dict[str, Any]:
    npz_path = result_dir / "final.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(f"missing endpoint array archive: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as saved:
        missing = {"q", "u", "Ainv"} - set(saved.files)
        if missing:
            raise ValueError(f"endpoint NPZ lacks required arrays: {sorted(missing)}")
        q = np.asarray(saved["q"], dtype=np.float64)
        u = np.asarray(saved["u"], dtype=np.float64)
        ainv = np.asarray(saved["Ainv"], dtype=np.float64)
        keys = sorted(saved.files)
    active = np.asarray(fixture.cell_data["ActivationMask"], dtype=bool)
    fixed = np.asarray(fixture.point_data["IsFixed"], dtype=bool)
    packed = np.asarray(
        result.cell_data["ActivationInverseMatrix"], dtype=np.float64
    ).reshape(-1, 3, 3)
    rest = np.asarray(result.point_data["RestPosition"], dtype=np.float64)
    points = np.asarray(result.points, dtype=np.float64)
    if u.shape != points.shape:
        raise ValueError("endpoint NPZ displacement shape differs from VTU points")
    if ainv.shape != (int(np.count_nonzero(active)), 3, 3):
        raise ValueError("endpoint NPZ activation shape differs from active-cell count")
    if (
        not np.isfinite(q).all()
        or not np.isfinite(u).all()
        or not np.isfinite(ainv).all()
    ):
        raise FloatingPointError("endpoint NPZ contains a non-finite array")
    if not np.array_equal(rest + u, points):
        raise ValueError("endpoint NPZ displacement differs from VTU coordinates")
    if not np.array_equal(ainv, packed[active]):
        raise ValueError("endpoint NPZ activation differs from active VTU cells")
    if not np.array_equal(
        packed[~active], np.broadcast_to(np.eye(3), packed[~active].shape)
    ):
        raise ValueError("inactive VTU activation matrices differ from identity")
    fixed_max = float(np.max(np.abs(u[fixed])))
    if fixed_max != 0.0:
        raise ValueError("endpoint NPZ moves a fixed vertex")
    return {
        "npz_path": str(npz_path),
        "npz_sha256": sha256(npz_path),
        "keys": keys,
        "q_shape": list(q.shape),
        "u_shape": list(u.shape),
        "Ainv_shape": list(ainv.shape),
        "q_sha256": array_sha256(q),
        "u_sha256": array_sha256(u),
        "Ainv_sha256": array_sha256(ainv),
        "coordinates_exact": True,
        "active_activation_exact": True,
        "inactive_activation_identity_exact": True,
        "fixed_vertices": int(np.count_nonzero(fixed)),
        "fixed_displacement_max_abs_m": fixed_max,
    }


def cell_record(
    cell_id: int,
    mesh: pv.UnstructuredGrid,
    tetrahedra: np.ndarray,
    detf: np.ndarray,
    stretch: np.ndarray,
    activation_eigen: np.ndarray,
    det_activation: np.ndarray,
    ops: SurfaceOperators,
) -> dict[str, Any]:
    vertex_ids = tetrahedra[cell_id]
    fixed = np.asarray(mesh.point_data["IsFixed"], dtype=bool)[vertex_ids]
    cut = np.asarray(mesh.point_data["ArtificialCutIncident"], dtype=bool)[vertex_ids]
    muscle_id = int(mesh.cell_data["MuscleId"][cell_id])
    names = tuple(
        str(value) for value in np.asarray(mesh.field_data["MuscleName"]).reshape(-1)
    )
    fractions = {
        name: float(mesh.cell_data[f"{name.title()}Fraction"][cell_id])
        for name in ("fat", "muscle", "aponeurosis")
    }
    control_id = int(mesh.cell_data["ActivationControlId"][cell_id])
    center = np.asarray(mesh.points)[tetrahedra[cell_id]].mean(axis=0)
    nearest = int(np.argmin(np.linalg.norm(ops.points - center, axis=1)))
    nearest_group = int(ops.group_ids[nearest])
    return {
        "cell_id": int(cell_id),
        "vertex_ids": vertex_ids.tolist(),
        "fixed_incidence": {
            "count": int(np.count_nonzero(fixed)),
            "vertex_ids": vertex_ids[fixed].tolist(),
        },
        "artificial_cut_incidence": {
            "count": int(np.count_nonzero(cut)),
            "vertex_ids": vertex_ids[cut].tolist(),
        },
        "center_m": center.tolist(),
        "nearest_visible_skin": {
            "distance_mm": 1000.0 * float(np.linalg.norm(ops.points[nearest] - center)),
            "global_point_id": int(ops.global_ids[nearest]),
            "group_id": nearest_group,
            "group_name": ops.group_names[nearest_group],
            "intrinsic_distance_to_lip_mm": 1000.0 * float(ops.lip_distance[nearest]),
        },
        "detF": float(detf[cell_id]),
        "principal_stretches": stretch[cell_id].tolist(),
        "detAinv": float(det_activation[cell_id]),
        "activation_eigenvalues": activation_eigen[cell_id].tolist(),
        "fractions": fractions,
        "dominant_material": max(fractions, key=fractions.get),
        "muscle_id": muscle_id,
        "muscle_name": names[muscle_id] if muscle_id >= 0 else None,
        "activation_control_id": control_id,
        "selected_expression_activation": control_id >= 0,
        "fiber_confidence": float(mesh.cell_data["ActivationFiberConfidence"][cell_id]),
    }


def volume_diagnostics(
    fixture: pv.UnstructuredGrid,
    result: pv.UnstructuredGrid,
    ops: SurfaceOperators,
    top_k: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    tetrahedra = tets(fixture)
    rest = np.asarray(fixture.points, dtype=np.float64)
    if "RestPosition" not in result.point_data or not np.array_equal(
        np.asarray(result.point_data["RestPosition"], dtype=np.float64), rest
    ):
        raise ValueError("result RestPosition differs from the fixture")
    final = np.asarray(result.points, dtype=np.float64)
    detf, stretch = deformation_fields(rest, final, tetrahedra)
    activation_eigen, det_activation = activation_fields(result)
    detg = detf * det_activation
    volume = np.asarray(fixture.cell_data["Volume"], dtype=np.float64)
    if "DetF" in result.cell_data:
        saved = np.asarray(result.cell_data["DetF"], dtype=np.float64)
        saved_error = float(np.max(np.abs(saved - detf)))
        if saved_error > 1e-10:
            raise ValueError("saved DetF differs from geometry recomputation")
    else:
        saved_error = None

    minimum_order = np.argsort(detf)[:top_k]
    records = [
        cell_record(
            int(cell_id),
            fixture,
            tetrahedra,
            detf,
            stretch,
            activation_eigen,
            det_activation,
            ops,
        )
        for cell_id in minimum_order
    ]
    active = np.asarray(fixture.cell_data["ActivationMask"], dtype=bool)
    low1000 = np.argsort(detf)[: min(1000, fixture.n_cells)]
    thresholds = {
        f"below_{threshold:g}": int(np.count_nonzero(detf < threshold))
        for threshold in (0.0, 0.25, 0.5, 0.75, 0.8, 0.9)
    }
    summary = {
        "detF": {
            "volume_weighted_quantiles": weighted_quantiles(detf, volume),
            "counts": thresholds,
            "saved_recompute_max_abs_error": saved_error,
        },
        "principal_stretch_min": weighted_quantiles(stretch[:, 0], volume),
        "principal_stretch_mid": weighted_quantiles(stretch[:, 1], volume),
        "principal_stretch_max": weighted_quantiles(stretch[:, 2], volume),
        "detG": {
            "volume_weighted_quantiles": weighted_quantiles(detg, volume),
            "nonpositive": int(np.count_nonzero(detg <= 0.0)),
        },
        "activation": {
            "eigen_min": weighted_quantiles(activation_eigen[:, 0], volume),
            "eigen_mid": weighted_quantiles(activation_eigen[:, 1], volume),
            "eigen_max": weighted_quantiles(activation_eigen[:, 2], volume),
            "detAinv": weighted_quantiles(det_activation, volume),
            "nonpositive_eigen_cells": int(
                np.count_nonzero(activation_eigen[:, 0] <= 0.0)
            ),
        },
        "lowest_detF_cells": records,
        "minimum_cell": records[0],
        "lowest_1000_selected_expression_fraction": float(active[low1000].mean()),
        "cells_below_0.8_selected_expression_fraction": (
            float(active[detf < 0.8].mean()) if np.any(detf < 0.8) else None
        ),
    }
    fields = {
        "DetFRecomputed": detf,
        "StretchMin": stretch[:, 0],
        "StretchMid": stretch[:, 1],
        "StretchMax": stretch[:, 2],
        "ActivationEigenMin": activation_eigen[:, 0],
        "ActivationEigenMid": activation_eigen[:, 1],
        "ActivationEigenMax": activation_eigen[:, 2],
        "DetAinvRecomputed": det_activation,
        "DetG": detg,
    }
    return summary, fields


def intersection_keys(
    points: np.ndarray, faces: np.ndarray, global_ids: np.ndarray
) -> tuple[bool, set[tuple[int, int, int, int, int]], dict[str, Any]]:
    vertices = np.asfortranarray(points, dtype=np.float64)
    faces32 = np.asfortranarray(faces, dtype=np.int32)
    edges = np.asfortranarray(ipctk.edges(faces32), dtype=np.int32)
    mesh = ipctk.CollisionMesh(vertices, edges, faces32)
    has_intersections = bool(ipctk.has_intersections(mesh, vertices))
    broad_phase = ipctk.LBVH()
    broad_phase.can_vertices_collide = mesh.can_collide
    inflation = 1e-6 * float(np.linalg.norm(np.ptp(vertices, axis=0)))
    broad_phase.build(vertices, edges, faces32, inflation)
    candidates = broad_phase.detect_edge_face_candidates()
    hits: set[tuple[int, int, int, int, int]] = set()
    for candidate in candidates:
        edge = edges[candidate.edge_id]
        face = faces32[candidate.face_id]
        if ipctk.is_edge_intersecting_triangle(
            vertices[edge[0]],
            vertices[edge[1]],
            vertices[face[0]],
            vertices[face[1]],
            vertices[face[2]],
        ):
            edge_global = np.sort(global_ids[edge])
            face_global = np.sort(global_ids[face])
            hits.add((*map(int, edge_global), *map(int, face_global)))
    if has_intersections != bool(hits):
        raise RuntimeError("IPC boolean and enumerated edge-face hits disagree")
    return (
        has_intersections,
        hits,
        {
            "vertices": len(vertices),
            "edges": len(edges),
            "triangles": len(faces),
            "inflation_radius_m": inflation,
            "broad_phase_candidates": len(candidates),
        },
    )


def intersection_diagnostics(
    fixture: pv.UnstructuredGrid,
    final: np.ndarray,
    ops: SurfaceOperators,
) -> dict[str, Any]:
    full = fixture.extract_surface(algorithm=None).triangulate()
    full_global = np.asarray(full.point_data["GlobalPointId"], dtype=np.int64)
    full_faces = triangles(full)
    domains = {
        "complete_tet_boundary": (
            np.asarray(fixture.points)[full_global],
            final[full_global],
            full_faces,
            full_global,
        ),
        "visible_IsFace_skin": (
            ops.points,
            final[ops.global_ids],
            ops.triangles,
            ops.global_ids,
        ),
    }
    output: dict[str, Any] = {}
    for name, (rest, deformed, faces, global_ids) in domains.items():
        rest_bool, rest_hits, topology = intersection_keys(rest, faces, global_ids)
        final_bool, final_hits, _ = intersection_keys(deformed, faces, global_ids)
        output[name] = {
            **topology,
            "rest_has_intersections": rest_bool,
            "deformed_has_intersections": final_bool,
            "rest_edge_face_hits": len(rest_hits),
            "deformed_edge_face_hits": len(final_hits),
            "new_edge_face_hits": len(final_hits - rest_hits),
            "resolved_edge_face_hits": len(rest_hits - final_hits),
            "rest_hit_keys": [list(key) for key in sorted(rest_hits)],
            "deformed_hit_keys": [list(key) for key in sorted(final_hits)],
        }
    return {
        "method": "IPC Toolkit static LBVH edge-face intersections",
        "ipctk_version": ipctk.__version__,
        "adjacency_policy": "CollisionMesh rejects edge-face pairs sharing a vertex",
        "temporal_scope": "endpoint only; no continuous collision detection between optimizer states",
        "domains": output,
    }


def surface_diagnostics(
    fixture: pv.UnstructuredGrid,
    result: pv.UnstructuredGrid,
    skin: pv.PolyData,
    ops: SurfaceOperators,
    scales_m: tuple[float, ...],
) -> tuple[dict[str, Any], pv.PolyData]:
    displacement = np.asarray(result.points, dtype=np.float64) - np.asarray(
        fixture.points, dtype=np.float64
    )
    target = np.asarray(fixture.point_data["Smile"], dtype=np.float64)
    if not np.isfinite(target[ops.global_ids]).all():
        raise ValueError("visible IsFace skin contains a non-finite Smile target")
    u = displacement[ops.global_ids]
    target_skin = target[ops.global_ids]
    residual = u - target_skin
    normal_displacement = np.einsum("ij,ij->i", u, ops.normals)
    normal_residual = np.einsum("ij,ij->i", residual, ops.normals)
    total_mass = float(ops.mass.sum())
    target_norm2 = float(np.sum(ops.mass[:, None] * target_skin**2))
    projection = float(np.sum(ops.mass[:, None] * u * target_skin) / target_norm2)
    target_rms = math.sqrt(target_norm2 / total_mass)
    orthogonal = u - projection * target_skin

    output_skin = skin.copy(deep=True)
    output_skin.points = np.asarray(result.points)[ops.global_ids]
    output_skin.point_data["RestPosition"] = ops.points
    output_skin.point_data["Displacement"] = u
    output_skin.point_data["TargetDisplacement"] = target_skin
    output_skin.point_data["RestNormal"] = ops.normals
    output_skin.point_data["NormalDisplacementMm"] = 1000.0 * normal_displacement
    output_skin.point_data["NormalResidualMm"] = 1000.0 * normal_residual
    output_skin.point_data["LipSeed"] = ops.lip_seed.astype(np.int8)
    output_skin.point_data["MouthROI10mm"] = ops.mouth.astype(np.int8)
    output_skin.point_data["MembraneBoundary"] = ops.boundary.astype(np.int8)
    output_skin.point_data["IntrinsicDistanceToLipMm"] = 1000.0 * ops.lip_distance

    scales: dict[str, Any] = {}
    for scale in scales_m:
        label = f"{scale * 1000:g}mm"
        smooth_u, high_u = diffuse_scalar(normal_displacement, ops, scale)
        smooth_r, high_r = diffuse_scalar(normal_residual, ops, scale)
        output_skin.point_data[f"NormalDisplacementLowPass{label}"] = 1000.0 * smooth_u
        output_skin.point_data[f"NormalDisplacementHighPass{label}"] = 1000.0 * high_u
        output_skin.point_data[f"NormalResidualLowPass{label}"] = 1000.0 * smooth_r
        output_skin.point_data[f"NormalResidualHighPass{label}"] = 1000.0 * high_r
        scales[label] = {
            "heat_time_m2": 0.25 * scale**2,
            "normal_displacement_highpass": {
                "full_face": scalar_stats(
                    high_u, ops.mass, np.ones(len(ops.mass), bool)
                ),
                "mouth_10mm": scalar_stats(high_u, ops.mass, ops.mouth),
            },
            "normal_residual_highpass": {
                "full_face": scalar_stats(
                    high_r, ops.mass, np.ones(len(ops.mass), bool)
                ),
                "mouth_10mm": scalar_stats(high_r, ops.mass, ops.mouth),
            },
        }
    summary = {
        "operator": {
            "mass": "lumped triangle area, one third per incident vertex",
            "stiffness": "piecewise-linear cotangent FEM stiffness on rest IsFace skin",
            "lowpass": "solve (M + t*K)y = M*x with t=scale^2/4",
            "named_scale": "2-D heat-kernel RMS radius sqrt(4t)",
            "boundary": (
                "natural Neumann/no-flux on "
                f"{ops.boundary_edges} open membrane boundary edges"
            ),
            "normal": "rest-state area-weighted vertex normal",
            "highpass": "scalar normal field minus cotangent-diffused scalar",
        },
        "surface_area_m2": total_mass,
        "mouth_definition": "intrinsic edge distance <= 10 mm from GroupName Lip* vertices",
        "mouth_area_m2": float(ops.mass[ops.mouth].sum()),
        "lip_seed_vertices": int(ops.lip_seed.sum()),
        "mouth_vertices": int(ops.mouth.sum()),
        "boundary_vertices": int(ops.boundary.sum()),
        "boundary_edges": ops.boundary_edges,
        "expression": {
            "target_rms_mm": 1000.0 * target_rms,
            "displacement_rms_mm": 1000.0
            * math.sqrt(float(np.sum(ops.mass[:, None] * u**2) / total_mass)),
            "residual_rms_mm": 1000.0
            * math.sqrt(float(np.sum(ops.mass[:, None] * residual**2) / total_mass)),
            "target_projection_amplitude": projection,
            "projection_orthogonal_rms_over_target": math.sqrt(
                float(np.sum(ops.mass[:, None] * orthogonal**2) / total_mass)
            )
            / target_rms,
        },
        "normal_displacement": {
            "full_face": scalar_stats(
                normal_displacement, ops.mass, np.ones(len(ops.mass), bool)
            ),
            "mouth_10mm": scalar_stats(normal_displacement, ops.mass, ops.mouth),
        },
        "normal_residual": {
            "full_face": scalar_stats(
                normal_residual, ops.mass, np.ones(len(ops.mass), bool)
            ),
            "mouth_10mm": scalar_stats(normal_residual, ops.mass, ops.mouth),
        },
        "scales": scales,
        "interpretation_limit": (
            "high-pass values are descriptive; the target may contain real high-frequency "
            "motion, so these metrics do not identify artifacts by themselves"
        ),
    }
    return summary, output_skin


def render_maps(
    surface: pv.PolyData,
    path: Path,
    scales_m: tuple[float, ...],
    *,
    mouth: bool,
) -> None:
    plotter = pv.Plotter(
        off_screen=True, shape=(2, len(scales_m)), window_size=(2400, 1400)
    )
    rest = surface.copy(deep=True)
    rest.points = np.asarray(surface.point_data["RestPosition"])
    normalled = rest.compute_normals(point_normals=True, cell_normals=False)
    if mouth:
        mask = np.asarray(surface.point_data["LipSeed"], dtype=bool)
        center = np.asarray(rest.points)[mask].mean(axis=0)
        normal = np.asarray(normalled.point_data["Normals"])[mask].mean(axis=0)
        extent = np.ptp(np.asarray(rest.points)[mask], axis=0)
        scale = 1.45 * max(float(extent[1]), float(extent[2]))
    else:
        center = np.average(rest.points, axis=0)
        normal = np.asarray(normalled.point_data["Normals"]).mean(axis=0)
        scale = 0.62 * float(np.ptp(rest.points[:, 1]))
    normal /= np.linalg.norm(normal)
    camera = [
        (center + 3.0 * scale * normal).tolist(),
        center.tolist(),
        [0.0, 1.0, 0.0],
    ]
    roi = (
        np.asarray(surface.point_data["MouthROI10mm"], bool)
        if mouth
        else np.ones(surface.n_points, dtype=bool)
    )
    prefixes = ("NormalDisplacementHighPass", "NormalResidualHighPass")
    row_limits = {
        prefix: max(
            float(
                np.quantile(
                    np.abs(
                        np.asarray(
                            surface.point_data[f"{prefix}{length * 1000:g}mm"],
                            dtype=np.float64,
                        )[roi]
                    ),
                    0.99,
                )
            )
            for length in scales_m
        )
        for prefix in prefixes
    }
    for column, length in enumerate(scales_m):
        label = f"{length * 1000:g}mm"
        for row, prefix in enumerate(prefixes):
            name = f"{prefix}{label}"
            limit = max(row_limits[prefix], 1e-6)
            plotter.subplot(row, column)
            plotter.set_background("#fbfaf7")
            plotter.add_mesh(
                surface,
                scalars=name,
                cmap="coolwarm",
                clim=(-limit, limit),
                smooth_shading=False,
                show_edges=False,
                show_scalar_bar=column == len(scales_m) - 1,
                scalar_bar_args={
                    "title": "motion HP (mm)" if row == 0 else "residual HP (mm)",
                    "fmt": "%.3g",
                },
            )
            plotter.add_text(
                f"{'Mouth' if mouth else 'Face'} {'motion' if row == 0 else 'residual'} HP {label}",
                font_size=12,
            )
            plotter.camera_position = camera
            plotter.enable_parallel_projection()
            plotter.camera.parallel_scale = scale
    path.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(path)
    plotter.close()


def save_volume_map(
    fixture: pv.UnstructuredGrid,
    result: pv.UnstructuredGrid,
    fields: dict[str, np.ndarray],
    path: Path,
) -> None:
    mesh = pv.UnstructuredGrid(fixture.cells, fixture.celltypes, result.points)
    for name in ("GlobalPointId", "IsFace", "IsFixed", "ArtificialCutIncident"):
        mesh.point_data[name] = fixture.point_data[name]
    mesh.point_data["RestPosition"] = fixture.points
    mesh.point_data["Displacement"] = np.asarray(result.points) - np.asarray(
        fixture.points
    )
    for name in (
        "MuscleId",
        "MuscleFraction",
        "FatFraction",
        "AponeurosisFraction",
        "ActivationMask",
        "ActivationControlId",
        "ActivationFiberConfidence",
    ):
        mesh.cell_data[name] = fixture.cell_data[name]
    for name, values in fields.items():
        mesh.cell_data[name] = values
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.save(path)


def resolve_result_dirs(raw: str) -> list[Path]:
    paths = []
    for item in raw.split(","):
        path = Path(item.strip())
        if not path.is_absolute():
            path = EXPERIMENT / path
        path = path.resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"missing result directory: {path}")
        paths.append(path)
    if not paths:
        raise ValueError("result_dirs selected no directories")
    return paths


def audit_case(
    cfg: Config,
    fixture: pv.UnstructuredGrid,
    skin: pv.PolyData,
    ops: SurfaceOperators,
    result_dir: Path,
    scales_m: tuple[float, ...],
) -> dict[str, Any]:
    endpoint_path = result_dir / cfg.endpoint_name
    if not endpoint_path.is_file():
        raise FileNotFoundError(f"missing endpoint: {endpoint_path}")
    result = pv.read(endpoint_path)
    if not isinstance(result, pv.UnstructuredGrid):
        raise TypeError("result endpoint did not read as UnstructuredGrid")
    if (
        result.n_points != fixture.n_points
        or result.n_cells != fixture.n_cells
        or not np.array_equal(result.cells, fixture.cells)
        or not np.array_equal(result.celltypes, fixture.celltypes)
    ):
        raise ValueError("result endpoint topology differs from fixture")

    case_id = result_dir.name
    case_out = cfg.output_dir / case_id
    case_out.mkdir(parents=True, exist_ok=True)
    volume_summary, volume_fields = volume_diagnostics(fixture, result, ops, cfg.top_k)
    file_consistency = endpoint_file_consistency(fixture, result, result_dir)
    surface_summary, surface_map = surface_diagnostics(
        fixture, result, skin, ops, scales_m
    )
    intersections = intersection_diagnostics(fixture, np.asarray(result.points), ops)
    volume_path = case_out / "volume-diagnostics.vtu"
    face_path = case_out / "face-diagnostics.vtp"
    save_volume_map(fixture, result, volume_fields, volume_path)
    surface_map.save(face_path)
    renders = []
    if cfg.render:
        for mouth, name in (
            (False, "face-highpass-maps.png"),
            (True, "mouth-highpass-maps.png"),
        ):
            path = case_out / name
            render_maps(surface_map, path, scales_m, mouth=mouth)
            renders.append(path)
    summary = {
        "case_id": case_id,
        "input": {
            "result_dir": str(result_dir),
            "endpoint": str(endpoint_path),
            "endpoint_sha256": sha256(endpoint_path),
            "saved_summary_sha256": (
                sha256(result_dir / "summary.json")
                if (result_dir / "summary.json").is_file()
                else None
            ),
        },
        "volume": volume_summary,
        "endpoint_file_consistency": file_consistency,
        "surface": surface_summary,
        "intersections": intersections,
        "outputs": {
            "volume_map": str(volume_path),
            "face_map": str(face_path),
            "renders": [str(path) for path in renders],
        },
    }
    metrics_path = case_out / "metrics.json"
    write_json(metrics_path, summary)
    return {**summary, "metrics_path": str(metrics_path)}


def main(cfg: Config) -> None:
    start = time.perf_counter()
    fixture_dir = cfg.fixture.resolve()
    output_dir = cfg.output_dir.resolve()
    if (output_dir / "summary.json").exists():
        raise FileExistsError(f"refusing to overwrite completed audit: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    fixture_path = fixture_dir / "volume.vtu"
    skin_path = fixture_dir / "skin.vtp"
    fixture = pv.read(fixture_path)
    skin = pv.read(skin_path)
    if not isinstance(fixture, pv.UnstructuredGrid) or not isinstance(
        skin, pv.PolyData
    ):
        raise TypeError("fixture volume/skin types are invalid")
    ops = cotangent_operators(skin)
    scales_m = tuple(float(value) / 1000.0 for value in cfg.scales_mm.split(","))
    if scales_m != (0.002, 0.005, 0.01):
        raise ValueError("this audit pins the physical scales 2, 5, and 10 mm")
    result_dirs = resolve_result_dirs(cfg.result_dirs)

    source_dir = output_dir / "sources"
    source_dir.mkdir(exist_ok=True)
    for path in (Path(__file__), HERE / "experiment_profile.py"):
        shutil.copy2(path, source_dir / path.name)
    results = []
    for result_dir in result_dirs:
        LOG.info("Auditing %s", result_dir)
        results.append(audit_case(cfg, fixture, skin, ops, result_dir, scales_m))
    summary = {
        "schema_version": 1,
        "method": "full-volume kinematics, IPC static intersections, and intrinsic cotangent heat high-pass",
        "fixture": {
            "path": str(fixture_dir),
            "volume_sha256": sha256(fixture_path),
            "skin_sha256": sha256(skin_path),
            "summary_sha256": sha256(fixture_dir / "summary.json"),
        },
        "config": cfg.model_dump(mode="json"),
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "wall_s": time.perf_counter() - start,
        "results": results,
    }
    summary_path = output_dir / "summary.json"
    write_json(summary_path, summary)
    artifacts = [
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "artifact-manifest.json"
    ]
    manifest = {
        "files": {
            path.relative_to(output_dir).as_posix(): {
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
            for path in sorted(artifacts)
        }
    }
    write_json(output_dir / "artifact-manifest.json", manifest)
    cherries.log_metrics(
        {
            "audit/cases": len(results),
            "audit/min_detF": min(
                case["volume"]["minimum_cell"]["detF"] for case in results
            ),
            "audit/new_intersections": sum(
                domain["new_edge_face_hits"]
                for case in results
                for domain in case["intersections"]["domains"].values()
            ),
        }
    )
    LOG.info("Wrote %s", summary_path)


if __name__ == "__main__":
    cherries.main(
        main, profile=None if os.environ.get("DEBUG") else ProfileCometNoCommit
    )
