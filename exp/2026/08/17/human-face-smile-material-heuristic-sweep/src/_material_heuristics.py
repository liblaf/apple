from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from _human_face_config import IS_FACE, SKIN_E, SKIN_NU, SMILE_TARGET

SCHEMA_VERSION = 2


@dataclass(frozen=True)
class MaterialCandidate:
    young_min_scale: float
    prestrain_gain: float

    @property
    def label(self) -> str:
        return (
            f"e{round(100 * self.young_min_scale):03d}"
            f"-p{round(100 * self.prestrain_gain):03d}"
        )


@dataclass(frozen=True)
class SurfaceGeometryBasis:
    surface: pv.PolyData
    triangles: np.ndarray
    rest_area: np.ndarray
    target_area: np.ndarray
    area_ratio: np.ndarray
    signed_log_area_raw: np.ndarray
    face_mask: np.ndarray
    finite_mask: np.ndarray
    eligible_mask: np.ndarray
    pair_i: np.ndarray
    pair_j: np.ndarray
    pair_edge_length: np.ndarray
    interior_i: np.ndarray
    interior_j: np.ndarray
    interior_edge_length: np.ndarray
    interior_conductance: np.ndarray
    boundary_cell: np.ndarray
    boundary_edge_length: np.ndarray
    boundary_conductance: np.ndarray
    geometry_metrics: dict[str, Any]


@dataclass(frozen=True)
class SignedHeatField:
    area_deadband: float
    cap_quantile: float
    diffusion_sigma: float
    expansion_cap: float
    contraction_cap: float
    deadbanded: np.ndarray
    capped: np.ndarray
    diffused: np.ndarray
    metrics: dict[str, Any]
    validation_errors: list[str]


@dataclass(frozen=True)
class CandidateFields:
    expansion_severity: np.ndarray
    contraction_severity: np.ndarray
    expansion_cap: float
    contraction_cap: float
    expansion_weight: np.ndarray
    contraction_log: np.ndarray
    young: np.ndarray
    lambda_: np.ndarray
    mu: np.ndarray
    activation_inv: np.ndarray
    activation_diag: np.ndarray
    stress_free_area_ratio: np.ndarray


def default_candidates() -> list[MaterialCandidate]:
    return [
        MaterialCandidate(young_min_scale=young_scale, prestrain_gain=gain)
        for young_scale in (1.0, 0.25)
        for gain in (0.0, 0.5, 1.0)
    ]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _update_hash(
    digest: Any,
    *,
    name: str,
    values: np.ndarray,
    dtype: str,
) -> None:
    canonical = np.ascontiguousarray(values, dtype=np.dtype(dtype))
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(canonical.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical.tobytes())


def skin_topology_content_hash(surface: pv.PolyData) -> str:
    from liblaf.apple.common import GLOBAL_POINT_ID

    digest = hashlib.sha256()
    for name, values, dtype in (
        ("points", np.asarray(surface.points), "<f8"),
        ("faces", np.asarray(surface.faces), "<i8"),
        (
            GLOBAL_POINT_ID.vtk,
            np.asarray(surface.point_data[GLOBAL_POINT_ID.vtk]),
            "<i8",
        ),
    ):
        _update_hash(digest, name=name, values=values, dtype=dtype)
    return digest.hexdigest()


def skin_material_content_hash(surface: pv.PolyData) -> str:
    from liblaf.apple.common import ACTIVATION_INV, FRACTION, LAMBDA, MU

    digest = hashlib.sha256()
    for name in (LAMBDA.vtk, MU.vtk, FRACTION.vtk, ACTIVATION_INV.vtk):
        _update_hash(
            digest,
            name=name,
            values=np.asarray(surface.cell_data[name]),
            dtype="<f8",
        )
    return digest.hexdigest()


def skin_solver_content_hash(surface: pv.PolyData) -> str:
    from liblaf.apple.common import (
        ACTIVATION_INV,
        FRACTION,
        GLOBAL_POINT_ID,
        LAMBDA,
        MU,
    )

    digest = hashlib.sha256()
    for name, values, dtype in (
        ("points", np.asarray(surface.points), "<f8"),
        ("faces", np.asarray(surface.faces), "<i8"),
        (
            GLOBAL_POINT_ID.vtk,
            np.asarray(surface.point_data[GLOBAL_POINT_ID.vtk]),
            "<i8",
        ),
        (LAMBDA.vtk, np.asarray(surface.cell_data[LAMBDA.vtk]), "<f8"),
        (MU.vtk, np.asarray(surface.cell_data[MU.vtk]), "<f8"),
        (FRACTION.vtk, np.asarray(surface.cell_data[FRACTION.vtk]), "<f8"),
        (
            ACTIVATION_INV.vtk,
            np.asarray(surface.cell_data[ACTIVATION_INV.vtk]),
            "<f8",
        ),
    ):
        _update_hash(digest, name=name, values=values, dtype=dtype)
    return digest.hexdigest()


def material_field_hash(*arrays: np.ndarray) -> str:
    """Compatibility hash for older callers; prefer the content-hash helpers."""
    digest = hashlib.sha256()
    for index, values in enumerate(arrays):
        _update_hash(
            digest,
            name=f"array-{index}",
            values=np.asarray(values),
            dtype="<f8",
        )
    return digest.hexdigest()


def triangle_faces(surface: pv.PolyData) -> np.ndarray:
    encoded = np.asarray(surface.faces, dtype=np.int64).reshape(-1, 4)
    if encoded.size == 0 or not np.all(encoded[:, 0] == 3):
        msg = "expected a non-empty triangulated surface"
        raise ValueError(msg)
    return encoded[:, 1:]


def triangle_area(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    p0 = points[triangles[:, 0]]
    p1 = points[triangles[:, 1]]
    p2 = points[triangles[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)


def weighted_quantile(
    values: np.ndarray, weights: np.ndarray, quantile: float
) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if values.ndim != 1 or weights.shape != values.shape:
        msg = (
            "weighted quantile expects matching 1D arrays, got "
            f"{values.shape} and {weights.shape}"
        )
        raise ValueError(msg)
    if not 0.0 < quantile <= 1.0:
        msg = f"weighted quantile must be in (0, 1], got {quantile}"
        raise ValueError(msg)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        msg = "weighted quantile received no finite positive-weight samples"
        raise ValueError(msg)
    ordered = np.argsort(values[valid], kind="stable")
    sorted_values = values[valid][ordered]
    sorted_weights = weights[valid][ordered]
    cumulative = np.cumsum(sorted_weights)
    threshold = quantile * cumulative[-1]
    index = min(
        int(np.searchsorted(cumulative, threshold, side="left")), len(ordered) - 1
    )
    return float(sorted_values[index])


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    denominator = float(weights.sum())
    return (
        float(np.dot(values, weights) / denominator) if denominator > 0.0 else math.nan
    )


def _weighted_rms(values: np.ndarray, weights: np.ndarray) -> float:
    return math.sqrt(_weighted_mean(np.square(values), weights))


def _stats(values: np.ndarray, weights: np.ndarray | None = None) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            "min": math.nan,
            "median": math.nan,
            "q99": math.nan,
            "max": math.nan,
            "mean": math.nan,
            "rms": math.nan,
        }
    result = {
        "min": float(values.min()),
        "median": float(np.median(values)),
        "q99": float(np.quantile(values, 0.99)),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "rms": float(np.linalg.norm(values) / math.sqrt(values.size)),
    }
    if weights is not None:
        result["area_weighted_mean"] = _weighted_mean(values, weights)
        result["area_weighted_rms"] = _weighted_rms(values, weights)
    return result


def _finite_volume_graph(  # noqa: C901
    points: np.ndarray,
    triangles: np.ndarray,
    area: np.ndarray,
    eligible: np.ndarray,
) -> dict[str, Any]:
    edge_owner: dict[tuple[int, int], list[int]] = {}
    for cell_id, triangle in enumerate(np.asarray(triangles, dtype=np.int64)):
        for a, b in (
            (int(triangle[0]), int(triangle[1])),
            (int(triangle[1]), int(triangle[2])),
            (int(triangle[2]), int(triangle[0])),
        ):
            edge = (a, b) if a < b else (b, a)
            edge_owner.setdefault(edge, []).append(cell_id)

    pair_i: list[int] = []
    pair_j: list[int] = []
    pair_length: list[float] = []
    interior_i: list[int] = []
    interior_j: list[int] = []
    interior_edge_length: list[float] = []
    interior_conductance: list[float] = []
    boundary_cell: list[int] = []
    boundary_edge_length: list[float] = []
    boundary_conductance: list[float] = []
    nonmanifold_edges_outside_eligible = 0
    for (point_i, point_j), owners in edge_owner.items():
        if len(owners) > 2:
            if np.any(eligible[np.asarray(owners, dtype=np.int64)]):
                msg = (
                    f"eligible surface edge {(point_i, point_j)} belongs to "
                    f"{len(owners)} triangles"
                )
                raise ValueError(msg)
            nonmanifold_edges_outside_eligible += 1
            continue
        edge_vector = points[point_j] - points[point_i]
        edge_length = float(np.linalg.norm(edge_vector))
        if not math.isfinite(edge_length) or edge_length <= 0.0:
            msg = f"surface edge {(point_i, point_j)} has invalid length {edge_length}"
            raise ValueError(msg)
        if len(owners) == 2:
            cell_i, cell_j = owners
            pair_i.append(cell_i)
            pair_j.append(cell_j)
            pair_length.append(edge_length)
            if eligible[cell_i] and eligible[cell_j]:
                interior_i.append(cell_i)
                interior_j.append(cell_j)
                interior_edge_length.append(edge_length)
                interior_conductance.append(
                    3.0 * edge_length**2 / (2.0 * (area[cell_i] + area[cell_j]))
                )
            elif eligible[cell_i] != eligible[cell_j]:
                active_cell = cell_i if eligible[cell_i] else cell_j
                boundary_cell.append(active_cell)
                boundary_edge_length.append(edge_length)
                boundary_conductance.append(
                    3.0 * edge_length**2 / (2.0 * area[active_cell])
                )
        elif eligible[owners[0]]:
            owner = owners[0]
            boundary_cell.append(owner)
            boundary_edge_length.append(edge_length)
            boundary_conductance.append(3.0 * edge_length**2 / (2.0 * area[owner]))
    return {
        "pair_i": np.asarray(pair_i, dtype=np.int64),
        "pair_j": np.asarray(pair_j, dtype=np.int64),
        "pair_edge_length": np.asarray(pair_length, dtype=np.float64),
        "interior_i": np.asarray(interior_i, dtype=np.int64),
        "interior_j": np.asarray(interior_j, dtype=np.int64),
        "interior_edge_length": np.asarray(interior_edge_length, dtype=np.float64),
        "interior_conductance": np.asarray(interior_conductance, dtype=np.float64),
        "boundary_cell": np.asarray(boundary_cell, dtype=np.int64),
        "boundary_edge_length": np.asarray(boundary_edge_length, dtype=np.float64),
        "boundary_conductance": np.asarray(boundary_conductance, dtype=np.float64),
        "nonmanifold_edges_outside_eligible": (nonmanifold_edges_outside_eligible),
    }


def _implicit_heat_diffusion(
    raw: np.ndarray,
    area: np.ndarray,
    eligible: np.ndarray,
    graph: dict[str, np.ndarray],
    *,
    diffusion_length: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if diffusion_length <= 0.0:
        msg = f"diffusion length must be positive, got {diffusion_length}"
        raise ValueError(msg)
    active_ids = np.flatnonzero(eligible).astype(np.int64)
    if active_ids.size == 0:
        msg = "signed log-area diffusion selected no eligible triangles"
        raise ValueError(msg)
    global_to_local = np.full(raw.size, -1, dtype=np.int64)
    global_to_local[active_ids] = np.arange(active_ids.size, dtype=np.int64)
    interior_i = global_to_local[graph["interior_i"]]
    interior_j = global_to_local[graph["interior_j"]]
    conductance = graph["interior_conductance"]
    boundary_ids = global_to_local[graph["boundary_cell"]]
    boundary_conductance = graph["boundary_conductance"]
    if np.any(interior_i < 0) or np.any(interior_j < 0) or np.any(boundary_ids < 0):
        msg = "finite-volume graph contains a non-eligible local index"
        raise ValueError(msg)

    heat_time = 0.5 * diffusion_length**2
    diagonal = np.asarray(area[active_ids], dtype=np.float64).copy()
    np.add.at(diagonal, interior_i, heat_time * conductance)
    np.add.at(diagonal, interior_j, heat_time * conductance)
    np.add.at(diagonal, boundary_ids, heat_time * boundary_conductance)
    row = np.concatenate((np.arange(active_ids.size), interior_i, interior_j))
    col = np.concatenate((np.arange(active_ids.size), interior_j, interior_i))
    data = np.concatenate(
        (diagonal, -heat_time * conductance, -heat_time * conductance)
    )
    matrix = sp.coo_matrix((data, (row, col)), shape=(active_ids.size,) * 2).tocsr()
    rhs = area[active_ids] * raw[active_ids]
    solution = np.asarray(spla.spsolve(matrix, rhs), dtype=np.float64)
    if not np.isfinite(solution).all():
        msg = "implicit heat diffusion produced non-finite signed log area"
        raise RuntimeError(msg)
    residual = matrix @ solution - rhs
    denominator = max(float(np.linalg.norm(rhs)), np.finfo(np.float64).tiny)
    relative_residual = float(np.linalg.norm(residual) / denominator)
    infinity_denominator = max(
        float(np.linalg.norm(rhs, ord=np.inf)), np.finfo(np.float64).tiny
    )
    relative_infinity_residual = float(
        np.linalg.norm(residual, ord=np.inf) / infinity_denominator
    )
    symmetry = matrix - matrix.T
    symmetry_error = float(np.abs(symmetry.data).max()) if symmetry.nnz else 0.0
    output = np.zeros_like(raw, dtype=np.float64)
    output[active_ids] = solution
    return output, {
        "method": "triangle-center finite-volume implicit heat",
        "mass": "rest triangle area",
        "interior_conductance": "3 * shared_edge_length^2 / (2 * sum_adjacent_area)",
        "boundary_condition": "zero Dirichlet at finite IsFace boundary",
        "boundary_conductance": ("3 * boundary_edge_length^2 / (2 * triangle_area)"),
        "diffusion_length": float(diffusion_length),
        "heat_time": float(heat_time),
        "n_unknowns": int(active_ids.size),
        "n_interior_edges": int(conductance.size),
        "n_boundary_edges": int(boundary_conductance.size),
        "linear_relative_residual": relative_residual,
        "linear_relative_infinity_residual": relative_infinity_residual,
        "matrix_symmetry_max_abs": symmetry_error,
        "matrix_off_diagonal_nonpositive": bool(np.all(data[active_ids.size :] <= 0.0)),
    }


def prepare_surface_geometry(mesh: pv.UnstructuredGrid) -> SurfaceGeometryBasis:
    from liblaf.apple.common import GLOBAL_POINT_ID

    surface = mesh.extract_surface(algorithm=None).triangulate()
    if "vtkOriginalPointIds" not in surface.point_data:
        msg = "extract_surface did not produce vtkOriginalPointIds"
        raise KeyError(msg)
    original_ids = np.asarray(surface.point_data["vtkOriginalPointIds"], dtype=np.int64)
    triangles = triangle_faces(surface)
    rest_points = np.asarray(surface.points, dtype=np.float64)
    smile = np.asarray(mesh.point_data[SMILE_TARGET], dtype=np.float64)[original_ids]
    finite_points = np.isfinite(smile).all(axis=1)
    target_points = rest_points + np.nan_to_num(smile, nan=0.0, posinf=0.0, neginf=0.0)
    rest_area = triangle_area(rest_points, triangles)
    target_area = triangle_area(target_points, triangles)
    valid_area = rest_area > np.finfo(np.float64).eps
    area_ratio = np.ones(surface.n_cells, dtype=np.float64)
    area_ratio[valid_area] = target_area[valid_area] / rest_area[valid_area]
    signed_log_area_raw = np.log(np.maximum(area_ratio, np.finfo(np.float64).tiny))
    face_points = np.asarray(mesh.point_data[IS_FACE], dtype=bool)[original_ids]
    face_mask = np.all(face_points[triangles], axis=1)
    finite_mask = np.all(finite_points[triangles], axis=1)
    eligible_mask = face_mask & finite_mask & valid_area
    mesh_point_ids = (
        np.asarray(mesh.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64)
        if GLOBAL_POINT_ID.vtk in mesh.point_data
        else np.arange(mesh.n_points, dtype=np.int64)
    )
    surface.point_data[GLOBAL_POINT_ID.vtk] = mesh_point_ids[original_ids]
    graph = _finite_volume_graph(rest_points, triangles, rest_area, eligible_mask)
    eligible_area = rest_area[eligible_mask]
    component_metrics = _component_metrics(
        eligible_mask,
        graph["pair_i"],
        graph["pair_j"],
        rest_area,
        float(eligible_area.sum()),
    )
    geometry_metrics = {
        "eligible_rest_area": float(eligible_area.sum()),
        "eligible_triangles": int(eligible_mask.sum()),
        "eligible_components": int(component_metrics["components"]),
        "eligible_singletons": int(component_metrics["singleton_components"]),
        "eligible_interior_edges": int(graph["interior_i"].size),
        "eligible_boundary_edges": int(graph["boundary_cell"].size),
        "nonmanifold_edges_outside_eligible": int(
            graph["nonmanifold_edges_outside_eligible"]
        ),
        "topology_identity_lhs_3F": int(3 * eligible_mask.sum()),
        "topology_identity_rhs_2Eint_plus_Eboundary": int(
            2 * graph["interior_i"].size + graph["boundary_cell"].size
        ),
        "interior_edge_length_median": float(np.median(graph["interior_edge_length"])),
        "signed_log_raw": _stats(signed_log_area_raw[eligible_mask], eligible_area),
    }
    return SurfaceGeometryBasis(
        surface=surface,
        triangles=triangles,
        rest_area=rest_area,
        target_area=target_area,
        area_ratio=area_ratio,
        signed_log_area_raw=signed_log_area_raw,
        face_mask=face_mask,
        finite_mask=finite_mask,
        eligible_mask=eligible_mask,
        pair_i=graph["pair_i"],
        pair_j=graph["pair_j"],
        pair_edge_length=graph["pair_edge_length"],
        interior_i=graph["interior_i"],
        interior_j=graph["interior_j"],
        interior_edge_length=graph["interior_edge_length"],
        interior_conductance=graph["interior_conductance"],
        boundary_cell=graph["boundary_cell"],
        boundary_edge_length=graph["boundary_edge_length"],
        boundary_conductance=graph["boundary_conductance"],
        geometry_metrics=geometry_metrics,
    )


def _weighted_correlation(
    left: np.ndarray, right: np.ndarray, weights: np.ndarray
) -> float:
    left_mean = _weighted_mean(left, weights)
    right_mean = _weighted_mean(right, weights)
    left_centered = left - left_mean
    right_centered = right - right_mean
    denominator = math.sqrt(
        float(np.dot(weights, np.square(left_centered)))
        * float(np.dot(weights, np.square(right_centered)))
    )
    return (
        float(np.dot(weights, left_centered * right_centered) / denominator)
        if denominator > 0.0
        else math.nan
    )


def make_signed_heat_field(  # noqa: C901, PLR0912, PLR0915
    geometry: SurfaceGeometryBasis,
    *,
    area_deadband: float,
    cap_quantile: float,
    diffusion_sigma: float,
    max_normalized_interior_jump_q99: float = 0.08,
    max_normalized_interior_jump: float = 0.20,
    max_normalized_boundary_jump_q99: float = 0.08,
    max_normalized_boundary_jump: float = 0.20,
) -> SignedHeatField:
    if not 0.0 <= area_deadband < 1.0:
        msg = f"area deadband must be in [0, 1), got {area_deadband}"
        raise ValueError(msg)
    if not 0.0 < cap_quantile <= 1.0:
        msg = f"cap quantile must be in (0, 1], got {cap_quantile}"
        raise ValueError(msg)
    if diffusion_sigma <= 0.0:
        msg = f"diffusion sigma must be positive, got {diffusion_sigma}"
        raise ValueError(msg)
    eligible = geometry.eligible_mask
    area = geometry.rest_area
    signed_raw = geometry.signed_log_area_raw
    soft_threshold = math.log1p(area_deadband)
    deadbanded = np.where(
        eligible,
        np.sign(signed_raw) * np.maximum(np.abs(signed_raw) - soft_threshold, 0.0),
        0.0,
    )
    expansion_mask = deadbanded > 0.0
    contraction_mask = deadbanded < 0.0
    if not np.any(expansion_mask) or not np.any(contraction_mask):
        msg = "soft log-area deadband removed all expansion or contraction support"
        raise ValueError(msg)
    expansion_cap = weighted_quantile(
        deadbanded[expansion_mask],
        area[expansion_mask],
        cap_quantile,
    )
    contraction_cap = weighted_quantile(
        -deadbanded[contraction_mask],
        area[contraction_mask],
        cap_quantile,
    )
    if expansion_cap <= 0.0 or contraction_cap <= 0.0:
        msg = "area-weighted severity caps must be positive"
        raise ValueError(msg)
    capped = np.clip(deadbanded, -contraction_cap, expansion_cap)
    graph = {
        "interior_i": geometry.interior_i,
        "interior_j": geometry.interior_j,
        "interior_conductance": geometry.interior_conductance,
        "boundary_cell": geometry.boundary_cell,
        "boundary_conductance": geometry.boundary_conductance,
    }
    diffused, solver_metrics = _implicit_heat_diffusion(
        capped,
        area,
        eligible,
        graph,
        diffusion_length=diffusion_sigma,
    )
    scale = max(expansion_cap, contraction_cap)
    interior_jump = (
        np.abs(diffused[geometry.interior_i] - diffused[geometry.interior_j]) / scale
    )
    boundary_jump = np.abs(diffused[geometry.boundary_cell]) / scale
    interior_q99 = weighted_quantile(interior_jump, geometry.interior_edge_length, 0.99)
    boundary_q99 = weighted_quantile(boundary_jump, geometry.boundary_edge_length, 0.99)
    input_m_norm = _weighted_rms(capped[eligible], area[eligible])
    output_m_norm = _weighted_rms(diffused[eligible], area[eligible])
    maximum_principle_violation = max(
        0.0,
        float(diffused[eligible].max() - expansion_cap),
        float(-contraction_cap - diffused[eligible].min()),
    )
    initially_active = deadbanded != 0.0
    sign_disagreement = initially_active & (np.sign(diffused) != np.sign(deadbanded))
    eligible_area = float(area[eligible].sum())
    dirichlet_energy = 0.5 * float(
        np.dot(
            geometry.interior_conductance,
            np.square(diffused[geometry.interior_i] - diffused[geometry.interior_j]),
        )
        + np.dot(
            geometry.boundary_conductance,
            np.square(diffused[geometry.boundary_cell]),
        )
    )
    metrics: dict[str, Any] = {
        **solver_metrics,
        "soft_deadband_log": float(soft_threshold),
        "cap_quantile": float(cap_quantile),
        "expansion_cap": float(expansion_cap),
        "contraction_cap": float(contraction_cap),
        "input_area_weighted_rms": input_m_norm,
        "output_area_weighted_rms": output_m_norm,
        "area_weighted_rms_attenuation": float(output_m_norm / input_m_norm),
        "area_weighted_correlation_with_capped_input": _weighted_correlation(
            capped[eligible], diffused[eligible], area[eligible]
        ),
        "maximum_principle_violation": maximum_principle_violation,
        "interior_normalized_jump_q99": float(interior_q99),
        "interior_normalized_jump_max": float(interior_jump.max()),
        "boundary_normalized_jump_q99": float(boundary_q99),
        "boundary_normalized_jump_max": float(boundary_jump.max()),
        "dirichlet_energy": dirichlet_energy,
        "sign_disagreement_rest_area_fraction": float(
            area[sign_disagreement].sum() / eligible_area
        ),
        "positive_rest_area_fraction": float(
            area[eligible & (diffused > 0.0)].sum() / eligible_area
        ),
        "zero_rest_area_fraction": float(
            area[eligible & (diffused == 0.0)].sum() / eligible_area
        ),
        "negative_rest_area_fraction": float(
            area[eligible & (diffused < 0.0)].sum() / eligible_area
        ),
        "deadbanded": _stats(deadbanded[eligible], area[eligible]),
        "capped": _stats(capped[eligible], area[eligible]),
        "diffused": _stats(diffused[eligible], area[eligible]),
    }
    errors: list[str] = []
    topology_lhs = int(geometry.geometry_metrics["topology_identity_lhs_3F"])
    topology_rhs = int(
        geometry.geometry_metrics["topology_identity_rhs_2Eint_plus_Eboundary"]
    )
    if int(geometry.geometry_metrics["eligible_components"]) != 1:
        errors.append(
            "eligible finite IsFace surface is not one edge-connected component"
        )
    if int(geometry.geometry_metrics["eligible_singletons"]) != 0:
        errors.append("eligible finite IsFace surface contains isolated triangles")
    if topology_lhs != topology_rhs:
        errors.append(
            f"eligible topology identity failed: {topology_lhs} != {topology_rhs}"
        )
    if int(geometry.geometry_metrics["eligible_boundary_edges"]) <= 0:
        errors.append("eligible finite IsFace component has no Dirichlet boundary")
    if float(solver_metrics["linear_relative_infinity_residual"]) > 1.0e-10:
        errors.append("implicit heat linear relative infinity residual exceeds 1e-10")
    if float(solver_metrics["matrix_symmetry_max_abs"]) > 1.0e-12:
        errors.append("implicit heat matrix is not symmetric to 1e-12")
    if not bool(solver_metrics["matrix_off_diagonal_nonpositive"]):
        errors.append("implicit heat matrix has a positive off-diagonal entry")
    if maximum_principle_violation > 1.0e-10 * scale:
        errors.append("implicit heat field violates the capped maximum principle")
    if output_m_norm > input_m_norm * (1.0 + 1.0e-12):
        errors.append("implicit heat diffusion increased the rest-area M-norm")
    if not np.allclose(diffused[~eligible], 0.0, rtol=0.0, atol=0.0):
        errors.append("implicit heat field is nonzero outside eligible finite IsFace")
    if interior_q99 > max_normalized_interior_jump_q99:
        errors.append("interior normalized jump q99 exceeds configured gate")
    if float(interior_jump.max()) > max_normalized_interior_jump:
        errors.append("interior normalized jump max exceeds configured gate")
    if boundary_q99 > max_normalized_boundary_jump_q99:
        errors.append("boundary normalized jump q99 exceeds configured gate")
    if float(boundary_jump.max()) > max_normalized_boundary_jump:
        errors.append("boundary normalized jump max exceeds configured gate")
    return SignedHeatField(
        area_deadband=float(area_deadband),
        cap_quantile=float(cap_quantile),
        diffusion_sigma=float(diffusion_sigma),
        expansion_cap=float(expansion_cap),
        contraction_cap=float(contraction_cap),
        deadbanded=deadbanded,
        capped=capped,
        diffused=diffused,
        metrics=metrics,
        validation_errors=errors,
    )


def candidate_fields(
    geometry: SurfaceGeometryBasis,
    signed_field: SignedHeatField,
    candidate: MaterialCandidate,
) -> CandidateFields:
    if not 0.0 < candidate.young_min_scale <= 1.0:
        msg = f"young_min_scale must be in (0, 1], got {candidate.young_min_scale}"
        raise ValueError(msg)
    if not 0.0 <= candidate.prestrain_gain <= 1.0:
        msg = f"prestrain_gain must be in [0, 1], got {candidate.prestrain_gain}"
        raise ValueError(msg)
    expansion_severity = np.maximum(signed_field.diffused, 0.0)
    contraction_severity = np.maximum(-signed_field.diffused, 0.0)
    expansion_weight = np.clip(
        expansion_severity / signed_field.expansion_cap, 0.0, 1.0
    )
    contraction_log = contraction_severity
    young = SKIN_E * np.exp(math.log(candidate.young_min_scale) * expansion_weight)
    activation_diag = np.exp(0.5 * candidate.prestrain_gain * contraction_log) - 1.0
    activation_inv = np.zeros((geometry.surface.n_cells, 3), dtype=np.float64)
    activation_inv[:, 0] = activation_diag
    activation_inv[:, 1] = activation_diag
    stress_free_area_ratio = np.reciprocal(np.square(1.0 + activation_diag))
    lambda_ = young * SKIN_NU / ((1.0 + SKIN_NU) * (1.0 - 2.0 * SKIN_NU))
    mu = young / (2.0 * (1.0 + SKIN_NU))
    return CandidateFields(
        expansion_severity=expansion_severity,
        contraction_severity=contraction_severity,
        expansion_cap=signed_field.expansion_cap,
        contraction_cap=signed_field.contraction_cap,
        expansion_weight=expansion_weight,
        contraction_log=contraction_log,
        young=young,
        lambda_=lambda_,
        mu=mu,
        activation_inv=activation_inv,
        activation_diag=activation_diag,
        stress_free_area_ratio=stress_free_area_ratio,
    )


def _component_metrics(
    mask: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    area: np.ndarray,
    eligible_area: float,
) -> dict[str, float | int]:
    active_ids = np.flatnonzero(mask).astype(np.int64)
    if active_ids.size == 0:
        return {
            "components": 0,
            "singleton_components": 0,
            "small_components_le4": 0,
            "largest_component_area_fraction_of_support": math.nan,
            "singleton_area_fraction_of_eligible": 0.0,
            "small_le4_area_fraction_of_eligible": 0.0,
        }
    parent = np.arange(mask.size, dtype=np.int64)
    size = np.ones(mask.size, dtype=np.int64)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            return
        if size[root_left] < size[root_right]:
            root_left, root_right = root_right, root_left
        parent[root_right] = root_left
        size[root_left] += size[root_right]

    selected_edges = mask[pair_i] & mask[pair_j]
    for left, right in zip(pair_i[selected_edges], pair_j[selected_edges], strict=True):
        union(int(left), int(right))
    roots = np.asarray([find(int(cell)) for cell in active_ids], dtype=np.int64)
    unique_roots, inverse, counts = np.unique(
        roots, return_inverse=True, return_counts=True
    )
    component_area = np.bincount(
        inverse, weights=area[active_ids], minlength=unique_roots.size
    )
    support_area = float(component_area.sum())
    singleton = counts == 1
    small = counts <= 4
    return {
        "components": int(unique_roots.size),
        "singleton_components": int(singleton.sum()),
        "small_components_le4": int(small.sum()),
        "largest_component_area_fraction_of_support": float(
            component_area.max() / support_area
        ),
        "singleton_area_fraction_of_eligible": float(
            component_area[singleton].sum() / eligible_area
        ),
        "small_le4_area_fraction_of_eligible": float(
            component_area[small].sum() / eligible_area
        ),
    }


def _edge_jump_metrics(
    values: np.ndarray,
    geometry: SurfaceGeometryBasis,
    *,
    boundary_reference: float,
) -> dict[str, float]:
    interior_jumps = np.abs(values[geometry.interior_i] - values[geometry.interior_j])
    boundary_jumps = np.abs(values[geometry.boundary_cell] - boundary_reference)
    jumps = np.concatenate((interior_jumps, boundary_jumps))
    weights = np.concatenate(
        (geometry.interior_edge_length, geometry.boundary_edge_length)
    )
    value_range = float(
        values[geometry.eligible_mask].max() - values[geometry.eligible_mask].min()
    )
    return {
        "max": float(jumps.max()),
        "edge_length_weighted_q99": weighted_quantile(jumps, weights, 0.99),
        "edge_length_weighted_rms": _weighted_rms(jumps, weights),
        "max_fraction_of_eligible_range": float(jumps.max() / value_range)
        if value_range > 0.0
        else 0.0,
        "face_boundary_max": float(boundary_jumps.max())
        if boundary_jumps.size
        else 0.0,
    }


def candidate_field_metrics(
    geometry: SurfaceGeometryBasis,
    candidate: MaterialCandidate,
    fields: CandidateFields,
) -> dict[str, Any]:
    eligible = geometry.eligible_mask
    area = geometry.rest_area
    eligible_area = float(area[eligible].sum())
    expansion_mask = fields.expansion_severity > 0.0
    contraction_mask = fields.contraction_severity > 0.0
    return {
        "skin/surface_triangles": int(geometry.surface.n_cells),
        "skin/is_face_triangles": int(geometry.face_mask.sum()),
        "skin/eligible_triangles": int(eligible.sum()),
        "skin/expansion_triangles": int(expansion_mask.sum()),
        "skin/contraction_triangles": int(contraction_mask.sum()),
        "skin/expansion_rest_area_fraction": float(
            area[expansion_mask].sum() / eligible_area
        ),
        "skin/contraction_rest_area_fraction": float(
            area[contraction_mask].sum() / eligible_area
        ),
        "skin/expansion_log_cap": float(fields.expansion_cap),
        "skin/contraction_log_cap": float(fields.contraction_cap),
        "skin/E_MPa_min": float(fields.young.min()),
        "skin/E_MPa_mean": float(fields.young.mean()),
        "skin/E_MPa_area_weighted_mean": _weighted_mean(
            fields.young[eligible], area[eligible]
        ),
        "skin/E_MPa_max": float(fields.young.max()),
        "skin/activation_inv_diag_max": float(fields.activation_diag.max()),
        "skin/activation_inv_diag_rms": float(
            np.linalg.norm(fields.activation_diag)
            / math.sqrt(fields.activation_diag.size)
        ),
        "skin/activation_inv_diag_area_weighted_mean": _weighted_mean(
            fields.activation_diag[eligible], area[eligible]
        ),
        "skin/activation_inv_diag_area_weighted_rms": _weighted_rms(
            fields.activation_diag[eligible], area[eligible]
        ),
        "skin/stress_free_area_ratio_min": float(fields.stress_free_area_ratio.min()),
        "field/expansion_components": _component_metrics(
            expansion_mask,
            geometry.pair_i,
            geometry.pair_j,
            area,
            eligible_area,
        ),
        "field/contraction_components": _component_metrics(
            contraction_mask,
            geometry.pair_i,
            geometry.pair_j,
            area,
            eligible_area,
        ),
        "field/E_edge_jump_MPa": _edge_jump_metrics(
            fields.young,
            geometry,
            boundary_reference=float(SKIN_E),
        ),
        "field/activation_edge_jump": _edge_jump_metrics(
            fields.activation_diag,
            geometry,
            boundary_reference=0.0,
        ),
        "area_ratio/is_face": _stats(
            geometry.area_ratio[geometry.face_mask], area[geometry.face_mask]
        ),
        "field/expansion_severity_active": _stats(
            fields.expansion_severity[expansion_mask], area[expansion_mask]
        ),
        "field/contraction_severity_active": _stats(
            fields.contraction_severity[contraction_mask], area[contraction_mask]
        ),
        "field/young_rule": (
            "E0 * exp(log(EminScale) * "
            "positive(diffused_capped_log_area) / positive_weighted_cap)"
        ),
        "field/prestrain_rule": (
            "Ainv_diag = exp(0.5 * gain * positive(-diffused_capped_log_area)) - 1"
        ),
        "skin/lame_conversion": (
            "existing 3D isotropic convention: "
            "lambda = E * nu / ((1 + nu) * (1 - 2 * nu)); "
            "mu = E / (2 * (1 + nu))"
        ),
        "field/candidate": candidate.label,
    }


def _validate_candidate(  # noqa: C901, PLR0912, PLR0915
    *,
    geometry: SurfaceGeometryBasis,
    signed_field: SignedHeatField,
    candidate: MaterialCandidate,
    fields: CandidateFields,
    metrics: dict[str, Any],
    max_e_edge_jump: float,
    max_activation_edge_jump: float,
    max_e_edge_rms: float,
    max_activation_edge_rms: float,
    max_singleton_components: int,
    max_small_component_area_fraction: float,
) -> list[str]:
    errors: list[str] = []
    tolerance = 1.0e-12
    eligible = geometry.eligible_mask
    errors.extend(signed_field.validation_errors)
    if not np.isfinite(fields.young).all() or np.any(fields.young <= 0.0):
        errors.append("skin Young's modulus contains non-finite or non-positive values")
    if not np.isfinite(fields.lambda_).all() or not np.isfinite(fields.mu).all():
        errors.append("skin Lame fields contain non-finite values")
    if not np.isfinite(fields.activation_inv).all():
        errors.append("skin ActivationInv contains non-finite values")
    if np.any(fields.activation_diag < 0.0):
        errors.append("skin prestrain contains negative values")
    if not np.allclose(fields.young[~eligible], SKIN_E, rtol=0.0, atol=tolerance):
        errors.append("Young's modulus changed outside eligible finite IsFace")
    if not np.allclose(fields.activation_inv[~eligible], 0.0, rtol=0.0, atol=tolerance):
        errors.append("prestrain changed outside eligible finite IsFace")
    if not np.allclose(
        fields.young[signed_field.diffused <= 0.0],
        SKIN_E,
        rtol=0.0,
        atol=tolerance,
    ):
        errors.append(
            "Young's modulus changed where diffused signed log area is not positive"
        )
    if not np.allclose(
        fields.activation_inv[signed_field.diffused >= 0.0],
        0.0,
        rtol=0.0,
        atol=tolerance,
    ):
        errors.append(
            "prestrain changed where diffused signed log area is not negative"
        )
    expected_lambda = fields.young * SKIN_NU / ((1.0 + SKIN_NU) * (1.0 - 2.0 * SKIN_NU))
    expected_mu = fields.young / (2.0 * (1.0 + SKIN_NU))
    if not np.allclose(fields.lambda_, expected_lambda, rtol=1.0e-13, atol=1.0e-14):
        errors.append("Lambda is inconsistent with Young's modulus and Poisson ratio")
    if not np.allclose(fields.mu, expected_mu, rtol=1.0e-13, atol=1.0e-14):
        errors.append("Mu is inconsistent with Young's modulus and Poisson ratio")
    if not np.allclose(
        fields.activation_inv[:, 0], fields.activation_diag, rtol=0.0, atol=0.0
    ) or not np.allclose(
        fields.activation_inv[:, 1], fields.activation_diag, rtol=0.0, atol=0.0
    ):
        errors.append("ActivationInv in-plane diagonal does not match helper field")
    if not np.allclose(fields.activation_inv[:, 2], 0.0, rtol=0.0, atol=0.0):
        errors.append("ActivationInv out-of-plane entry is not zero")
    expected_stress_free = np.reciprocal(np.square(1.0 + fields.activation_diag))
    if not np.allclose(
        fields.stress_free_area_ratio,
        expected_stress_free,
        rtol=1.0e-13,
        atol=1.0e-14,
    ):
        errors.append("stress-free area ratio is inconsistent with ActivationInv")
    expected_min = SKIN_E * candidate.young_min_scale
    if (
        fields.young.min() < expected_min - tolerance
        or fields.young.max() > SKIN_E + tolerance
    ):
        errors.append("Young's modulus escaped the configured range")
    if candidate.young_min_scale == 1.0 and not np.allclose(
        fields.young, SKIN_E, rtol=0.0, atol=tolerance
    ):
        errors.append("unit Young's-modulus scale did not preserve the baseline")
    if candidate.prestrain_gain == 0.0 and not np.allclose(
        fields.activation_inv, 0.0, rtol=0.0, atol=tolerance
    ):
        errors.append("zero prestrain gain did not preserve zero prestrain")

    e_jump = metrics["field/E_edge_jump_MPa"]
    activation_jump = metrics["field/activation_edge_jump"]
    if float(e_jump["max"]) > max_e_edge_jump:
        errors.append(
            f"E edge jump {e_jump['max']:.6g} exceeds {max_e_edge_jump:.6g} MPa"
        )
    if float(activation_jump["max"]) > max_activation_edge_jump:
        errors.append(
            "ActivationInv edge jump "
            f"{activation_jump['max']:.6g} exceeds {max_activation_edge_jump:.6g}"
        )
    if float(e_jump["edge_length_weighted_rms"]) > max_e_edge_rms:
        errors.append(
            "E edge-jump weighted RMS "
            f"{e_jump['edge_length_weighted_rms']:.6g} exceeds "
            f"{max_e_edge_rms:.6g} MPa"
        )
    if float(activation_jump["edge_length_weighted_rms"]) > max_activation_edge_rms:
        errors.append(
            "ActivationInv edge-jump weighted RMS "
            f"{activation_jump['edge_length_weighted_rms']:.6g} exceeds "
            f"{max_activation_edge_rms:.6g}"
        )
    for sign in ("expansion", "contraction"):
        components = metrics[f"field/{sign}_components"]
        if int(components["singleton_components"]) > max_singleton_components:
            errors.append(
                f"{sign} singleton components {components['singleton_components']} "
                f"exceed {max_singleton_components}"
            )
        if (
            float(components["small_le4_area_fraction_of_eligible"])
            > max_small_component_area_fraction
        ):
            errors.append(
                f"{sign} small-component area fraction "
                f"{components['small_le4_area_fraction_of_eligible']:.6g} exceeds "
                f"{max_small_component_area_fraction:.6g}"
            )
    return errors


def make_candidate_skin(
    geometry: SurfaceGeometryBasis,
    signed_field: SignedHeatField,
    candidate: MaterialCandidate,
    *,
    max_e_edge_jump: float,
    max_activation_edge_jump: float,
    max_e_edge_rms: float,
    max_activation_edge_rms: float,
    max_singleton_components: int,
    max_small_component_area_fraction: float,
) -> tuple[pv.PolyData, dict[str, Any]]:
    from liblaf.apple.common import ACTIVATION_INV, FRACTION, LAMBDA, MU

    fields = candidate_fields(
        geometry,
        signed_field,
        candidate,
    )
    metrics = candidate_field_metrics(geometry, candidate, fields)
    surface = geometry.surface.copy(deep=True)
    surface.cell_data[LAMBDA.vtk] = fields.lambda_
    surface.cell_data[MU.vtk] = fields.mu
    surface.cell_data[FRACTION.vtk] = np.ones(surface.n_cells, dtype=np.float64)
    surface.cell_data[ACTIVATION_INV.vtk] = fields.activation_inv
    surface.cell_data["RestArea"] = geometry.rest_area
    surface.cell_data["TargetArea"] = geometry.target_area
    surface.cell_data["TargetRestAreaRatio"] = geometry.area_ratio
    surface.cell_data["LogAreaRaw"] = geometry.signed_log_area_raw
    surface.cell_data["LogAreaDeadbanded"] = signed_field.deadbanded
    surface.cell_data["LogAreaCapped"] = signed_field.capped
    surface.cell_data["LogAreaDiffused"] = signed_field.diffused
    surface.cell_data["IsFaceTriangle"] = geometry.face_mask.astype(np.int8)
    surface.cell_data["FiniteTargetTriangle"] = geometry.finite_mask.astype(np.int8)
    surface.cell_data["EligibleMaterialTriangle"] = geometry.eligible_mask.astype(
        np.int8
    )
    surface.cell_data["ExpansionMaterialMask"] = (
        fields.expansion_severity > 0.0
    ).astype(np.int8)
    surface.cell_data["ContractionPrestrainMask"] = (
        fields.contraction_severity > 0.0
    ).astype(np.int8)
    surface.cell_data["ExpansionSeverityLogSoftThreshold"] = fields.expansion_severity
    surface.cell_data["ContractionSeverityLogSoftThreshold"] = (
        fields.contraction_severity
    )
    surface.cell_data["ExpansionWeight"] = fields.expansion_weight
    surface.cell_data["ContractionSeverityLogCapped"] = fields.contraction_log
    surface.cell_data["SkinYoungModulusMPa"] = fields.young
    surface.cell_data["SkinActivationInvDiag"] = fields.activation_diag
    surface.cell_data["StressFreeAreaRatio"] = fields.stress_free_area_ratio

    validation_errors = _validate_candidate(
        geometry=geometry,
        signed_field=signed_field,
        candidate=candidate,
        fields=fields,
        metrics=metrics,
        max_e_edge_jump=max_e_edge_jump,
        max_activation_edge_jump=max_activation_edge_jump,
        max_e_edge_rms=max_e_edge_rms,
        max_activation_edge_rms=max_activation_edge_rms,
        max_singleton_components=max_singleton_components,
        max_small_component_area_fraction=max_small_component_area_fraction,
    )
    metrics = {
        "schema_version": SCHEMA_VERSION,
        "label": candidate.label,
        "young_min_scale": float(candidate.young_min_scale),
        "prestrain_gain": float(candidate.prestrain_gain),
        "area_deadband": signed_field.area_deadband,
        "cap_quantile": signed_field.cap_quantile,
        "diffusion_sigma": signed_field.diffusion_sigma,
        "heat/metrics": signed_field.metrics,
        "skin/base_E_MPa": float(SKIN_E),
        "skin/nu": float(SKIN_NU),
        **metrics,
        "content/n_points": int(surface.n_points),
        "content/n_triangles": int(surface.n_cells),
        "content/topology_sha256": skin_topology_content_hash(surface),
        "content/material_sha256": skin_material_content_hash(surface),
        "content/solver_sha256": skin_solver_content_hash(surface),
        "validation/errors": validation_errors,
        "validation/ok": not validation_errors,
    }
    return surface, metrics
