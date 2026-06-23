from __future__ import annotations

from typing import Any

import numpy as np
import pyvista as pv
from _human_face_config import (
    ACTIVE_FRACTION,
    ACTIVE_FRACTION_TOL,
    APONEUROSIS_FRACTION,
    FAT_FRACTION,
    FRACTION_SUM,
    IN_FACE_CONVEX,
    IS_FACE,
    IS_FIXED,
    MUSCLE_FRACTION,
    SMILE_LOSS_MASK,
    SMILE_TARGET,
    TARGET_FINITE,
)


def tetra_cells(mesh: pv.UnstructuredGrid) -> np.ndarray:
    if pv.CellType.TETRA not in mesh.cells_dict:
        msg = "expected a tetrahedral unstructured grid"
        raise ValueError(msg)
    return np.asarray(mesh.cells_dict[pv.CellType.TETRA], dtype=np.int64)


def tetra_signed_volumes(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    p0 = points[tets[:, 0]]
    p1 = points[tets[:, 1]]
    p2 = points[tets[:, 2]]
    p3 = points[tets[:, 3]]
    return np.einsum("ij,ij->i", np.cross(p1 - p0, p2 - p0), p3 - p0) / 6.0


def tetra_volumes(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    return np.abs(tetra_signed_volumes(points, tets))


def orient_tetra_mesh(mesh: pv.UnstructuredGrid) -> tuple[pv.UnstructuredGrid, int]:
    tets = tetra_cells(mesh).copy()
    points = np.asarray(mesh.points, dtype=np.float64)
    signed = tetra_signed_volumes(points, tets)
    flipped = signed < 0.0
    n_flipped = int(flipped.sum())
    if n_flipped:
        tets[flipped, 2], tets[flipped, 3] = tets[flipped, 3], tets[flipped, 2].copy()
    cells = np.empty((tets.shape[0], 5), dtype=np.int64)
    cells[:, 0] = 4
    cells[:, 1:] = tets
    cell_types = np.full(tets.shape[0], int(pv.CellType.TETRA), dtype=np.uint8)
    oriented = pv.UnstructuredGrid(cells.ravel(), cell_types, points)
    for name, values in mesh.point_data.items():
        oriented.point_data[name] = np.asarray(values)
    for name, values in mesh.cell_data.items():
        oriented.cell_data[name] = np.asarray(values)
    for name, values in mesh.field_data.items():
        oriented.field_data[name] = np.asarray(values)
    return oriented, n_flipped


def extract_simulation_mesh(
    mesh: pv.UnstructuredGrid,
) -> tuple[pv.UnstructuredGrid, dict[str, Any]]:
    if IN_FACE_CONVEX not in mesh.cell_data:
        msg = f"missing required cell data {IN_FACE_CONVEX!r}"
        raise KeyError(msg)
    in_face_convex = np.asarray(mesh.cell_data[IN_FACE_CONVEX], dtype=bool)
    if in_face_convex.shape != (mesh.n_cells,):
        msg = (
            f"{IN_FACE_CONVEX!r} must have one value per cell, got "
            f"{in_face_convex.shape}"
        )
        raise ValueError(msg)
    if not np.any(in_face_convex):
        msg = f"{IN_FACE_CONVEX!r} selected no tetrahedra"
        raise ValueError(msg)

    selected = mesh.extract_cells(in_face_convex)
    if not isinstance(selected, pv.UnstructuredGrid):
        selected = selected.cast_to_unstructured_grid()
    return selected, {
        "source/n_points": int(mesh.n_points),
        "source/n_tets": int(mesh.n_cells),
        "source/n_in_face_convex_tets": int(in_face_convex.sum()),
        "simulation/subset": IN_FACE_CONVEX,
        "simulation/n_points_before_orient": int(selected.n_points),
        "simulation/n_tets_before_orient": int(selected.n_cells),
    }


def lame_parameters(E: float, nu: float) -> tuple[float, float]:
    lambda_ = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    return float(lambda_), float(mu)


def finite_vec3(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        msg = f"expected vec3 data, got shape {values.shape}"
        raise ValueError(msg)
    return np.isfinite(values).all(axis=1)


def surface_original_ids(mesh: pv.UnstructuredGrid) -> tuple[pv.PolyData, np.ndarray]:
    surface = mesh.extract_surface(algorithm=None).triangulate()
    if "vtkOriginalPointIds" not in surface.point_data:
        msg = "extract_surface did not produce vtkOriginalPointIds"
        raise KeyError(msg)
    original_ids = np.asarray(surface.point_data["vtkOriginalPointIds"], dtype=np.int64)
    return surface, original_ids


def add_required_fields(mesh: pv.UnstructuredGrid) -> dict[str, Any]:
    from liblaf.apple.common import ACTIVATION, ACTIVATION_INV, FIXED_MASK, FIXED_VALUE

    for name in (APONEUROSIS_FRACTION, FAT_FRACTION, MUSCLE_FRACTION):
        if name not in mesh.cell_data:
            msg = f"missing required cell data {name!r}"
            raise KeyError(msg)
        mesh.cell_data[name] = np.nan_to_num(
            np.asarray(mesh.cell_data[name], dtype=np.float64), nan=0.0
        )

    points = np.asarray(mesh.points, dtype=np.float64)
    tets = tetra_cells(mesh)
    volumes = tetra_volumes(points, tets)
    fractions = np.column_stack(
        (
            np.asarray(mesh.cell_data[APONEUROSIS_FRACTION], dtype=np.float64),
            np.asarray(mesh.cell_data[FAT_FRACTION], dtype=np.float64),
            np.asarray(mesh.cell_data[MUSCLE_FRACTION], dtype=np.float64),
        )
    )
    fraction_sum = fractions.sum(axis=1)
    active = fractions[:, 2] > ACTIVE_FRACTION_TOL

    mesh.cell_data["Volume"] = volumes
    mesh.cell_data[FRACTION_SUM] = fraction_sum
    mesh.cell_data[ACTIVE_FRACTION] = fractions[:, 2]
    mesh.cell_data["ActivationMask"] = active.astype(np.int8)
    mesh.cell_data[ACTIVATION.vtk] = np.zeros((mesh.n_cells, 6), dtype=np.float64)
    mesh.cell_data[ACTIVATION_INV.vtk] = np.zeros((mesh.n_cells, 6), dtype=np.float64)

    if IS_FIXED not in mesh.point_data:
        msg = f"missing required point data {IS_FIXED!r}"
        raise KeyError(msg)
    fixed = np.asarray(mesh.point_data[IS_FIXED], dtype=bool)
    mesh.point_data[FIXED_MASK.vtk] = np.repeat(fixed[:, None], 3, axis=1)
    mesh.point_data[FIXED_VALUE.vtk] = np.zeros((mesh.n_points, 3), dtype=np.float64)

    if SMILE_TARGET not in mesh.point_data:
        msg = f"missing required point data {SMILE_TARGET!r}"
        raise KeyError(msg)
    if IS_FACE not in mesh.point_data:
        msg = f"missing required point data {IS_FACE!r}"
        raise KeyError(msg)
    smile = np.asarray(mesh.point_data[SMILE_TARGET], dtype=np.float64)
    finite = finite_vec3(smile)
    face = np.asarray(mesh.point_data[IS_FACE], dtype=bool)
    smile_loss = face & finite
    mesh.point_data[TARGET_FINITE] = finite.astype(np.int8)
    mesh.point_data[SMILE_LOSS_MASK] = smile_loss.astype(np.int8)

    surface, original_ids = surface_original_ids(mesh)

    return {
        "n_points": int(mesh.n_points),
        "n_tets": int(mesh.n_cells),
        "n_surface_points": int(np.unique(original_ids).size),
        "n_surface_triangles": int(surface.n_cells),
        "n_fixed_points": int(fixed.sum()),
        "n_smile_finite_points": int(finite.sum()),
        "n_smile_nan_points": int((~finite).sum()),
        "n_smile_loss_points": int(smile_loss.sum()),
        "n_active_tets": int(active.sum()),
    }


def geometry_summary(mesh: pv.UnstructuredGrid) -> dict[str, float]:
    fractions = np.column_stack(
        (
            np.asarray(mesh.cell_data[APONEUROSIS_FRACTION], dtype=np.float64),
            np.asarray(mesh.cell_data[FAT_FRACTION], dtype=np.float64),
            np.asarray(mesh.cell_data[MUSCLE_FRACTION], dtype=np.float64),
        )
    )
    volumes = np.asarray(mesh.cell_data["Volume"], dtype=np.float64)
    weighted = fractions * volumes[:, None]
    return {
        "volume/total": float(volumes.sum()),
        "volume/aponeurosis": float(weighted[:, 0].sum()),
        "volume/fat": float(weighted[:, 1].sum()),
        "volume/muscle": float(weighted[:, 2].sum()),
        "fraction_sum/min": float(fractions.sum(axis=1).min()),
        "fraction_sum/max": float(fractions.sum(axis=1).max()),
        "fraction_sum/mean": float(fractions.sum(axis=1).mean()),
    }
