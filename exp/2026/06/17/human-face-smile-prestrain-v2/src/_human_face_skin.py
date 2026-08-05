from __future__ import annotations

import math
from typing import Any

import numpy as np
import pyvista as pv
from _human_face_config import (
    IS_FACE,
    SKIN_E,
    SKIN_NU,
    SMILE_TARGET,
    InverseCase,
)
from _human_face_mesh import finite_vec3, lame_parameters, surface_original_ids


def triangle_area(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    p0 = points[triangles[:, 0]]
    p1 = points[triangles[:, 1]]
    p2 = points[triangles[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)


def skin_prestrain_fields(  # noqa: PLR0915
    mesh: pv.UnstructuredGrid,
    *,
    area_ratio_floor: float,
    constant_tightening: float = 0.0,
) -> tuple[pv.PolyData, dict[str, Any]]:
    from liblaf.apple.common import (
        ACTIVATION_INV,
        FRACTION,
        GLOBAL_POINT_ID,
        LAMBDA,
        MU,
    )

    if area_ratio_floor <= 0.0:
        msg = f"area_ratio_floor must be positive, got {area_ratio_floor}"
        raise ValueError(msg)
    if constant_tightening < 0.0 or constant_tightening >= 1.0:
        msg = f"constant_tightening must be in [0, 1), got {constant_tightening}"
        raise ValueError(msg)

    surface, original_ids = surface_original_ids(mesh)
    faces = np.asarray(surface.faces, dtype=np.int64).reshape(-1, 4)
    triangles = faces[:, 1:]
    rest_points = np.asarray(surface.points, dtype=np.float64)
    smile = np.asarray(mesh.point_data[SMILE_TARGET], dtype=np.float64)[original_ids]
    target_points = rest_points + np.nan_to_num(smile, nan=0.0, posinf=0.0, neginf=0.0)

    rest_area = triangle_area(rest_points, triangles)
    target_area = triangle_area(target_points, triangles)
    valid_rest_area = rest_area > np.finfo(np.float64).eps
    area_ratio = np.ones_like(rest_area)
    area_ratio[valid_rest_area] = (
        target_area[valid_rest_area] / rest_area[valid_rest_area]
    )

    face_points = np.asarray(mesh.point_data[IS_FACE], dtype=bool)[original_ids]
    finite_points = finite_vec3(smile)
    is_face_triangle = np.all(face_points[triangles], axis=1)
    finite_triangle = np.all(finite_points[triangles], axis=1)
    active_prestrain = (
        is_face_triangle & finite_triangle & valid_rest_area & (area_ratio < 1.0)
    )

    a_est = np.ones(surface.n_cells, dtype=np.float64)
    clamped_ratio = np.maximum(area_ratio, area_ratio_floor)
    a_est[active_prestrain] = 1.0 / np.sqrt(clamped_ratio[active_prestrain])
    a_const = np.ones(surface.n_cells, dtype=np.float64)
    a_const[is_face_triangle] = 1.0 / (1.0 - constant_tightening)
    a_total = a_est * a_const
    activation_inv = np.zeros((surface.n_cells, 3), dtype=np.float64)
    activation_inv[:, 0] = a_total - 1.0
    activation_inv[:, 1] = a_total - 1.0
    stress_free_area_ratio = 1.0 / np.square(a_total)

    lambda_, mu = lame_parameters(SKIN_E, SKIN_NU)
    if GLOBAL_POINT_ID.vtk in mesh.point_data:
        mesh_point_ids = np.asarray(
            mesh.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64
        )
    else:
        mesh_point_ids = np.arange(mesh.n_points, dtype=np.int64)
    surface.point_data[GLOBAL_POINT_ID.vtk] = mesh_point_ids[original_ids]
    surface.cell_data[LAMBDA.vtk] = np.full(surface.n_cells, lambda_, dtype=np.float64)
    surface.cell_data[MU.vtk] = np.full(surface.n_cells, mu, dtype=np.float64)
    surface.cell_data[FRACTION.vtk] = np.ones(surface.n_cells, dtype=np.float64)
    surface.cell_data[ACTIVATION_INV.vtk] = activation_inv
    surface.cell_data["RestArea"] = rest_area
    surface.cell_data["TargetArea"] = target_area
    surface.cell_data["TargetRestAreaRatio"] = area_ratio
    surface.cell_data["LogTargetRestAreaRatio"] = np.log(
        np.maximum(area_ratio, np.finfo(np.float64).tiny)
    )
    surface.cell_data["IsFaceTriangle"] = is_face_triangle.astype(np.int8)
    surface.cell_data["FiniteTargetTriangle"] = finite_triangle.astype(np.int8)
    surface.cell_data["ActivePrestrainMask"] = active_prestrain.astype(np.int8)
    surface.cell_data["TargetDerivedActivePrestrainMask"] = active_prestrain.astype(
        np.int8
    )
    surface.cell_data["ConstantTightening"] = np.full(
        surface.n_cells, constant_tightening, dtype=np.float64
    )
    surface.cell_data["ConstantTighteningValue"] = np.where(
        is_face_triangle, constant_tightening, 0.0
    )
    surface.cell_data["ConstantTighteningInvStretch"] = a_const
    surface.cell_data["EstimatedInvStretch"] = a_est
    surface.cell_data["TotalInvStretch"] = a_total
    surface.cell_data["StressFreeAreaRatio"] = stress_free_area_ratio
    surface.cell_data["SkinActivationInvDiag"] = activation_inv[:, 0]
    surface.cell_data["SkinActivationInvNorm"] = np.linalg.norm(activation_inv, axis=1)

    face_rest_area = rest_area[is_face_triangle]
    face_target_area = target_area[is_face_triangle]
    active_area = rest_area[active_prestrain]
    metrics: dict[str, Any] = {
        "skin/prestrain/source": "Smile triangle area ratio on IsFace surface triangles",
        "skin/prestrain/area_ratio_floor": float(area_ratio_floor),
        "skin/prestrain/constant_tightening": float(constant_tightening),
        "skin/prestrain/constant_tightening_inv_stretch": float(
            1.0 / (1.0 - constant_tightening)
        ),
        "skin/surface_triangles": int(surface.n_cells),
        "skin/is_face_triangles": int(is_face_triangle.sum()),
        "skin/finite_is_face_triangles": int(
            (is_face_triangle & finite_triangle).sum()
        ),
        "skin/prestrain_active_triangles": int(active_prestrain.sum()),
        "skin/constant_tightening_triangles": int(
            is_face_triangle.sum() if constant_tightening != 0.0 else 0
        ),
        "skin/prestrain_active_rest_area": float(active_area.sum()),
        "skin/prestrain_active_rest_area_fraction": float(
            active_area.sum() / rest_area.sum()
        )
        if rest_area.sum() > 0.0
        else math.nan,
        "skin/area_ratio_total": float(target_area.sum() / rest_area.sum())
        if rest_area.sum() > 0.0
        else math.nan,
        "skin/is_face_area_ratio_total": float(
            face_target_area.sum() / face_rest_area.sum()
        )
        if face_rest_area.sum() > 0.0
        else math.nan,
        "skin/area_ratio_min": float(area_ratio[is_face_triangle].min())
        if np.any(is_face_triangle)
        else math.nan,
        "skin/area_ratio_median": float(np.median(area_ratio[is_face_triangle]))
        if np.any(is_face_triangle)
        else math.nan,
        "skin/area_ratio_max": float(area_ratio[is_face_triangle].max())
        if np.any(is_face_triangle)
        else math.nan,
        "skin/activation_inv_diag_max": float(activation_inv[:, 0].max()),
        "skin/activation_inv_norm_rms": float(
            np.linalg.norm(activation_inv) / math.sqrt(max(1, activation_inv.size))
        ),
        "skin/stress_free_area_ratio_min": float(stress_free_area_ratio.min()),
        "skin/stress_free_area_ratio_mean": float(stress_free_area_ratio.mean()),
    }
    return surface, metrics


def skin_for_case(
    mesh: pv.UnstructuredGrid, case: InverseCase, *, area_ratio_floor: float
) -> tuple[pv.PolyData, dict[str, Any]]:
    from liblaf.apple.common import ACTIVATION_INV

    surface, metrics = skin_prestrain_fields(
        mesh,
        area_ratio_floor=area_ratio_floor,
        constant_tightening=case.skin_constant_tightening,
    )
    if not case.skin_prestrain_enabled:
        activation_inv = np.zeros((surface.n_cells, 3), dtype=np.float64)
        surface.cell_data[ACTIVATION_INV.vtk] = activation_inv
        surface.cell_data["ActivePrestrainMask"] = np.zeros(
            surface.n_cells, dtype=np.int8
        )
        surface.cell_data["TargetDerivedActivePrestrainMask"] = np.zeros(
            surface.n_cells, dtype=np.int8
        )
        surface.cell_data["ConstantTighteningValue"] = np.zeros(
            surface.n_cells, dtype=np.float64
        )
        surface.cell_data["ConstantTighteningInvStretch"] = np.ones(
            surface.n_cells, dtype=np.float64
        )
        surface.cell_data["EstimatedInvStretch"] = np.ones(
            surface.n_cells, dtype=np.float64
        )
        surface.cell_data["TotalInvStretch"] = np.ones(
            surface.n_cells, dtype=np.float64
        )
        surface.cell_data["StressFreeAreaRatio"] = np.ones(
            surface.n_cells, dtype=np.float64
        )
        surface.cell_data["SkinActivationInvDiag"] = np.zeros(
            surface.n_cells, dtype=np.float64
        )
        surface.cell_data["SkinActivationInvNorm"] = np.zeros(
            surface.n_cells, dtype=np.float64
        )
        metrics = {
            **metrics,
            "skin/prestrain/source": "zero",
            "skin/prestrain/constant_tightening": 0.0,
            "skin/prestrain/constant_tightening_inv_stretch": 1.0,
            "skin/prestrain_active_triangles": 0,
            "skin/constant_tightening_triangles": 0,
            "skin/prestrain_active_rest_area": 0.0,
            "skin/prestrain_active_rest_area_fraction": 0.0,
            "skin/activation_inv_diag_max": 0.0,
            "skin/activation_inv_norm_rms": 0.0,
            "skin/stress_free_area_ratio_min": 1.0,
            "skin/stress_free_area_ratio_mean": 1.0,
        }
    return surface, metrics


def filtered_isface_skin(surface: pv.PolyData) -> pv.PolyData:
    mask = np.asarray(surface.cell_data["IsFaceTriangle"], dtype=bool)
    filtered = surface.extract_cells(mask)
    if not isinstance(filtered, pv.PolyData):
        filtered = filtered.extract_surface().triangulate()
    return filtered
