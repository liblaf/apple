from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
from _human_face_mesh import surface_original_ids


def add_metric_fields(
    mesh: pv.UnstructuredGrid, metrics: dict[str, float | int | bool | str]
) -> None:
    for name, value in metrics.items():
        if isinstance(value, str):
            continue
        mesh.field_data[name] = np.asarray([value])


def make_result_mesh(
    mesh: pv.UnstructuredGrid,
    target: np.ndarray,
    mask: np.ndarray,
    displacement: np.ndarray,
    activation_inv: np.ndarray,
    metrics: dict[str, float | int | bool | str],
) -> pv.UnstructuredGrid:
    from liblaf.apple.common import ACTIVATION_INV

    result = mesh.copy(deep=True)
    error = displacement - target
    result.point_data["Displacement"] = displacement
    result.point_data["TargetDisplacement"] = target
    result.point_data["LossMask"] = mask.astype(np.int8)
    result.point_data["DisplacementError"] = error
    result.point_data["DisplacementErrorNorm"] = np.linalg.norm(error, axis=1)
    result.point_data["DeformedPoint"] = result.points + displacement
    result.point_data["TargetPoint"] = result.points + target
    result.cell_data[ACTIVATION_INV.vtk] = activation_inv
    result.cell_data["RecoveredActivationInv"] = activation_inv
    result.cell_data["RecoveredActivationInvNorm"] = np.linalg.norm(
        activation_inv, axis=1
    )
    add_metric_fields(result, metrics)
    return result


def sanitize_vtkhdf_name(name: str) -> str:
    return name.replace("/", "_").replace(".", "_")


def copy_attrs_for_vtkhdf(
    source: pv.DataSetAttributes, target: pv.DataSetAttributes
) -> None:
    for name in list(target.keys()):
        del target[name]
    for name, value in source.items():
        arr = np.asarray(value)
        if arr.dtype.kind in {"O", "U", "S"}:
            continue
        target[sanitize_vtkhdf_name(name)] = arr


def make_history_mesh(mesh: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    result = mesh.copy(deep=True)
    copy_attrs_for_vtkhdf(mesh.point_data, result.point_data)
    copy_attrs_for_vtkhdf(mesh.cell_data, result.cell_data)
    copy_attrs_for_vtkhdf(mesh.field_data, result.field_data)
    return result


def surface_triangles(mesh: pv.UnstructuredGrid) -> np.ndarray:
    surface, original_ids = surface_original_ids(mesh)
    faces = np.asarray(surface.faces, dtype=np.int64).reshape(-1, 4)
    return original_ids[faces[:, 1:]]


def unique_edges(triangles: np.ndarray) -> np.ndarray:
    edges = np.vstack(
        (
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        )
    )
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def surface_edges_for_mask(mesh: pv.UnstructuredGrid, mask: np.ndarray) -> np.ndarray:
    triangles = surface_triangles(mesh)
    selected = triangles[np.all(mask[triangles], axis=1)]
    if selected.size == 0:
        edges = unique_edges(triangles)
        edges = edges[np.all(mask[edges], axis=1)]
        return edges.astype(np.int64)
    return unique_edges(selected).astype(np.int64)


def bumpiness_metrics(
    *,
    mask: np.ndarray,
    edges: np.ndarray,
    displacement: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    ids = np.flatnonzero(mask)
    residual = displacement - target
    if edges.size == 0:
        return {
            "bumpiness/displacement_edge_rms": math.nan,
            "bumpiness/residual_edge_rms": math.nan,
            "bumpiness/displacement_laplacian_rms": math.nan,
            "bumpiness/residual_laplacian_rms": math.nan,
            "bumpiness/displacement_norm_std": math.nan,
            "bumpiness/residual_norm_std": math.nan,
        }
    disp_edge = displacement[edges[:, 0]] - displacement[edges[:, 1]]
    residual_edge = residual[edges[:, 0]] - residual[edges[:, 1]]
    n_points = displacement.shape[0]
    neighbor_sum = np.zeros_like(displacement)
    residual_neighbor_sum = np.zeros_like(residual)
    neighbor_count = np.zeros(n_points, dtype=np.float64)
    np.add.at(neighbor_sum, edges[:, 0], displacement[edges[:, 1]])
    np.add.at(neighbor_sum, edges[:, 1], displacement[edges[:, 0]])
    np.add.at(residual_neighbor_sum, edges[:, 0], residual[edges[:, 1]])
    np.add.at(residual_neighbor_sum, edges[:, 1], residual[edges[:, 0]])
    np.add.at(neighbor_count, edges[:, 0], 1.0)
    np.add.at(neighbor_count, edges[:, 1], 1.0)
    active = neighbor_count > 0.0
    disp_lap = np.zeros_like(displacement)
    residual_lap = np.zeros_like(residual)
    disp_lap[active] = (
        displacement[active] - neighbor_sum[active] / neighbor_count[active, None]
    )
    residual_lap[active] = (
        residual[active] - residual_neighbor_sum[active] / neighbor_count[active, None]
    )
    return {
        "bumpiness/displacement_edge_rms": float(
            np.linalg.norm(disp_edge) / math.sqrt(edges.shape[0])
        ),
        "bumpiness/residual_edge_rms": float(
            np.linalg.norm(residual_edge) / math.sqrt(edges.shape[0])
        ),
        "bumpiness/displacement_laplacian_rms": float(
            np.linalg.norm(disp_lap[ids]) / math.sqrt(ids.size)
        ),
        "bumpiness/residual_laplacian_rms": float(
            np.linalg.norm(residual_lap[ids]) / math.sqrt(ids.size)
        ),
        "bumpiness/displacement_norm_std": float(
            np.linalg.norm(displacement[ids], axis=1).std()
        ),
        "bumpiness/residual_norm_std": float(
            np.linalg.norm(residual[ids], axis=1).std()
        ),
    }


def format_float(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if not isinstance(value, int | float):
        return str(value)
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return f"{float(value):.6g}"


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "| case | setup | target | baseline complete | baseline evals | baseline best step | baseline best loss mm2 | baseline best RMS mm | stop | best step | best loss mm2 | best loss m2 | error RMS mm | error max mm | error/target | disp lap RMS | residual lap RMS | activation max | LR initial | LR final | LR deviations | min-delta final | forward fails | adjoint fails | last forward | last adjoint |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    lines.extend(
        (
            "| "
            + " | ".join(
                [
                    str(row["case"]),
                    str(row["case/setup"]),
                    str(row["target/name"]),
                    str(row["baseline/completed"]),
                    format_float(row["baseline/evaluations"]),
                    format_float(row["baseline/best_step"]),
                    format_float(row["baseline/best_loss_mm2"]),
                    format_float(row["baseline/best_error_rms_mm"]),
                    str(row["inverse/stop_reason"]),
                    format_float(row["best/step"]),
                    format_float(row["best/loss_mm2"]),
                    format_float(row["best/loss_m2"]),
                    format_float(row["best/error_rms_mm"]),
                    format_float(row["best/error_max_mm"]),
                    format_float(row["best/error_rms_fraction_of_target"]),
                    format_float(row["bumpiness/displacement_laplacian_rms"]),
                    format_float(row["bumpiness/residual_laplacian_rms"]),
                    format_float(row["activation_inv/max_abs"]),
                    format_float(row["inverse/lr_initial"]),
                    format_float(row["inverse/lr_final"]),
                    format_float(row["baseline/lr_deviation_count"]),
                    format_float(row["inverse/effective_min_delta_abs_final"]),
                    format_float(row["inverse/forward_fail_count"]),
                    format_float(row["inverse/adjoint_fail_count"]),
                    str(row["last/forward/result"]),
                    str(row["last/adjoint/result"]),
                ]
            )
            + " |"
        )
        for row in rows
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
