from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
import pyvista as pv

from liblaf import cherries, melon

SOURCE_MESH = Path(
    "/home/liblaf/github/liblaf/melon/exp/2026/05/27/head/data/62-tetmesh-3191k.vtu"
)
IS_FACE = "IsFace"
TARGET = "Smile"


class AreaPrestrainConfig(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    input_mesh: Path = cherries.input(SOURCE_MESH)
    target: str = TARGET
    area_ratio_floor: float = 0.1
    output_mesh: Path = cherries.output(
        "39-smile-isface-area-prestrain.vtp", mkdir=True
    )
    output_summary: Path = cherries.output(
        "39-smile-isface-area-prestrain-summary.json", mkdir=True
    )
    output_report: Path = cherries.output(
        "39-smile-isface-area-prestrain.md", mkdir=True
    )


def triangle_faces(surface: pv.PolyData) -> np.ndarray:
    faces = np.asarray(surface.faces, dtype=np.int64).reshape(-1, 4)
    if not np.all(faces[:, 0] == 3):
        msg = "expected triangulated surface"
        raise ValueError(msg)
    return faces[:, 1:]


def triangle_areas(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    p0 = points[faces[:, 0]]
    p1 = points[faces[:, 1]]
    p2 = points[faces[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)


def stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {
            "min": math.nan,
            "q01": math.nan,
            "median": math.nan,
            "q99": math.nan,
            "max": math.nan,
            "mean": math.nan,
            "rms": math.nan,
        }
    return {
        "min": float(np.min(values)),
        "q01": float(np.quantile(values, 0.01)),
        "median": float(np.quantile(values, 0.5)),
        "q99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "rms": float(np.linalg.norm(values) / math.sqrt(values.size)),
    }


def format_float(value: float) -> str:
    return f"{value:.6g}"


def format_percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def write_report(summary: dict[str, Any], cfg: AreaPrestrainConfig) -> None:
    area = summary["target_rest_area_ratio"]
    shrink = summary["estimated_length_prestrain"]
    activation = summary["estimated_activation_inv_diag"]
    lines = [
        "# Smile IsFace Area Prestrain",
        "",
        f"- Mesh: `{cfg.input_mesh}`",
        f"- Target: `{cfg.target}`",
        f"- Area ratio floor: `{cfg.area_ratio_floor}`",
        f"- Output mesh: `{cfg.output_mesh}`",
        f"- Output triangles: `{summary['output_triangles']}`",
        f"- Active contracted prestrain triangles: `{summary['active_prestrain_triangles']}`",
        f"- Total target/rest area ratio: `{format_float(summary['total_target_rest_area_ratio'])}`",
        "",
        "## Cell Fields",
        "",
        "- `TargetRestAreaRatio`: target triangle area divided by rest area.",
        "- `TargetRestLengthRatio`: `sqrt(TargetRestAreaRatio)`.",
        "- `EstimatedStressFreeLengthRatio`: length ratio used by the shrink prestrain.",
        "- `EstimatedLengthPrestrain`: positive length shrink, `1 - EstimatedStressFreeLengthRatio`.",
        "- `EstimatedInvLengthFactor`: actual isotropic `A_inv` diagonal factor.",
        "- `EstimatedActivationInvDiag`: stored skin `ActivationInv` diagonal offset.",
        "",
        "## Summary",
        "",
        "| field | min | q1 | median | q99 | max | mean | rms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        (
            f"| target/rest area | {format_float(area['min'])} | "
            f"{format_float(area['q01'])} | {format_float(area['median'])} | "
            f"{format_float(area['q99'])} | {format_float(area['max'])} | "
            f"{format_float(area['mean'])} | {format_float(area['rms'])} |"
        ),
        (
            f"| length prestrain | {format_percent(shrink['min'])} | "
            f"{format_percent(shrink['q01'])} | {format_percent(shrink['median'])} | "
            f"{format_percent(shrink['q99'])} | {format_percent(shrink['max'])} | "
            f"{format_percent(shrink['mean'])} | {format_percent(shrink['rms'])} |"
        ),
        (
            f"| activation inv diag | {format_float(activation['min'])} | "
            f"{format_float(activation['q01'])} | {format_float(activation['median'])} | "
            f"{format_float(activation['q99'])} | {format_float(activation['max'])} | "
            f"{format_float(activation['mean'])} | {format_float(activation['rms'])} |"
        ),
    ]
    cfg.output_report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(cfg: AreaPrestrainConfig) -> dict[str, Any]:  # noqa: PLR0915
    mesh = pv.read(cfg.input_mesh)
    if not isinstance(mesh, pv.UnstructuredGrid):
        mesh = mesh.cast_to_unstructured_grid()
    if cfg.target not in mesh.point_data:
        msg = f"{cfg.input_mesh} does not contain target point data {cfg.target!r}"
        raise KeyError(msg)
    if IS_FACE not in mesh.point_data:
        msg = f"{cfg.input_mesh} does not contain point data {IS_FACE!r}"
        raise KeyError(msg)

    surface = mesh.extract_surface(algorithm=None).triangulate()
    original_ids = np.asarray(surface.point_data["vtkOriginalPointIds"], dtype=np.int64)
    faces = triangle_faces(surface)
    rest_points = np.asarray(surface.points, dtype=np.float64)
    target_raw = np.asarray(mesh.point_data[cfg.target], dtype=np.float64)
    target_finite = np.isfinite(target_raw).all(axis=1)
    target = np.nan_to_num(target_raw, nan=0.0, posinf=0.0, neginf=0.0)
    surface_target = target[original_ids]
    target_points = rest_points + surface_target

    rest_area = triangle_areas(rest_points, faces)
    target_area = triangle_areas(target_points, faces)
    valid_area = rest_area > 0.0
    area_ratio = np.ones_like(rest_area)
    area_ratio[valid_area] = target_area[valid_area] / rest_area[valid_area]
    length_ratio = np.sqrt(np.maximum(area_ratio, 0.0))

    isface_points = np.asarray(mesh.point_data[IS_FACE], dtype=bool)[original_ids]
    finite_surface = target_finite[original_ids]
    isface_cells = np.all(isface_points[faces], axis=1)
    finite_cells = np.all(finite_surface[faces], axis=1)
    output_cells = isface_cells & valid_area
    active_cells = output_cells & finite_cells & (area_ratio < 1.0)

    clipped_area_ratio = np.ones_like(area_ratio)
    clipped_area_ratio[active_cells] = np.maximum(
        area_ratio[active_cells], cfg.area_ratio_floor
    )
    stress_free_length_ratio = np.ones_like(area_ratio)
    stress_free_length_ratio[active_cells] = np.sqrt(clipped_area_ratio[active_cells])
    length_prestrain = 1.0 - stress_free_length_ratio
    inv_length_factor = np.ones_like(area_ratio)
    inv_length_factor[active_cells] = 1.0 / stress_free_length_ratio[active_cells]
    activation_inv_diag = inv_length_factor - 1.0
    stress_free_area_ratio = stress_free_length_ratio**2

    output = surface.copy(deep=True)
    output.point_data[f"{cfg.target}Displacement"] = surface_target
    output.point_data[f"{cfg.target}TargetPoint"] = target_points
    output.cell_data["RestArea"] = rest_area
    output.cell_data["TargetArea"] = target_area
    output.cell_data["TargetRestAreaRatio"] = area_ratio
    output.cell_data["LogTargetRestAreaRatio"] = np.log(
        np.maximum(area_ratio, np.finfo(np.float64).tiny)
    )
    output.cell_data["TargetRestLengthRatio"] = length_ratio
    output.cell_data["AreaRatioFloor"] = np.full(
        output.n_cells, cfg.area_ratio_floor, dtype=np.float64
    )
    output.cell_data["IsFaceAreaCell"] = isface_cells.astype(np.int8)
    output.cell_data["TargetFiniteCell"] = finite_cells.astype(np.int8)
    output.cell_data["IsContractedPrestrainCell"] = active_cells.astype(np.int8)
    output.cell_data["ClippedPrestrainAreaRatio"] = clipped_area_ratio
    output.cell_data["EstimatedStressFreeAreaRatio"] = stress_free_area_ratio
    output.cell_data["EstimatedStressFreeLengthRatio"] = stress_free_length_ratio
    output.cell_data["EstimatedLengthPrestrain"] = length_prestrain
    output.cell_data["EstimatedInvLengthFactor"] = inv_length_factor
    output.cell_data["EstimatedActivationInvDiag"] = activation_inv_diag

    isface_surface = output.extract_cells(output_cells)
    isface_surface = isface_surface.extract_surface(algorithm=None).triangulate()
    melon.save(isface_surface, cfg.output_mesh)

    cell_area_ratio = np.asarray(isface_surface.cell_data["TargetRestAreaRatio"])
    cell_length_prestrain = np.asarray(
        isface_surface.cell_data["EstimatedLengthPrestrain"]
    )
    cell_activation = np.asarray(isface_surface.cell_data["EstimatedActivationInvDiag"])
    cell_target_area = np.asarray(isface_surface.cell_data["TargetArea"])
    cell_rest_area = np.asarray(isface_surface.cell_data["RestArea"])
    cell_active = np.asarray(
        isface_surface.cell_data["IsContractedPrestrainCell"], dtype=bool
    )
    summary: dict[str, Any] = {
        "input_mesh": str(cfg.input_mesh),
        "target": cfg.target,
        "area_ratio_floor": cfg.area_ratio_floor,
        "surface_triangles": int(surface.n_cells),
        "isface_surface_triangles": int(output_cells.sum()),
        "output_points": int(isface_surface.n_points),
        "output_triangles": int(isface_surface.n_cells),
        "finite_target_output_triangles": int(
            np.asarray(isface_surface.cell_data["TargetFiniteCell"], dtype=bool).sum()
        ),
        "active_prestrain_triangles": int(cell_active.sum()),
        "total_target_rest_area_ratio": float(
            cell_target_area.sum() / cell_rest_area.sum()
        ),
        "target_rest_area_ratio": stats(cell_area_ratio),
        "estimated_length_prestrain": stats(cell_length_prestrain),
        "estimated_length_prestrain_active": stats(cell_length_prestrain[cell_active]),
        "estimated_activation_inv_diag": stats(cell_activation),
        "estimated_activation_inv_diag_active": stats(cell_activation[cell_active]),
    }
    cfg.output_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, cfg)
    return summary


if __name__ == "__main__":
    cherries.main(run)
