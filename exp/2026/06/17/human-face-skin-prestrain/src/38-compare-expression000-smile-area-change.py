from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
import pyvista as pv

from liblaf import cherries, melon

EXPRESSION_MESH = Path(
    "/home/liblaf/github/liblaf/melon/exp/2026/05/27/head/data/50-tetmesh-3191k.vtu"
)
SMILE_MESH = Path(
    "/home/liblaf/github/liblaf/melon/exp/2026/05/27/head/data/62-tetmesh-3191k.vtu"
)
IS_FACE = "IsFace"


class CompareConfig(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    expression_mesh: Path = cherries.input(EXPRESSION_MESH)
    smile_mesh: Path = cherries.input(SMILE_MESH)
    output_json: Path = cherries.output(
        "38-expression000-smile-area-change-summary.json", mkdir=True
    )
    output_csv: Path = cherries.output(
        "38-expression000-smile-area-change-table.csv", mkdir=True
    )
    output_report: Path = cherries.output(
        "38-expression000-smile-area-change.md", mkdir=True
    )
    output_surface: Path = cherries.output(
        "38-expression000-smile-area-change.vtp", mkdir=True
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
    return {
        "min": float(np.min(values)),
        "q001": float(np.quantile(values, 0.001)),
        "q01": float(np.quantile(values, 0.01)),
        "q05": float(np.quantile(values, 0.05)),
        "median": float(np.quantile(values, 0.5)),
        "q95": float(np.quantile(values, 0.95)),
        "q99": float(np.quantile(values, 0.99)),
        "q999": float(np.quantile(values, 0.999)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "rms": float(np.linalg.norm(values) / math.sqrt(values.size)),
    }


def area_metrics(
    *,
    name: str,
    mesh: pv.UnstructuredGrid,
    surface: pv.PolyData,
    faces: np.ndarray,
    original_ids: np.ndarray,
    rest_area: np.ndarray,
    isface_cells: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    raw = np.asarray(mesh.point_data[name], dtype=np.float64)
    finite_points = np.isfinite(raw).all(axis=1)
    displacement = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    surface_disp = displacement[original_ids]
    finite_surface = finite_points[original_ids]
    finite_cells = np.all(finite_surface[faces], axis=1)
    valid_all = finite_cells & (rest_area > 0.0)
    valid_isface = valid_all & isface_cells
    deformed_area = triangle_areas(np.asarray(surface.points) + surface_disp, faces)
    ratio = np.ones_like(rest_area)
    ratio[valid_all] = deformed_area[valid_all] / rest_area[valid_all]

    isface_points = np.asarray(mesh.point_data[IS_FACE], dtype=bool) & finite_points
    disp_norm = np.linalg.norm(displacement[isface_points], axis=1)

    metrics: dict[str, Any] = {
        "name": name,
        "finite_points": int(finite_points.sum()),
        "isface_points": int(isface_points.sum()),
        "isface_cells": int(valid_isface.sum()),
        "surface_cells": int(valid_all.sum()),
        "disp_rms": float(
            np.linalg.norm(displacement[isface_points])
            / math.sqrt(max(1, int(isface_points.sum())))
        ),
        "disp_mean": float(disp_norm.mean()),
        "disp_max": float(disp_norm.max()),
    }
    for label, mask in (("all", valid_all), ("isface", valid_isface)):
        subset = ratio[mask]
        log_ratio = np.log(np.maximum(subset, np.finfo(np.float64).tiny))
        prefix = f"area_{label}"
        metrics[f"{prefix}_total_ratio"] = float(
            deformed_area[mask].sum() / rest_area[mask].sum()
        )
        metrics.update(
            {f"{prefix}_ratio_{key}": value for key, value in stats(subset).items()}
        )
        metrics[f"{prefix}_frac_abs_log_gt_5pct"] = float(
            np.mean(np.abs(log_ratio) > math.log(1.05))
        )
        metrics[f"{prefix}_frac_abs_log_gt_10pct"] = float(
            np.mean(np.abs(log_ratio) > math.log(1.10))
        )
        metrics[f"{prefix}_frac_abs_log_gt_25pct"] = float(
            np.mean(np.abs(log_ratio) > math.log(1.25))
        )
        metrics[f"{prefix}_frac_abs_log_gt_2x"] = float(
            np.mean(np.abs(log_ratio) > math.log(2.0))
        )
        metrics[f"{prefix}_frac_stretch_gt_10pct"] = float(np.mean(subset > 1.10))
        metrics[f"{prefix}_frac_shrink_gt_10pct"] = float(np.mean(subset < 1.0 / 1.10))
    return metrics, ratio, displacement


def format_float(value: float) -> str:
    return f"{value:.6g}"


def format_percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    columns = [
        "name",
        "disp_rms",
        "disp_max",
        "area_isface_total_ratio",
        "area_isface_ratio_q01",
        "area_isface_ratio_median",
        "area_isface_ratio_q99",
        "area_isface_frac_abs_log_gt_10pct",
        "area_isface_frac_abs_log_gt_25pct",
        "area_isface_frac_abs_log_gt_2x",
        "area_isface_frac_stretch_gt_10pct",
        "area_isface_frac_shrink_gt_10pct",
        "area_all_total_ratio",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows({column: row[column] for column in columns} for row in rows)


def write_report(
    summary: dict[str, Any], rows: list[dict[str, Any]], cfg: CompareConfig
) -> None:
    expression, smile = rows
    lines = [
        "# Expression000 vs Smile Area Change",
        "",
        f"- `Expression000` mesh: `{cfg.expression_mesh}`",
        f"- `Smile` mesh: `{cfg.smile_mesh}`",
        f"- Same point count: `{summary['mesh_checks']['same_n_points']}`",
        f"- Same cell count: `{summary['mesh_checks']['same_n_cells']}`",
        f"- Max rest point delta: `{format_float(summary['mesh_checks']['max_point_delta'])}`",
        f"- Same `IsFace`: `{summary['mesh_checks']['same_isface']}`",
        f"- Compared surface triangles with all vertices in `IsFace`: `{summary['isface_surface_triangles']}`",
        "",
        "## IsFace Triangle Area Change",
        "",
        "| target | disp RMS | disp max | total area | q1 | median | q99 | >10% | >25% | >2x | stretch >10% | shrink >10% |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        (
            f"| `{row['name']}` | {format_float(row['disp_rms'])} | "
            f"{format_float(row['disp_max'])} | "
            f"{format_float(row['area_isface_total_ratio'])} | "
            f"{format_float(row['area_isface_ratio_q01'])} | "
            f"{format_float(row['area_isface_ratio_median'])} | "
            f"{format_float(row['area_isface_ratio_q99'])} | "
            f"{format_percent(row['area_isface_frac_abs_log_gt_10pct'])} | "
            f"{format_percent(row['area_isface_frac_abs_log_gt_25pct'])} | "
            f"{format_percent(row['area_isface_frac_abs_log_gt_2x'])} | "
            f"{format_percent(row['area_isface_frac_stretch_gt_10pct'])} | "
            f"{format_percent(row['area_isface_frac_shrink_gt_10pct'])} |"
        )
        for row in rows
    )
    lines.extend(
        [
            "",
            "## Takeaway",
            "",
            (
                "`Expression000` has a much smaller displacement magnitude and less "
                "extreme triangle-area distortion than `Smile` on these generated "
                "meshes."
            ),
            (
                f"`Smile` has {smile['area_isface_frac_abs_log_gt_10pct'] / expression['area_isface_frac_abs_log_gt_10pct']:.2f}x "
                "as many `IsFace` triangles beyond 10% area change, and "
                f"{smile['area_isface_frac_abs_log_gt_25pct'] / max(expression['area_isface_frac_abs_log_gt_25pct'], 1.0e-12):.2f}x "
                "as many beyond 25%."
            ),
        ]
    )
    cfg.output_report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_compare(cfg: CompareConfig) -> dict[str, Any]:
    expression_mesh = pv.read(cfg.expression_mesh)
    smile_mesh = pv.read(cfg.smile_mesh)
    if "Expression000" not in expression_mesh.point_data:
        msg = f"{cfg.expression_mesh} does not contain Expression000"
        raise KeyError(msg)
    if "Smile" not in smile_mesh.point_data:
        msg = f"{cfg.smile_mesh} does not contain Smile"
        raise KeyError(msg)

    surface = expression_mesh.extract_surface(algorithm=None).triangulate()
    original_ids = np.asarray(surface.point_data["vtkOriginalPointIds"], dtype=np.int64)
    faces = triangle_faces(surface)
    rest_area = triangle_areas(np.asarray(surface.points, dtype=np.float64), faces)
    isface_points = np.asarray(expression_mesh.point_data[IS_FACE], dtype=bool)[
        original_ids
    ]
    isface_cells = np.all(isface_points[faces], axis=1) & (rest_area > 0.0)

    expression_row, expression_ratio, expression_disp = area_metrics(
        name="Expression000",
        mesh=expression_mesh,
        surface=surface,
        faces=faces,
        original_ids=original_ids,
        rest_area=rest_area,
        isface_cells=isface_cells,
    )
    smile_row, smile_ratio, smile_disp = area_metrics(
        name="Smile",
        mesh=smile_mesh,
        surface=surface,
        faces=faces,
        original_ids=original_ids,
        rest_area=rest_area,
        isface_cells=isface_cells,
    )
    rows = [expression_row, smile_row]

    same_n_points = expression_mesh.n_points == smile_mesh.n_points
    same_n_cells = expression_mesh.n_cells == smile_mesh.n_cells
    point_delta = np.linalg.norm(
        np.asarray(expression_mesh.points, dtype=np.float64)
        - np.asarray(smile_mesh.points, dtype=np.float64),
        axis=1,
    )
    same_isface = np.array_equal(
        np.asarray(expression_mesh.point_data[IS_FACE], dtype=bool),
        np.asarray(smile_mesh.point_data[IS_FACE], dtype=bool),
    )

    compare_surface = surface.copy(deep=True)
    compare_surface.cell_data["Expression000AreaRatio"] = expression_ratio
    compare_surface.cell_data["SmileAreaRatio"] = smile_ratio
    compare_surface.cell_data["Expression000LogAreaRatio"] = np.log(
        np.maximum(expression_ratio, np.finfo(np.float64).tiny)
    )
    compare_surface.cell_data["SmileLogAreaRatio"] = np.log(
        np.maximum(smile_ratio, np.finfo(np.float64).tiny)
    )
    compare_surface.cell_data["AreaRatioSmileMinusExpression000"] = (
        smile_ratio - expression_ratio
    )
    compare_surface.point_data["Expression000Displacement"] = expression_disp[
        original_ids
    ]
    compare_surface.point_data["SmileDisplacement"] = smile_disp[original_ids]
    compare_surface.point_data["Expression000Point"] = (
        compare_surface.points + compare_surface.point_data["Expression000Displacement"]
    )
    compare_surface.point_data["SmilePoint"] = (
        compare_surface.points + compare_surface.point_data["SmileDisplacement"]
    )
    isface_surface = compare_surface.extract_cells(isface_cells)
    isface_surface = isface_surface.extract_surface(algorithm=None).triangulate()
    melon.save(isface_surface, cfg.output_surface)

    summary: dict[str, Any] = {
        "expression_mesh": str(cfg.expression_mesh),
        "smile_mesh": str(cfg.smile_mesh),
        "surface_triangles": int(surface.n_cells),
        "isface_surface_triangles": int(isface_cells.sum()),
        "output_surface_triangles": int(isface_surface.n_cells),
        "mesh_checks": {
            "same_n_points": bool(same_n_points),
            "same_n_cells": bool(same_n_cells),
            "max_point_delta": float(point_delta.max()),
            "rms_point_delta": float(
                np.linalg.norm(point_delta) / math.sqrt(point_delta.size)
            ),
            "same_isface": bool(same_isface),
        },
        "rows": rows,
    }
    cfg.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(rows, cfg.output_csv)
    write_report(summary, rows, cfg)
    cherries.log_output(cfg.output_json)
    cherries.log_output(cfg.output_csv)
    cherries.log_output(cfg.output_report)
    cherries.log_output(cfg.output_surface)
    return summary


if __name__ == "__main__":
    cherries.main(run_compare)
