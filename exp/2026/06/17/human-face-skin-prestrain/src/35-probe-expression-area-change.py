from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
import pyvista as pv
from _human_face_config import IS_FACE, SOURCE_MESH

from liblaf import cherries


class ProbeConfig(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    input_mesh: Path = cherries.input(SOURCE_MESH)
    output_json: Path = cherries.output("35-expression-area-probe.json", mkdir=True)
    output_csv: Path = cherries.output("35-expression-area-probe.csv", mkdir=True)
    output_report: Path = cherries.output("35-expression-area-probe.md", mkdir=True)

    displacement_min_rms: float = 1.0e-3
    moderate_area_change_fraction: float = 0.15
    large_area_change_fraction: float = 0.04
    extreme_area_change_fraction: float = 0.005


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


def expression_names(mesh: pv.UnstructuredGrid) -> list[str]:
    non_expr = {
        "GroupId",
        "IsFace",
        "IsFixed",
        "IsGingiva",
        "IsLip",
        "IsTeeth",
        "vtkOriginalPointIds",
        "FixedMask",
        "FixedValue",
        "TargetFinite",
        "SmileLossMask",
    }
    names: list[str] = []
    for name in mesh.point_data:
        values = np.asarray(mesh.point_data[name])
        if name in non_expr:
            continue
        if values.ndim == 2 and values.shape[1] == 3 and np.issubdtype(values.dtype, np.floating):
            names.append(name)
    return names


def jaw_like_expression(name: str) -> bool:
    lowered = name.lower()
    return any(term in lowered for term in ("jaw", "mouthopen", "scream"))


def score_row(row: dict[str, Any]) -> float:
    rms = float(row["disp_rms"])
    return (
        abs(math.log(float(row["area_total_ratio"])))
        + float(row["frac_abs_log_gt_10pct"]) * 2.0
        + float(row["frac_abs_log_gt_25pct"]) * 5.0
        + float(row["frac_abs_log_gt_2x"]) * 20.0
        + max(0.0, 0.001 - rms) * 50.0
    )


def probe_expression(
    mesh: pv.UnstructuredGrid,
    surface: pv.PolyData,
    faces: np.ndarray,
    original_ids: np.ndarray,
    rest_area: np.ndarray,
    valid_cells: np.ndarray,
    name: str,
) -> dict[str, Any]:
    raw_disp = np.asarray(mesh.point_data[name], dtype=np.float64)
    point_finite = np.isfinite(raw_disp).all(axis=1)
    disp = np.nan_to_num(raw_disp, nan=0.0, posinf=0.0, neginf=0.0)
    surface_disp = disp[original_ids]
    surface_finite = point_finite[original_ids]
    valid = valid_cells & np.all(surface_finite[faces], axis=1)

    target_area = triangle_areas(np.asarray(surface.points) + surface_disp, faces)
    ratio = target_area[valid] / rest_area[valid]
    log_ratio = np.log(np.maximum(ratio, np.finfo(np.float64).tiny))
    area_quantiles = np.quantile(ratio, [0, 0.001, 0.01, 0.05, 0.5, 0.95, 0.99, 0.999, 1])

    is_face = np.asarray(mesh.point_data[IS_FACE], dtype=bool)
    point_mask = is_face & point_finite
    disp_norm = np.linalg.norm(disp[point_mask], axis=1)
    row: dict[str, Any] = {
        "name": name,
        "jaw_filtered": jaw_like_expression(name),
        "n_isface_triangles": int(valid.sum()),
        "disp_rms": float(np.linalg.norm(disp[point_mask]) / math.sqrt(max(1, int(point_mask.sum())))),
        "disp_mean": float(disp_norm.mean()),
        "disp_max": float(disp_norm.max()),
        "area_total_ratio": float(target_area[valid].sum() / rest_area[valid].sum()),
        "area_ratio_min": float(area_quantiles[0]),
        "area_ratio_q001": float(area_quantiles[1]),
        "area_ratio_q01": float(area_quantiles[2]),
        "area_ratio_q05": float(area_quantiles[3]),
        "area_ratio_median": float(area_quantiles[4]),
        "area_ratio_q95": float(area_quantiles[5]),
        "area_ratio_q99": float(area_quantiles[6]),
        "area_ratio_q999": float(area_quantiles[7]),
        "area_ratio_max": float(area_quantiles[8]),
        "frac_abs_log_gt_5pct": float(np.mean(np.abs(log_ratio) > math.log(1.05))),
        "frac_abs_log_gt_10pct": float(np.mean(np.abs(log_ratio) > math.log(1.10))),
        "frac_abs_log_gt_25pct": float(np.mean(np.abs(log_ratio) > math.log(1.25))),
        "frac_abs_log_gt_2x": float(np.mean(np.abs(log_ratio) > math.log(2.0))),
        "frac_stretch_gt_10pct": float(np.mean(ratio > 1.10)),
        "frac_shrink_gt_10pct": float(np.mean(ratio < 1.0 / 1.10)),
    }
    row["score"] = score_row(row)
    return row


def format_float(value: float) -> str:
    return f"{value:.6g}"


def format_percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def shortlist(rows: list[dict[str, Any]], cfg: ProbeConfig) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if not row["jaw_filtered"]
        and row["disp_rms"] >= cfg.displacement_min_rms
        and row["frac_abs_log_gt_10pct"] <= cfg.moderate_area_change_fraction
        and row["frac_abs_log_gt_25pct"] <= cfg.large_area_change_fraction
        and row["frac_abs_log_gt_2x"] <= cfg.extreme_area_change_fraction
    ]
    return sorted(candidates, key=lambda row: row["score"])


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    columns = [
        "name",
        "jaw_filtered",
        "score",
        "disp_rms",
        "disp_max",
        "area_total_ratio",
        "area_ratio_q001",
        "area_ratio_q01",
        "area_ratio_q05",
        "area_ratio_median",
        "area_ratio_q95",
        "area_ratio_q99",
        "area_ratio_q999",
        "frac_abs_log_gt_10pct",
        "frac_abs_log_gt_25pct",
        "frac_abs_log_gt_2x",
        "frac_stretch_gt_10pct",
        "frac_shrink_gt_10pct",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows({column: row[column] for column in columns} for row in rows)


def write_report(
    rows: list[dict[str, Any]],
    recommended: list[dict[str, Any]],
    cfg: ProbeConfig,
    *,
    surface_triangles: int,
    isface_triangles: int,
) -> None:
    nonjaw = [row for row in rows if not row["jaw_filtered"]]
    jaw = [row for row in rows if row["jaw_filtered"]]
    lines = [
        "# Expression Surface Area Probe",
        "",
        f"- Mesh: `{cfg.input_mesh}`",
        f"- Surface triangles: `{surface_triangles}`",
        f"- Surface triangles with all vertices in `IsFace`: `{isface_triangles}`",
        "- Jaw filter terms: `jaw`, `mouthopen`, `scream`.",
        "- Score favors robustly small area distortion and nontrivial displacement.",
        "",
        "## Recommended Non-Jaw Shortlist",
        "",
        "| expression | disp RMS | total area | q1 | q99 | >10% area cells | >25% area cells | >2x cells |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        (
            f"| `{row['name']}` | {format_float(row['disp_rms'])} | "
            f"{format_float(row['area_total_ratio'])} | "
            f"{format_float(row['area_ratio_q01'])} | "
            f"{format_float(row['area_ratio_q99'])} | "
            f"{format_percent(row['frac_abs_log_gt_10pct'])} | "
            f"{format_percent(row['frac_abs_log_gt_25pct'])} | "
            f"{format_percent(row['frac_abs_log_gt_2x'])} |"
        )
        for row in recommended
    )
    lines.extend(
        [
            "",
            "## All Non-Jaw Expressions",
            "",
            "| expression | score | disp RMS | total area | q0.1 | q99.9 | >10% | >25% | >2x |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    lines.extend(
        (
            f"| `{row['name']}` | {format_float(row['score'])} | "
            f"{format_float(row['disp_rms'])} | "
            f"{format_float(row['area_total_ratio'])} | "
            f"{format_float(row['area_ratio_q001'])} | "
            f"{format_float(row['area_ratio_q999'])} | "
            f"{format_percent(row['frac_abs_log_gt_10pct'])} | "
            f"{format_percent(row['frac_abs_log_gt_25pct'])} | "
            f"{format_percent(row['frac_abs_log_gt_2x'])} |"
        )
        for row in nonjaw
    )
    lines.extend(
        [
            "",
            "## Filtered Jaw-Like Expressions",
            "",
            "| expression | disp RMS | total area | q1 | q99 | >10% | >25% |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    lines.extend(
        (
            f"| `{row['name']}` | {format_float(row['disp_rms'])} | "
            f"{format_float(row['area_total_ratio'])} | "
            f"{format_float(row['area_ratio_q01'])} | "
            f"{format_float(row['area_ratio_q99'])} | "
            f"{format_percent(row['frac_abs_log_gt_10pct'])} | "
            f"{format_percent(row['frac_abs_log_gt_25pct'])} |"
        )
        for row in jaw
    )
    cfg.output_report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_probe(cfg: ProbeConfig) -> dict[str, Any]:
    mesh = pv.read(cfg.input_mesh)
    surface = mesh.extract_surface(algorithm=None).triangulate()
    original_ids = np.asarray(surface.point_data["vtkOriginalPointIds"], dtype=np.int64)
    faces = triangle_faces(surface)
    rest_area = triangle_areas(np.asarray(surface.points, dtype=np.float64), faces)
    face_point_mask = np.asarray(mesh.point_data[IS_FACE], dtype=bool)[original_ids]
    valid_cells = np.all(face_point_mask[faces], axis=1) & (rest_area > 0.0)
    rows = [
        probe_expression(mesh, surface, faces, original_ids, rest_area, valid_cells, name)
        for name in expression_names(mesh)
    ]
    rows = sorted(rows, key=lambda row: (row["jaw_filtered"], row["score"]))
    recommended = shortlist(rows, cfg)
    summary = {
        "mesh": str(cfg.input_mesh),
        "surface_points": int(surface.n_points),
        "surface_triangles": int(surface.n_cells),
        "isface_surface_triangles": int(valid_cells.sum()),
        "recommended": [row["name"] for row in recommended],
        "jaw_filter_terms": ["jaw", "mouthopen", "scream"],
        "rows": rows,
    }
    cfg.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(rows, cfg.output_csv)
    write_report(
        rows,
        recommended,
        cfg,
        surface_triangles=surface.n_cells,
        isface_triangles=int(valid_cells.sum()),
    )
    cherries.log_output(cfg.output_json)
    cherries.log_output(cfg.output_csv)
    cherries.log_output(cfg.output_report)
    return summary


if __name__ == "__main__":
    cherries.main(run_probe)
