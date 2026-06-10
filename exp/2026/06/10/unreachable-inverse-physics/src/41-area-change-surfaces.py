from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
import pyvista as pv

from liblaf import cherries

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SurfaceCase:
    name: str
    path: Path
    target_mask: str
    target_volume_is_physical: bool


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    output_summary: Path = cherries.output("41-area-change-surfaces-summary.json")
    output_csv: Path = cherries.output("41-area-change-surfaces-cases.csv")
    output_table: Path = cherries.output("41-area-change-surfaces-table.md")

    include_target_sweep: bool = True


def repo_root() -> Path:
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    msg = f"could not find repo root from {path}"
    raise RuntimeError(msg)


def read_grid(path: Path) -> pv.DataSet:
    if not path.exists():
        msg = f"missing input: {path}"
        raise FileNotFoundError(msg)
    cherries.log_input(path)
    return pv.read(path)


def displacement_from(mesh: pv.DataSet, name: str) -> np.ndarray | None:
    if name not in mesh.point_data:
        return None
    return np.asarray(mesh.point_data[name], dtype=np.float64)


def mask_from_mesh(mesh: pv.DataSet, name: str) -> np.ndarray:
    if name in mesh.point_data:
        return np.asarray(mesh.point_data[name], dtype=bool)
    if "TargetSurfaceMask" in mesh.point_data:
        return np.asarray(mesh.point_data["TargetSurfaceMask"], dtype=bool)
    if "IsFace" in mesh.point_data:
        return np.asarray(mesh.point_data["IsFace"], dtype=bool)
    return np.ones(mesh.n_points, dtype=bool)


def triangle_areas(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    p0 = points[triangles[:, 0]]
    p1 = points[triangles[:, 1]]
    p2 = points[triangles[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)


def surface_triangles(surface: pv.PolyData) -> np.ndarray:
    if "vtkOriginalPointIds" not in surface.point_data:
        msg = "surface extraction did not preserve vtkOriginalPointIds"
        raise KeyError(msg)
    faces = np.asarray(surface.faces, dtype=np.int64).reshape(-1, 4)
    if not np.all(faces[:, 0] == 3):
        msg = "surface triangulation produced non-triangle faces"
        raise ValueError(msg)
    original_ids = np.asarray(surface.point_data["vtkOriginalPointIds"], dtype=np.int64)
    return original_ids[faces[:, 1:]]


def quantiles(values: np.ndarray, prefix: str) -> dict[str, float]:
    if values.size == 0:
        return {
            f"{prefix}/min": math.nan,
            f"{prefix}/p05": math.nan,
            f"{prefix}/p50": math.nan,
            f"{prefix}/p95": math.nan,
            f"{prefix}/max": math.nan,
        }
    return {
        f"{prefix}/min": float(np.min(values)),
        f"{prefix}/p05": float(np.quantile(values, 0.05)),
        f"{prefix}/p50": float(np.quantile(values, 0.50)),
        f"{prefix}/p95": float(np.quantile(values, 0.95)),
        f"{prefix}/max": float(np.max(values)),
    }


def rel_change(new: np.ndarray, old: np.ndarray) -> np.ndarray:
    return np.divide(
        new,
        old,
        out=np.full_like(new, np.nan, dtype=np.float64),
        where=old != 0.0,
    ) - 1.0


def add_area_state(
    *,
    surface: pv.PolyData,
    case_name: str,
    state: str,
    prefix: str,
    points: np.ndarray,
    triangles: np.ndarray,
    rest_area: np.ndarray,
    displacement: np.ndarray,
    triangle_mask_all: np.ndarray,
    physical: bool,
) -> dict[str, Any]:
    area = triangle_areas(points + displacement, triangles)
    area_rel = rel_change(area, rest_area)
    surface.cell_data[f"{prefix}Area"] = area
    surface.cell_data[f"{prefix}AreaDelta"] = area - rest_area
    surface.cell_data[f"{prefix}AreaRelChange"] = area_rel

    rest_sum = float(np.sum(rest_area))
    area_sum = float(np.sum(area))
    rest_mask_sum = float(np.sum(rest_area[triangle_mask_all]))
    area_mask_sum = float(np.sum(area[triangle_mask_all]))
    row: dict[str, Any] = {
        "case": case_name,
        "state": state,
        "physical": physical,
        "n_triangles": int(rest_area.size),
        "n_mask_triangles": int(triangle_mask_all.sum()),
        "rest_surface_area": rest_sum,
        "deformed_surface_area": area_sum,
        "surface_area_rel_change": area_sum / rest_sum - 1.0,
        "rest_mask_area": rest_mask_sum,
        "deformed_mask_area": area_mask_sum,
        "mask_area_rel_change": area_mask_sum / rest_mask_sum - 1.0
        if rest_mask_sum > 0.0
        else math.nan,
    }
    row.update(quantiles(area_rel, "triangle_area_rel_change"))
    row.update(
        quantiles(
            area_rel[triangle_mask_all],
            "mask_triangle_area_rel_change",
        )
    )
    return row


def write_surface_case(case: SurfaceCase, output_dir: Path) -> list[dict[str, Any]]:
    mesh = read_grid(case.path)
    points = np.asarray(mesh.points, dtype=np.float64)
    surface = mesh.extract_surface(algorithm=None).triangulate()
    triangles = surface_triangles(surface)
    rest_area = triangle_areas(points, triangles)
    mask = mask_from_mesh(mesh, case.target_mask)
    mask_count = np.sum(mask[triangles], axis=1).astype(np.int8)
    triangle_mask_all = mask_count == 3
    target_raw = displacement_from(mesh, "TargetDisplacement")
    if target_raw is None:
        target_raw = np.zeros((mesh.n_points, 3), dtype=np.float64)
    target_masked = np.zeros_like(target_raw)
    target_masked[mask] = target_raw[mask]
    solution = displacement_from(mesh, "Displacement")
    if solution is None:
        solution = np.zeros((mesh.n_points, 3), dtype=np.float64)

    surface.point_data["TargetValidPoint"] = mask[
        np.asarray(surface.point_data["vtkOriginalPointIds"], dtype=np.int64)
    ].astype(np.int8)
    surface.cell_data["RestArea"] = rest_area
    surface.cell_data["TargetValidPointCount"] = mask_count
    surface.cell_data["TargetValidTriangleAny"] = (mask_count > 0).astype(np.int8)
    surface.cell_data["TargetValidTriangleAll"] = triangle_mask_all.astype(np.int8)

    rows = [
        add_area_state(
            surface=surface,
            case_name=case.name,
            state="target-raw",
            prefix="TargetRaw",
            points=points,
            triangles=triangles,
            rest_area=rest_area,
            displacement=target_raw,
            triangle_mask_all=triangle_mask_all,
            physical=case.target_volume_is_physical,
        ),
        add_area_state(
            surface=surface,
            case_name=case.name,
            state="target-masked-diagnostic",
            prefix="TargetMasked",
            points=points,
            triangles=triangles,
            rest_area=rest_area,
            displacement=target_masked,
            triangle_mask_all=triangle_mask_all,
            physical=False,
        ),
        add_area_state(
            surface=surface,
            case_name=case.name,
            state="solution",
            prefix="Solution",
            points=points,
            triangles=triangles,
            rest_area=rest_area,
            displacement=solution,
            triangle_mask_all=triangle_mask_all,
            physical=True,
        ),
    ]
    surface.field_data["TargetMaskName"] = np.asarray([case.target_mask])
    surface.field_data["TargetVolumeIsPhysical"] = np.asarray(
        [int(case.target_volume_is_physical)]
    )
    output_path = output_dir / f"{case.name}-area-change.vtp"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    surface.save(output_path)
    cherries.log_output(output_path)
    for row in rows:
        row["surface_path"] = str(output_path)
    logger.info("Wrote %s", output_path)
    return rows


def real_cases(root: Path) -> list[SurfaceCase]:
    return [
        SurfaceCase(
            name="3152k-expression001",
            path=root / "exp/2026/05/27/inverse-face/data/20-inverse-face-3152k.vtu",
            target_mask="IsFace",
            target_volume_is_physical=False,
        ),
        SurfaceCase(
            name="515k-nosmas",
            path=root
            / "exp/2026/05/27/forward-face/data/30-inverse-face-515k-nosmas.vtu",
            target_mask="IsFace",
            target_volume_is_physical=True,
        ),
    ]


def toy_cases(data_dir: Path, *, include_target_sweep: bool) -> list[SurfaceCase]:
    patterns = ["20-toy-*.vtu", "40-toy-forward-activation-*.vtu"]
    if include_target_sweep:
        patterns.append("30-toy-*.vtu")
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(sorted(data_dir.glob(pattern)))
    result_paths = [
        path
        for path in paths
        if not path.name.endswith("-input.vtu")
        and not path.name.endswith("-target.vtu")
        and path.is_file()
    ]
    return [
        SurfaceCase(
            name=path.stem,
            path=path,
            target_mask="TargetSurfaceMask",
            target_volume_is_physical=True,
        )
        for path in result_paths
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


def format_float(value: Any) -> str:
    if not isinstance(value, int | float):
        return ""
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    return f"{float(value):.6g}"


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "| case | state | triangles | mask triangles | surface area change | mask area change | mask p05 | mask p50 | mask p95 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        (
            "| "
            + " | ".join(
                [
                    str(row["case"]),
                    str(row["state"]),
                    str(row["n_triangles"]),
                    str(row["n_mask_triangles"]),
                    format_float(row["surface_area_rel_change"]),
                    format_float(row["mask_area_rel_change"]),
                    format_float(row["mask_triangle_area_rel_change/p05"]),
                    format_float(row["mask_triangle_area_rel_change/p50"]),
                    format_float(row["mask_triangle_area_rel_change/p95"]),
                ]
            )
            + " |"
        )
        for row in rows
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def numeric_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for row in rows:
        prefix = f"{row['case']}/{row['state']}"
        for key in (
            "surface_area_rel_change",
            "mask_area_rel_change",
            "mask_triangle_area_rel_change/p50",
            "mask_triangle_area_rel_change/p95",
        ):
            value = row.get(key)
            if isinstance(value, int | float):
                metrics[f"{prefix}/{key}"] = float(value)
    return metrics


def main(cfg: Config) -> None:
    root = repo_root()
    data_dir = cfg.output_summary.parent
    output_dir = data_dir / "41-area-change-surfaces"
    cases = real_cases(root) + toy_cases(
        data_dir, include_target_sweep=cfg.include_target_sweep
    )
    rows: list[dict[str, Any]] = []
    for case in cases:
        rows.extend(write_surface_case(case, output_dir))

    cfg.output_summary.write_text(
        json.dumps({"cases": rows}, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_csv(cfg.output_csv, rows)
    write_table(cfg.output_table, rows)
    cherries.log_metrics(numeric_metrics(rows))
    logger.info("Wrote %s", cfg.output_summary)
    logger.info("Wrote %s", cfg.output_csv)
    logger.info("Wrote %s", cfg.output_table)


if __name__ == "__main__":
    cherries.main(main)
