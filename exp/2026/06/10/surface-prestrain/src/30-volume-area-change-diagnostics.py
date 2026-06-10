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

from liblaf import cherries, melon

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[6]
EXP_DIR = REPO_ROOT / "exp/2026/06/10/surface-prestrain"


@dataclass(frozen=True)
class Case:
    name: str
    result: Path


CASES = (
    Case(
        name="snh1-515k-expression000-smas1",
        result=EXP_DIR / "data/20-surface-prestrain-515k-expression000-smas1.vtu",
    ),
    Case(
        name="snh1-3152k-expression000-smas100",
        result=EXP_DIR / "data/20-surface-prestrain-3152k-expression000-smas100.vtu",
    ),
    Case(
        name="snh4-515k-expression000-smas1",
        result=EXP_DIR
        / "data/21-surface-prestrain-expression000-prestrain04-515k-expression000-smas1.vtu",
    ),
    Case(
        name="snh4-3152k-expression000-smas100",
        result=EXP_DIR
        / "data/21-surface-prestrain-expression000-prestrain04-3152k-expression000-smas100.vtu",
    ),
    Case(
        name="metric10-515k-expression000-smas1",
        result=EXP_DIR
        / "data/22-surface-metric-penalty-expression000-prestrain10-515k-expression000-smas1.vtu",
    ),
    Case(
        name="metric10-3152k-expression000-smas100",
        result=EXP_DIR
        / "data/22-surface-metric-penalty-expression000-prestrain10-3152k-expression000-smas100.vtu",
    ),
)

DISPLACEMENT_ARRAYS = {
    "Target": "TargetDisplacement",
    "PreviousInverse": "PreviousInverseDisplacement",
    "Prestrain": "PrestrainDisplacement",
}


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    output_summary: Path = cherries.output(
        "30-volume-area-change-diagnostics-summary.json"
    )
    output_csv: Path = cherries.output("30-volume-area-change-diagnostics-cases.csv")
    output_table: Path = cherries.output("30-volume-area-change-diagnostics-table.md")

    cases: tuple[str, ...] = tuple(case.name for case in CASES)
    target_point_mask: str = "IsFace"


def require_path(path: Path) -> None:
    if path.exists():
        return
    msg = f"missing input: {path}"
    raise FileNotFoundError(msg)


def load_case(case: Case) -> pv.UnstructuredGrid:
    require_path(case.result)
    cherries.log_input(case.result)
    mesh = pv.read(case.result)
    if not isinstance(mesh, pv.UnstructuredGrid):
        mesh = mesh.cast_to_unstructured_grid()
    return mesh


def tetra_connectivity(mesh: pv.UnstructuredGrid) -> np.ndarray:
    cells = np.asarray(mesh.cells, dtype=np.int64)
    if cells.size != mesh.n_cells * 5:
        msg = "expected a tetra-only unstructured grid with 5 entries per cell"
        raise ValueError(msg)
    cells = cells.reshape(-1, 5)
    if not np.all(cells[:, 0] == 4):
        msg = "expected every cell to have exactly four vertices"
        raise ValueError(msg)
    return cells[:, 1:].astype(np.int64, copy=False)


def signed_tetra_volumes(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    p0 = points[tets[:, 0]]
    p1 = points[tets[:, 1]]
    p2 = points[tets[:, 2]]
    p3 = points[tets[:, 3]]
    return np.einsum("ij,ij->i", np.cross(p1 - p0, p2 - p0), p3 - p0) / 6.0


def triangle_areas(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    p0 = points[triangles[:, 0]]
    p1 = points[triangles[:, 1]]
    p2 = points[triangles[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)


def safe_ratio(value: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.divide(
        value,
        reference,
        out=np.full_like(value, np.nan, dtype=np.float64),
        where=reference != 0.0,
    )


def relative_change(value: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return safe_ratio(value, reference) - 1.0


def displacement_arrays(mesh: pv.UnstructuredGrid) -> dict[str, np.ndarray]:
    output: dict[str, np.ndarray] = {}
    for label, name in DISPLACEMENT_ARRAYS.items():
        if name not in mesh.point_data:
            continue
        values = np.asarray(mesh.point_data[name], dtype=np.float64)
        if values.shape != (mesh.n_points, 3):
            msg = f"point_data[{name!r}] has unexpected shape {values.shape}"
            raise ValueError(msg)
        output[label] = values
    if not output:
        msg = "mesh has no recognized displacement arrays"
        raise KeyError(msg)
    return output


def isface_triangles(
    mesh: pv.UnstructuredGrid, mask_name: str
) -> tuple[np.ndarray, pv.PolyData]:
    if mask_name not in mesh.point_data:
        msg = f"mesh has no point_data[{mask_name!r}]"
        raise KeyError(msg)
    point_mask = np.asarray(mesh.point_data[mask_name], dtype=bool)
    surface = mesh.extract_surface(algorithm=None).triangulate()
    if "vtkOriginalPointIds" not in surface.point_data:
        msg = "extract_surface did not produce vtkOriginalPointIds"
        raise KeyError(msg)
    original_ids = np.asarray(surface.point_data["vtkOriginalPointIds"], dtype=np.int64)
    faces = np.asarray(surface.faces, dtype=np.int64).reshape(-1, 4)
    triangles = original_ids[faces[:, 1:]]
    triangle_mask = np.all(point_mask[triangles], axis=1)
    triangles = triangles[triangle_mask].astype(np.int64, copy=False)
    if triangles.size == 0:
        msg = f"no IsFace triangles selected by all-vertices {mask_name}"
        raise ValueError(msg)

    faces = np.empty((triangles.shape[0], 4), dtype=np.int64)
    faces[:, 0] = 3
    faces[:, 1:] = triangles
    isface = pv.PolyData(mesh.points, faces)
    isface.point_data["vtkOriginalPointIds"] = np.arange(mesh.n_points, dtype=np.int64)
    isface.point_data[mask_name] = point_mask.astype(np.int8)
    return triangles, isface


def add_volume_change_arrays(
    output: pv.UnstructuredGrid,
    *,
    points: np.ndarray,
    tets: np.ndarray,
    displacements: dict[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    rest_signed = signed_tetra_volumes(points, tets)
    rest_volume = np.abs(rest_signed)
    output.cell_data["RestVolume"] = rest_volume

    stats: dict[str, dict[str, Any]] = {}
    for label, displacement in displacements.items():
        deformed_signed = signed_tetra_volumes(points + displacement, tets)
        deformed_volume = np.abs(deformed_signed)
        rel = relative_change(deformed_volume, rest_volume)
        signed_ratio = safe_ratio(deformed_signed, rest_signed)
        inverted = np.signbit(deformed_signed) != np.signbit(rest_signed)
        output.cell_data[f"{label}Volume"] = deformed_volume
        output.cell_data[f"{label}VolumeRelChange"] = rel
        output.cell_data[f"{label}SignedVolumeRatio"] = signed_ratio
        output.cell_data[f"{label}InvertedTet"] = inverted.astype(np.int8)
        stats[label] = scalar_stats(rel) | {
            "rest_total": float(np.sum(rest_volume)),
            "deformed_total": float(np.sum(deformed_volume)),
            "total_rel_change": float(
                np.sum(deformed_volume) / np.sum(rest_volume) - 1.0
            ),
            "n_inverted": int(np.count_nonzero(inverted)),
        }
    return stats


def add_area_change_arrays(
    output: pv.PolyData,
    *,
    points: np.ndarray,
    triangles: np.ndarray,
    displacements: dict[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    rest_area = triangle_areas(points, triangles)
    output.cell_data["RestArea"] = rest_area

    stats: dict[str, dict[str, Any]] = {}
    for label, displacement in displacements.items():
        deformed_area = triangle_areas(points + displacement, triangles)
        rel = relative_change(deformed_area, rest_area)
        output.cell_data[f"{label}Area"] = deformed_area
        output.cell_data[f"{label}AreaRelChange"] = rel
        stats[label] = scalar_stats(rel) | {
            "rest_total": float(np.sum(rest_area)),
            "deformed_total": float(np.sum(deformed_area)),
            "total_rel_change": float(np.sum(deformed_area) / np.sum(rest_area) - 1.0),
        }
    return stats


def scalar_stats(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "mean": math.nan,
            "rms": math.nan,
            "min": math.nan,
            "p01": math.nan,
            "p05": math.nan,
            "median": math.nan,
            "p95": math.nan,
            "p99": math.nan,
            "max": math.nan,
        }
    return {
        "mean": float(np.mean(finite)),
        "rms": float(np.sqrt(np.mean(np.square(finite)))),
        "min": float(np.min(finite)),
        "p01": float(np.quantile(finite, 0.01)),
        "p05": float(np.quantile(finite, 0.05)),
        "median": float(np.median(finite)),
        "p95": float(np.quantile(finite, 0.95)),
        "p99": float(np.quantile(finite, 0.99)),
        "max": float(np.max(finite)),
    }


def solve_case(case: Case, cfg: Config) -> list[dict[str, Any]]:
    mesh = load_case(case)
    points = np.asarray(mesh.points, dtype=np.float64)
    tets = tetra_connectivity(mesh)
    displacements = displacement_arrays(mesh)
    triangles, surface = isface_triangles(mesh, cfg.target_point_mask)

    volume_output = mesh.copy(deep=True)
    surface_output = surface.copy(deep=True)
    for label, displacement in displacements.items():
        surface_output.point_data[f"{label}Displacement"] = displacement

    volume_stats = add_volume_change_arrays(
        volume_output, points=points, tets=tets, displacements=displacements
    )
    area_stats = add_area_change_arrays(
        surface_output, points=points, triangles=triangles, displacements=displacements
    )

    volume_path = cfg.output_summary.parent / f"30-volume-change-{case.name}.vtu"
    surface_path = cfg.output_summary.parent / f"30-area-change-{case.name}-isface.vtp"
    melon.save(volume_path, volume_output)
    melon.save(surface_path, surface_output)
    cherries.log_output(volume_path)
    cherries.log_output(surface_path)
    logger.info("Wrote %s", volume_path)
    logger.info("Wrote %s", surface_path)

    rows = []
    for label in displacements:
        row = {
            "case": case.name,
            "label": label,
            "mesh/n_points": int(mesh.n_points),
            "mesh/n_tets": int(mesh.n_cells),
            "surface/n_isface_triangles": int(triangles.shape[0]),
            "output/volume": str(volume_path),
            "output/surface": str(surface_path),
        }
        row.update(
            {f"volume/{key}": value for key, value in volume_stats[label].items()}
        )
        row.update({f"area/{key}": value for key, value in area_stats[label].items()})
        rows.append(row)
        cherries.log_metrics(
            {
                f"{case.name}/{label}/volume_rel_rms": row["volume/rms"],
                f"{case.name}/{label}/area_rel_rms": row["area/rms"],
                f"{case.name}/{label}/area_total_rel_change": row[
                    "area/total_rel_change"
                ],
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in keys})


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
        "| case | label | tets | IsFace tris | volume rel RMS | volume total rel | area rel RMS | area total rel | inverted tets |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["case"]),
                    str(row["label"]),
                    format_float(row["mesh/n_tets"]),
                    format_float(row["surface/n_isface_triangles"]),
                    format_float(row["volume/rms"]),
                    format_float(row["volume/total_rel_change"]),
                    format_float(row["area/rms"]),
                    format_float(row["area/total_rel_change"]),
                    format_float(row["volume/n_inverted"]),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(cfg: Config) -> None:
    selected = {case.name: case for case in CASES}
    rows: list[dict[str, Any]] = []
    for name in cfg.cases:
        if name not in selected:
            msg = f"unknown case {name!r}; choose from {sorted(selected)}"
            raise ValueError(msg)
        rows.extend(solve_case(selected[name], cfg))
    cfg.output_summary.write_text(
        json.dumps({"cases": rows}, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_csv(cfg.output_csv, rows)
    write_table(cfg.output_table, rows)
    logger.info("Wrote %s", cfg.output_summary)
    logger.info("Wrote %s", cfg.output_csv)
    logger.info("Wrote %s", cfg.output_table)


if __name__ == "__main__":
    cherries.main(main)
