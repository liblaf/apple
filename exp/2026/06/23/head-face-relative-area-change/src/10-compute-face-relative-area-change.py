from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydantic_settings as ps
import pyvista as pv

from liblaf import cherries

logger = logging.getLogger(__name__)

SOURCE_MESH = Path(
    "/home/liblaf/github/liblaf/melon/exp/2026/05/27/head/data/62-tetmesh-3191k.vtu"
)
IS_FACE = "IsFace"


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    input_mesh: Path = cherries.input(SOURCE_MESH)
    output_surface: Path = cherries.output(
        "10-face-relative-area-change.vtp", mkdir=True
    )
    output_npz: Path = cherries.output("10-face-relative-area-change.npz", mkdir=True)
    output_csv: Path = cherries.output("10-face-relative-area-change.csv", mkdir=True)
    output_summary: Path = cherries.output(
        "10-face-relative-area-change-summary.json", mkdir=True
    )
    output_expression_table: Path = cherries.output(
        "10-expression-area-change-summary.csv", mkdir=True
    )
    mask_name: str = IS_FACE
    require_all_mask_vertices: bool = True


def expression_names(mesh: pv.UnstructuredGrid) -> list[str]:
    ignored = {
        "Normals",
        "vtkOriginalPointIds",
        "vtkOriginalCellIds",
    }
    names: list[str] = []
    for name, values in mesh.point_data.items():
        array = np.asarray(values)
        if name in ignored:
            continue
        if array.shape == (mesh.n_points, 3) and np.issubdtype(
            array.dtype, np.number
        ):
            names.append(name)
    return names


def triangle_areas(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    p0 = points[triangles[:, 0]]
    p1 = points[triangles[:, 1]]
    p2 = points[triangles[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)


def selected_surface_triangles(
    mesh: pv.UnstructuredGrid, mask_name: str, *, require_all_mask_vertices: bool
) -> tuple[pv.PolyData, np.ndarray, np.ndarray]:
    surface = mesh.extract_surface(algorithm=None).triangulate()
    if "vtkOriginalPointIds" not in surface.point_data:
        msg = "extract_surface did not produce vtkOriginalPointIds"
        raise KeyError(msg)
    if mask_name not in mesh.point_data:
        msg = f"missing required point-data mask {mask_name!r}"
        raise KeyError(msg)

    original_ids = np.asarray(surface.point_data["vtkOriginalPointIds"], dtype=np.int64)
    faces = np.asarray(surface.faces, dtype=np.int64).reshape(-1, 4)
    if not np.all(faces[:, 0] == 3):
        msg = "expected a triangulated surface"
        raise ValueError(msg)
    surface_triangles = faces[:, 1:]
    mesh_triangles = original_ids[surface_triangles]

    mask = np.asarray(mesh.point_data[mask_name], dtype=bool)
    if require_all_mask_vertices:
        selected = np.all(mask[mesh_triangles], axis=1)
    else:
        selected = np.any(mask[mesh_triangles], axis=1)

    return surface, surface_triangles[selected], mesh_triangles[selected]


def selected_surface(
    surface: pv.PolyData,
    selected_surface_triangles: np.ndarray,
    selected_mesh_triangles: np.ndarray,
) -> pv.PolyData:
    faces = np.empty((selected_surface_triangles.shape[0], 4), dtype=np.int64)
    faces[:, 0] = 3
    faces[:, 1:] = selected_surface_triangles
    result = pv.PolyData(surface.points.copy(), faces.ravel())
    for name, values in surface.point_data.items():
        array = np.asarray(values)
        if array.dtype.kind in {"O", "U", "S"}:
            continue
        result.point_data[name] = array
    result.cell_data["SurfaceTriangleId"] = np.arange(
        selected_surface_triangles.shape[0], dtype=np.int64
    )
    result.cell_data["MeshPointId0"] = selected_mesh_triangles[:, 0]
    result.cell_data["MeshPointId1"] = selected_mesh_triangles[:, 1]
    result.cell_data["MeshPointId2"] = selected_mesh_triangles[:, 2]
    return result


def write_csv(
    path: Path,
    *,
    mesh_triangles: np.ndarray,
    base_area: np.ndarray,
    min_change: np.ndarray,
    max_change: np.ndarray,
    min_expression: np.ndarray,
    max_expression: np.ndarray,
) -> None:
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "triangle_id",
                "point_id_0",
                "point_id_1",
                "point_id_2",
                "base_area",
                "min_relative_area_change",
                "min_expression",
                "max_relative_area_change",
                "max_expression",
            ]
        )
        for triangle_id in range(mesh_triangles.shape[0]):
            writer.writerow(
                [
                    triangle_id,
                    int(mesh_triangles[triangle_id, 0]),
                    int(mesh_triangles[triangle_id, 1]),
                    int(mesh_triangles[triangle_id, 2]),
                    f"{base_area[triangle_id]:.17g}",
                    f"{min_change[triangle_id]:.17g}",
                    min_expression[triangle_id],
                    f"{max_change[triangle_id]:.17g}",
                    max_expression[triangle_id],
                ]
            )


def write_expression_table(
    path: Path,
    *,
    expression_stats: list[dict[str, float | str]],
) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "expression",
                "min_relative_area_change",
                "mean_relative_area_change",
                "max_relative_area_change",
            ],
        )
        writer.writeheader()
        writer.writerows(expression_stats)


@dataclass(frozen=True)
class AreaChangeResult:
    base_area: np.ndarray
    relative_changes: np.ndarray
    min_change: np.ndarray
    max_change: np.ndarray
    min_expression_index: np.ndarray
    max_expression_index: np.ndarray
    expression_stats: list[dict[str, float | str]]


def compute_area_changes(
    mesh: pv.UnstructuredGrid,
    *,
    names: list[str],
    mesh_triangles: np.ndarray,
) -> AreaChangeResult:
    points = np.asarray(mesh.points, dtype=np.float64)
    base_area = triangle_areas(points, mesh_triangles)
    if not np.all(np.isfinite(base_area)):
        msg = "base triangle area contains non-finite values"
        raise ValueError(msg)
    positive_area = base_area > 0.0
    if not np.all(positive_area):
        n_bad = int((~positive_area).sum())
        msg = f"{n_bad} selected triangles have non-positive base area"
        raise ValueError(msg)

    n_triangles = mesh_triangles.shape[0]
    min_change = np.full(n_triangles, np.inf, dtype=np.float64)
    max_change = np.full(n_triangles, -np.inf, dtype=np.float64)
    min_expression_index = np.full(n_triangles, -1, dtype=np.int32)
    max_expression_index = np.full(n_triangles, -1, dtype=np.int32)
    relative_changes = np.empty((len(names), n_triangles), dtype=np.float32)
    expression_stats: list[dict[str, float | str]] = []

    for expression_id, name in enumerate(names):
        displacement = np.asarray(mesh.point_data[name], dtype=np.float64)
        finite = np.isfinite(displacement[mesh_triangles]).all(axis=(1, 2))
        if not np.all(finite):
            msg = (
                f"expression {name!r} has non-finite displacement on "
                f"{int((~finite).sum())} selected triangles"
            )
            raise ValueError(msg)

        area = triangle_areas(points + displacement, mesh_triangles)
        relative = area / base_area - 1.0
        relative_changes[expression_id] = relative.astype(np.float32)

        lower = relative < min_change
        upper = relative > max_change
        min_change[lower] = relative[lower]
        max_change[upper] = relative[upper]
        min_expression_index[lower] = expression_id
        max_expression_index[upper] = expression_id
        expression_stats.append(
            {
                "expression": name,
                "min_relative_area_change": float(relative.min()),
                "mean_relative_area_change": float(relative.mean()),
                "max_relative_area_change": float(relative.max()),
            }
        )
        logger.info(
            "%s: min=%+.6e mean=%+.6e max=%+.6e",
            name,
            float(relative.min()),
            float(relative.mean()),
            float(relative.max()),
        )

    return AreaChangeResult(
        base_area=base_area,
        relative_changes=relative_changes,
        min_change=min_change,
        max_change=max_change,
        min_expression_index=min_expression_index,
        max_expression_index=max_expression_index,
        expression_stats=expression_stats,
    )


def write_outputs(
    cfg: Config,
    *,
    names: list[str],
    surface: pv.PolyData,
    surface_triangles: np.ndarray,
    mesh_triangles: np.ndarray,
    result_data: AreaChangeResult,
) -> None:
    min_expression = np.asarray(
        [names[i] for i in result_data.min_expression_index], dtype=object
    )
    max_expression = np.asarray(
        [names[i] for i in result_data.max_expression_index], dtype=object
    )
    result = selected_surface(surface, surface_triangles, mesh_triangles)
    result.cell_data["BaseArea"] = result_data.base_area
    result.cell_data["MinRelativeAreaChange"] = result_data.min_change
    result.cell_data["MaxRelativeAreaChange"] = result_data.max_change
    result.cell_data["MinExpressionIndex"] = result_data.min_expression_index
    result.cell_data["MaxExpressionIndex"] = result_data.max_expression_index
    result.save(cfg.output_surface)

    np.savez_compressed(
        cfg.output_npz,
        expression_names=np.asarray(names, dtype=np.str_),
        mesh_triangles=mesh_triangles,
        surface_triangles=surface_triangles,
        base_area=result_data.base_area,
        relative_changes=result_data.relative_changes,
        min_relative_area_change=result_data.min_change,
        max_relative_area_change=result_data.max_change,
        min_expression_index=result_data.min_expression_index,
        max_expression_index=result_data.max_expression_index,
    )
    write_csv(
        cfg.output_csv,
        mesh_triangles=mesh_triangles,
        base_area=result_data.base_area,
        min_change=result_data.min_change,
        max_change=result_data.max_change,
        min_expression=min_expression,
        max_expression=max_expression,
    )
    write_expression_table(
        cfg.output_expression_table, expression_stats=result_data.expression_stats
    )


def make_summary(
    cfg: Config,
    *,
    mesh: pv.UnstructuredGrid,
    surface: pv.PolyData,
    names: list[str],
    n_triangles: int,
    result_data: AreaChangeResult,
) -> dict[str, object]:
    min_change = result_data.min_change
    max_change = result_data.max_change
    base_area = result_data.base_area
    min_expression = [names[i] for i in result_data.min_expression_index]
    max_expression = [names[i] for i in result_data.max_expression_index]

    return {
        "input_mesh": str(cfg.input_mesh),
        "mask_name": cfg.mask_name,
        "selection_rule": (
            "all triangle vertices have mask=True"
            if cfg.require_all_mask_vertices
            else "at least one triangle vertex has mask=True"
        ),
        "n_points": int(mesh.n_points),
        "n_tets": int(mesh.n_cells),
        "n_surface_points": int(surface.n_points),
        "n_surface_triangles": int(surface.n_cells),
        "n_selected_face_triangles": int(n_triangles),
        "n_expressions": len(names),
        "expression_names": names,
        "base_area": {
            "min": float(base_area.min()),
            "mean": float(base_area.mean()),
            "max": float(base_area.max()),
            "sum": float(base_area.sum()),
        },
        "min_relative_area_change": {
            "min": float(min_change.min()),
            "mean": float(min_change.mean()),
            "max": float(min_change.max()),
            "argmin_triangle_id": int(min_change.argmin()),
        },
        "max_relative_area_change": {
            "min": float(max_change.min()),
            "mean": float(max_change.mean()),
            "max": float(max_change.max()),
            "argmax_triangle_id": int(max_change.argmax()),
        },
        "global_min": {
            "relative_area_change": float(min_change.min()),
            "triangle_id": int(min_change.argmin()),
            "expression": str(min_expression[min_change.argmin()]),
        },
        "global_max": {
            "relative_area_change": float(max_change.max()),
            "triangle_id": int(max_change.argmax()),
            "expression": str(max_expression[max_change.argmax()]),
        },
        "outputs": {
            "surface": str(cfg.output_surface),
            "npz": str(cfg.output_npz),
            "csv": str(cfg.output_csv),
            "summary": str(cfg.output_summary),
            "expression_table": str(cfg.output_expression_table),
        },
    }


def main(cfg: Config) -> None:
    mesh = pv.read(cfg.input_mesh)
    if not isinstance(mesh, pv.UnstructuredGrid):
        msg = f"expected UnstructuredGrid, got {type(mesh).__name__}"
        raise TypeError(msg)

    names = expression_names(mesh)
    if not names:
        msg = "no numeric point-data arrays with shape (n_points, 3) were found"
        raise ValueError(msg)

    surface, surface_triangles, mesh_triangles = selected_surface_triangles(
        mesh,
        cfg.mask_name,
        require_all_mask_vertices=cfg.require_all_mask_vertices,
    )
    if mesh_triangles.size == 0:
        msg = f"no selected surface triangles for mask {cfg.mask_name!r}"
        raise ValueError(msg)

    result_data = compute_area_changes(mesh, names=names, mesh_triangles=mesh_triangles)
    write_outputs(
        cfg,
        names=names,
        surface=surface,
        surface_triangles=surface_triangles,
        mesh_triangles=mesh_triangles,
        result_data=result_data,
    )

    n_triangles = mesh_triangles.shape[0]
    summary = make_summary(
        cfg,
        mesh=mesh,
        surface=surface,
        names=names,
        n_triangles=n_triangles,
        result_data=result_data,
    )
    cfg.output_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    cherries.log_metrics(
        {
            "mesh/n_points": mesh.n_points,
            "mesh/n_tets": mesh.n_cells,
            "surface/n_triangles": surface.n_cells,
            "face/n_triangles": n_triangles,
            "expressions/count": len(names),
            "relative_area_change/min": float(result_data.min_change.min()),
            "relative_area_change/max": float(result_data.max_change.max()),
        }
    )
    logger.info("Wrote %s", cfg.output_surface)
    logger.info("Wrote %s", cfg.output_npz)
    logger.info("Wrote %s", cfg.output_csv)
    logger.info("Wrote %s", cfg.output_summary)


if __name__ == "__main__":
    cherries.main(main)
