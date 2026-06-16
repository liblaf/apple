from __future__ import annotations

import csv
import importlib.util
import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pydantic_settings as ps
import pyvista as pv

from liblaf import cherries, melon

logger = logging.getLogger(__name__)


def load_base_module() -> Any:
    path = Path(__file__).with_name("20-toy-unreachable-inverse.py")
    spec = importlib.util.spec_from_file_location("toy_unreachable_inverse", path)
    if spec is None or spec.loader is None:
        msg = f"could not load base toy experiment from {path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


BASE = load_base_module()

ALL_BOUNDS = (0.0, 1.0, 0.0, 0.1, 0.0, 1.0)
SMAS_BOUNDS = (0.0, 1.0, 0.04, 0.06, 0.0, 1.0)
MUSCLE_BOUNDS = (0.0, 0.5, 0.04, 0.06, 0.4, 0.6)


@dataclass(frozen=True)
class TetwildSpec:
    name: str
    lr: float
    x_segments: int = 0
    y_levels: tuple[float, ...] = ()
    z_segments: int = 0


@dataclass(frozen=True)
class TetwildCase:
    resolution: TetwildSpec
    mode: Literal["stretch", "squash"]
    target_y: float

    @property
    def stem(self) -> str:
        return f"50-toy-tetwild-{self.mode}-{self.resolution.name}"


@dataclass(frozen=True)
class ForwardCase:
    resolution: TetwildSpec

    @property
    def stem(self) -> str:
        return f"50-toy-tetwild-forward-{self.resolution.name}"


class Config(BASE.Config):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    output_summary: Path = cherries.output("50-toy-tetwild-experiments-summary.json")
    output_csv: Path = cherries.output("50-toy-tetwild-experiments-cases.csv")
    output_table: Path = cherries.output("50-toy-tetwild-experiments-table.md")

    lrs: str = "0.05,0.02,0.01"
    forward_lrs: str = "0.05"
    inverse_lrs: str = "0.05"
    modes: str = "stretch,squash"
    target_magnitude: float = 0.02
    fraction_samples_per_tet: int = 16
    fraction_chunk_tets: int = 20_000
    run_forward: bool = True
    run_inverse: bool = True
    inverse_max_steps: int = 80
    series_stride: int = 10


def label_lr(lr: float) -> str:
    return f"lr{lr:g}".replace("0.", "0").replace(".", "p")


def parse_floats(values: str) -> tuple[float, ...]:
    return tuple(float(value.strip()) for value in values.split(",") if value.strip())


def parse_modes(values: str) -> tuple[Literal["stretch", "squash"], ...]:
    modes: list[Literal["stretch", "squash"]] = []
    for value in values.split(","):
        mode = value.strip()
        if mode not in {"stretch", "squash"}:
            msg = f"unknown mode {mode!r}; expected stretch or squash"
            raise ValueError(msg)
        modes.append(mode)  # pyright: ignore[reportArgumentType]
    return tuple(modes)


def specs(cfg: Config) -> list[TetwildSpec]:
    return [TetwildSpec(name=label_lr(lr), lr=lr) for lr in parse_floats(cfg.lrs)]


def tetwild_surface() -> pv.PolyData:
    return pv.Box(ALL_BOUNDS, quads=False).triangulate()


def tetra_signed_volumes(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    p0 = points[tets[:, 0]]
    p1 = points[tets[:, 1]]
    p2 = points[tets[:, 2]]
    p3 = points[tets[:, 3]]
    return np.einsum("ij,ij->i", np.cross(p1 - p0, p2 - p0), p3 - p0) / 6.0


def orient_tetra_mesh(mesh: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    tets = np.asarray(mesh.cells_dict[pv.CellType.TETRA], dtype=np.int64).copy()
    points = np.asarray(mesh.points, dtype=np.float64)
    signed = tetra_signed_volumes(points, tets)
    flipped = signed < 0.0
    if np.any(flipped):
        tets[flipped, 2], tets[flipped, 3] = tets[flipped, 3], tets[flipped, 2].copy()
    cells = np.empty((tets.shape[0], 5), dtype=np.int64)
    cells[:, 0] = 4
    cells[:, 1:] = tets
    cell_types = np.full(tets.shape[0], int(pv.CellType.TETRA), dtype=np.uint8)
    result = pv.UnstructuredGrid(cells.ravel(), cell_types, points)
    return result


def make_tetwild_mesh(spec: TetwildSpec) -> pv.UnstructuredGrid:
    start = time.perf_counter()
    mesh = melon.tetwild(tetwild_surface(), lr=spec.lr)
    if not isinstance(mesh, pv.UnstructuredGrid):
        mesh = mesh.cast_to_unstructured_grid()
    mesh = orient_tetra_mesh(mesh)
    logger.info(
        "TetWild %s lr=%g produced %d points and %d tetrahedra in %.2fs",
        spec.name,
        spec.lr,
        mesh.n_points,
        mesh.n_cells,
        time.perf_counter() - start,
    )
    return mesh


def sample_barycentric(n_samples: int) -> np.ndarray:
    rng = np.random.default_rng(20_260_610)
    values = rng.exponential(scale=1.0, size=(n_samples, 4))
    return values / values.sum(axis=1, keepdims=True)


def inside_box(
    points: np.ndarray, bounds: tuple[float, float, float, float, float, float]
) -> np.ndarray:
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    eps = 1.0e-12
    return (
        (points[..., 0] >= xmin - eps)
        & (points[..., 0] <= xmax + eps)
        & (points[..., 1] >= ymin - eps)
        & (points[..., 1] <= ymax + eps)
        & (points[..., 2] >= zmin - eps)
        & (points[..., 2] <= zmax + eps)
    )


def sampled_box_fraction(
    *,
    points: np.ndarray,
    tets: np.ndarray,
    bounds: tuple[float, float, float, float, float, float],
    barycentric: np.ndarray,
    chunk_tets: int,
) -> np.ndarray:
    fractions = np.empty(tets.shape[0], dtype=np.float64)
    for start in range(0, tets.shape[0], chunk_tets):
        end = min(start + chunk_tets, tets.shape[0])
        tet_points = points[tets[start:end]]
        samples = np.einsum("sf,tfc->tsc", barycentric, tet_points)
        fractions[start:end] = inside_box(samples, bounds).mean(axis=1)
    return fractions


def add_fraction_fields(mesh: pv.UnstructuredGrid, cfg: Config) -> None:
    from liblaf.apple.common import ACTIVATION, ACTIVATION_INV, FIXED_MASK, FIXED_VALUE

    points = np.asarray(mesh.points, dtype=np.float64)
    tets = BASE.tetra_cells(mesh)
    barycentric = sample_barycentric(cfg.fraction_samples_per_tet)
    muscle = sampled_box_fraction(
        points=points,
        tets=tets,
        bounds=MUSCLE_BOUNDS,
        barycentric=barycentric,
        chunk_tets=cfg.fraction_chunk_tets,
    )
    smas = sampled_box_fraction(
        points=points,
        tets=tets,
        bounds=SMAS_BOUNDS,
        barycentric=barycentric,
        chunk_tets=cfg.fraction_chunk_tets,
    )
    muscle = np.minimum(muscle, smas)
    aponeurosis = np.maximum(0.0, smas - muscle)
    fat = np.clip(1.0 - aponeurosis - muscle, 0.0, 1.0)
    active = muscle > cfg.active_fraction_tol
    zero_activation = np.zeros((mesh.n_cells, 6), dtype=np.float64)

    mesh.cell_data[BASE.MUSCLE_FRACTION] = muscle
    mesh.cell_data[BASE.SMAS_FRACTION] = smas
    mesh.cell_data[BASE.APONEUROSIS_FRACTION] = aponeurosis
    mesh.cell_data[BASE.FAT_FRACTION] = fat
    mesh.cell_data[BASE.BACKGROUND_FRACTION] = fat
    mesh.cell_data[BASE.ACTIVE_FRACTION] = muscle
    mesh.cell_data[BASE.SMAS_STIFFNESS_FRACTION] = aponeurosis
    mesh.cell_data["ActivationMask"] = active.astype(np.int8)
    mesh.cell_data["Volume"] = BASE.tetra_volumes(points, tets)
    mesh.cell_data[ACTIVATION.vtk] = zero_activation.copy()
    mesh.cell_data[ACTIVATION_INV.vtk] = zero_activation.copy()
    mesh.field_data["FractionSamplesPerTet"] = np.asarray(
        [cfg.fraction_samples_per_tet]
    )

    eps = max(1.0e-5, 0.03 * min(parse_floats(cfg.lrs)))
    point_x, point_y, point_z = points[:, 0], points[:, 1], points[:, 2]
    bottom = point_y <= ALL_BOUNDS[2] + eps
    top = point_y >= ALL_BOUNDS[3] - eps
    sides = (
        (point_x <= ALL_BOUNDS[0] + eps)
        | (point_x >= ALL_BOUNDS[1] - eps)
        | (point_z <= ALL_BOUNDS[4] + eps)
        | (point_z >= ALL_BOUNDS[5] - eps)
    )
    fixed = bottom | sides
    target = top & ~fixed
    fixed_mask = np.repeat(fixed[:, np.newaxis], 3, axis=1)

    mesh.point_data["FixedBottom"] = bottom.astype(np.int8)
    mesh.point_data["FixedSide"] = sides.astype(np.int8)
    mesh.point_data[BASE.FIXED_BOUNDARY] = fixed.astype(np.int8)
    mesh.point_data[BASE.TOP_SURFACE_MASK] = top.astype(np.int8)
    mesh.point_data[BASE.TARGET_SURFACE_MASK] = target.astype(np.int8)
    mesh.point_data[FIXED_MASK.vtk] = fixed_mask
    mesh.point_data[FIXED_VALUE.vtk] = np.zeros((mesh.n_points, 3), dtype=np.float64)


def write_area_surface(
    *,
    mesh: pv.UnstructuredGrid,
    stem: str,
    target: np.ndarray,
    solution: np.ndarray,
    output_dir: Path,
) -> Path:
    surface = mesh.extract_surface(algorithm=None).triangulate()
    original_ids = np.asarray(surface.point_data["vtkOriginalPointIds"], dtype=np.int64)
    faces = np.asarray(surface.faces, dtype=np.int64).reshape(-1, 4)
    triangles = original_ids[faces[:, 1:]]
    points = np.asarray(mesh.points, dtype=np.float64)
    rest_area = BASE.triangle_areas(points, triangles)
    target_area = BASE.triangle_areas(points + target, triangles)
    solution_area = BASE.triangle_areas(points + solution, triangles)
    target_mask = np.asarray(mesh.point_data[BASE.TARGET_SURFACE_MASK], dtype=bool)
    target_count = np.sum(target_mask[triangles], axis=1).astype(np.int8)

    surface.cell_data["RestArea"] = rest_area
    surface.cell_data["TargetSurfacePointCount"] = target_count
    surface.cell_data["TargetSurfaceTriangleAll"] = (target_count == 3).astype(np.int8)
    surface.cell_data["TargetArea"] = target_area
    surface.cell_data["TargetAreaRelChange"] = BASE.rel_change(target_area, rest_area)
    surface.cell_data["SolutionArea"] = solution_area
    surface.cell_data["SolutionAreaRelChange"] = BASE.rel_change(
        solution_area, rest_area
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{stem}-area-change.vtp"
    surface.save(path)
    cherries.log_output(path)
    return path


def target_rows(
    mesh: pv.UnstructuredGrid, spec: TetwildSpec, cfg: Config
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    data_dir = cfg.output_summary.parent
    input_path = data_dir / f"50-toy-tetwild-{spec.name}-input.vtu"
    melon.save(input_path, mesh)
    cherries.log_output(input_path)
    for mode in parse_modes(cfg.modes):
        target_y = cfg.target_magnitude if mode == "stretch" else -cfg.target_magnitude
        case = TetwildCase(resolution=spec, mode=mode, target_y=target_y)
        target = BASE.target_displacement(mesh, target_y)
        target_mesh = BASE.make_target_mesh(mesh, target)
        target_path = data_dir / f"{case.stem}-target.vtu"
        melon.save(target_path, target_mesh)
        area_path = write_area_surface(
            mesh=mesh,
            stem=case.stem,
            target=target,
            solution=target,
            output_dir=data_dir / "50-toy-tetwild-area-surfaces",
        )
        target_mask = np.asarray(mesh.point_data[BASE.TARGET_SURFACE_MASK], dtype=bool)
        row: dict[str, Any] = {
            "kind": "target",
            "case": case.stem,
            "mode": mode,
            "resolution": spec.name,
            "lr": spec.lr,
            "n_points": int(mesh.n_points),
            "n_tets": int(mesh.n_cells),
            "n_active_tets": int(
                np.asarray(mesh.cell_data["ActivationMask"], dtype=bool).sum()
            ),
            "n_target_points": int(target_mask.sum()),
            "target_y": target_y,
            "input_path": str(input_path),
            "target_path": str(target_path),
            "area_surface_path": str(area_path),
        }
        row.update(
            {
                f"target/{key}": value
                for key, value in BASE.geometry_change(
                    mesh, target, target_mask
                ).items()
            }
        )
        rows.append(row)
        cherries.log_output(target_path)
    return rows


def activation_inv_from_activation(
    activation: tuple[float, float, float, float, float, float],
) -> tuple[float, float, float, float, float, float]:
    xx, yy, zz, xy, xz, yz = activation
    A = np.asarray(
        [[1.0 + xx, xy, xz], [xy, 1.0 + yy, yz], [xz, yz, 1.0 + zz]],
        dtype=np.float64,
    )
    A_inv = np.linalg.inv(A) - np.eye(3, dtype=np.float64)
    return (
        float(A_inv[0, 0]),
        float(A_inv[1, 1]),
        float(A_inv[2, 2]),
        float(A_inv[0, 1]),
        float(A_inv[0, 2]),
        float(A_inv[1, 2]),
    )


def apply_forward_activation(
    mesh: pv.UnstructuredGrid,
    activation: tuple[float, float, float, float, float, float],
) -> np.ndarray:
    from liblaf.apple.common import ACTIVATION, ACTIVATION_INV

    active = np.asarray(mesh.cell_data["ActivationMask"], dtype=bool)
    activation_values = np.zeros((mesh.n_cells, 6), dtype=np.float64)
    activation_inv = np.zeros((mesh.n_cells, 6), dtype=np.float64)
    activation_values[active] = np.asarray(activation, dtype=np.float64)
    activation_inv[active] = np.asarray(activation_inv_from_activation(activation))
    mesh.cell_data[ACTIVATION.vtk] = activation_values
    mesh.cell_data[ACTIVATION_INV.vtk] = activation_inv
    mesh.cell_data["ActivationNorm"] = np.linalg.norm(activation_values, axis=1)
    mesh.cell_data["ActivationInvNorm"] = np.linalg.norm(activation_inv, axis=1)
    return activation_inv


def forward_solution_metrics(solution: Any) -> dict[str, Any]:
    if solution is None:
        return {"forward/result": "missing", "forward/success": False}
    convergence_state = solution.state.convergence_state
    line_search_state = solution.state.line_search_state
    return {
        "forward/result": str(solution.result),
        "forward/success": bool(solution.success),
        "forward/steps": int(convergence_state.step),
        "forward/grad_norm": float(convergence_state.grad_norm),
        "forward/grad_norm_first": float(convergence_state.grad_norm_first),
        "forward/line_search_ok": bool(line_search_state.ok),
        "forward/line_search_steps": int(line_search_state.step),
    }


def forward_rows(
    meshes: dict[float, pv.UnstructuredGrid], cfg: Config
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs(cfg):
        if spec.lr not in parse_floats(cfg.forward_lrs):
            continue
        mesh = meshes[spec.lr].copy(deep=True)
        start = time.perf_counter()
        activation_inv = apply_forward_activation(mesh, (-0.5, 0.0, 0.0, 0.0, 0.0, 0.0))
        forward = BASE.build_forward(mesh, cfg)
        solution = forward.step()
        displacement = BASE.to_numpy(forward.state.u)
        target = np.zeros_like(displacement)
        data_dir = cfg.output_summary.parent
        stem = ForwardCase(spec).stem
        output_path = data_dir / f"{stem}.vtu"
        result = BASE.make_result_mesh(
            mesh,
            target,
            displacement,
            activation_inv,
            {},
        )
        result.cell_data["ForwardActivationInv"] = activation_inv
        melon.save(output_path, result)
        area_path = write_area_surface(
            mesh=mesh,
            stem=stem,
            target=target,
            solution=displacement,
            output_dir=data_dir / "50-toy-tetwild-area-surfaces",
        )
        target_mask = np.asarray(mesh.point_data[BASE.TARGET_SURFACE_MASK], dtype=bool)
        row: dict[str, Any] = {
            "kind": "forward",
            "case": stem,
            "mode": "forward",
            "resolution": spec.name,
            "lr": spec.lr,
            "n_points": int(mesh.n_points),
            "n_tets": int(mesh.n_cells),
            "n_active_tets": int(
                np.asarray(mesh.cell_data["ActivationMask"], dtype=bool).sum()
            ),
            "n_target_points": int(target_mask.sum()),
            "elapsed_s": time.perf_counter() - start,
            "output_path": str(output_path),
            "area_surface_path": str(area_path),
        }
        row.update(forward_solution_metrics(solution))
        row.update(
            {
                f"forward/{key}": value
                for key, value in BASE.geometry_change(
                    mesh, displacement, target_mask
                ).items()
            }
        )
        row.update(
            {
                f"forward/{key}": value
                for key, value in BASE.top_roughness(mesh, displacement).items()
            }
        )
        rows.append(row)
        cherries.log_output(output_path)
        cherries.log_metrics(
            {
                f"{stem}/{key}": value
                for key, value in row.items()
                if isinstance(value, int | float | bool)
            }
        )
    return rows


def inverse_rows(
    meshes: dict[float, pv.UnstructuredGrid], cfg: Config
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    original_make_tet_box = BASE.make_tet_box
    original_add_material = BASE.add_material_and_boundary_fields
    try:
        for spec in specs(cfg):
            if spec.lr not in parse_floats(cfg.inverse_lrs):
                continue
            for mode in parse_modes(cfg.modes):
                target_y = (
                    cfg.target_magnitude if mode == "stretch" else -cfg.target_magnitude
                )
                case = TetwildCase(resolution=spec, mode=mode, target_y=target_y)
                mesh = meshes[spec.lr].copy(deep=True)
                BASE.make_tet_box = lambda _resolution, mesh=mesh: mesh.copy(deep=True)
                BASE.add_material_and_boundary_fields = lambda _mesh, _cfg: None
                row = BASE.solve_case(case, cfg)
                row["kind"] = "inverse"
                row["lr"] = spec.lr
                output_path = cfg.output_summary.parent / f"{case.stem}.vtu"
                result = pv.read(output_path)
                if not isinstance(result, pv.UnstructuredGrid):
                    result = result.cast_to_unstructured_grid()
                area_path = write_area_surface(
                    mesh=result,
                    stem=case.stem,
                    target=np.asarray(
                        result.point_data["TargetDisplacement"], dtype=np.float64
                    ),
                    solution=np.asarray(
                        result.point_data["Displacement"], dtype=np.float64
                    ),
                    output_dir=cfg.output_summary.parent
                    / "50-toy-tetwild-area-surfaces",
                )
                row["area_surface_path"] = str(area_path)
                rows.append(row)
    finally:
        BASE.make_tet_box = original_make_tet_box
        BASE.add_material_and_boundary_fields = original_add_material
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    excluded = {"trace", "y_levels"}
    keys = sorted({key for row in rows for key in row if key not in excluded})
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
        "| kind | case | lr | tets | active tets | signed dV | area dA | error/target | top y std | status |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        if row["kind"] == "target":
            signed_dv = row.get("target/volume/rel_change")
            area_da = row.get("target/target_area/rel_change")
            error = ""
            top_std = ""
            status = "kinematic target"
        elif row["kind"] == "forward":
            signed_dv = row.get("forward/volume/rel_change")
            area_da = row.get("forward/target_area/rel_change")
            error = ""
            top_std = row.get("forward/top_y/std")
            status = row.get("forward/result", "")
        else:
            signed_dv = row.get("inverse/volume/rel_change")
            area_da = row.get("inverse/target_area/rel_change")
            error = row.get("best/error_rms_fraction_of_target")
            top_std = row.get("inverse/top_y/std")
            status = row.get("convergence/status", "")
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("kind", "")),
                    str(row.get("case", "")),
                    format_float(row.get("lr")),
                    str(row.get("n_tets", "")),
                    str(row.get("n_active_tets", "")),
                    format_float(signed_dv),
                    format_float(area_da),
                    format_float(error),
                    format_float(top_std),
                    str(status),
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(cfg: Config) -> None:
    BASE.configure_runtime()
    meshes: dict[float, pv.UnstructuredGrid] = {}
    rows: list[dict[str, Any]] = []
    for spec in specs(cfg):
        mesh = make_tetwild_mesh(spec)
        add_fraction_fields(mesh, cfg)
        meshes[spec.lr] = mesh
        rows.extend(target_rows(mesh, spec, cfg))
    if cfg.run_forward:
        rows.extend(forward_rows(meshes, cfg))
    if cfg.run_inverse:
        rows.extend(inverse_rows(meshes, cfg))

    cfg.output_summary.write_text(
        json.dumps({"cases": rows}, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_csv(cfg.output_csv, rows)
    write_table(cfg.output_table, rows)
    logger.info("Wrote %s", cfg.output_summary)
    logger.info("Wrote %s", cfg.output_csv)
    logger.info("Wrote %s", cfg.output_table)


if __name__ == "__main__":
    cherries.main(main)
