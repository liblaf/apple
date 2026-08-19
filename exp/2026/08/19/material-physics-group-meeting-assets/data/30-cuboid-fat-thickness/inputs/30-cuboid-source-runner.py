from __future__ import annotations

import csv
import importlib.util
import json
import logging
import math
import os
import shlex
import sys
import time
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pyvista as pv
import torch

from liblaf import cherries

BOTTOM_FAT_THICKNESS = 0.04
SMAS_THICKNESS = 0.02
TOP_FAT_THICKNESSES = (0.04, 0.08, 0.12)
REPORT_PRESSURES = (0.30, 0.45, 0.60)
EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]

logger = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    top_fat_thicknesses: tuple[float, ...] = TOP_FAT_THICKNESSES
    report_pressures: tuple[float, ...] = REPORT_PRESSURES
    continuation_step: float = 0.075

    E_fat: float = 1.0
    nu: float = 0.49
    smas_stiffness_ratio: float = 1.0e2
    tetwild_lr: float = 0.02
    coarsen: bool = False
    smas_prestrain: tuple[float, float, float, float, float, float] = (
        0.8,
        1.0,
        0.8,
        0.0,
        0.0,
        0.0,
    )
    optimizer_max_steps: int = 3000
    rtol: float = 5.0e-4
    boundary_atol: float = 1.0e-2
    bottom_atol: float = 1.0e-6

    grid_size: int = 101
    grid_margin: float = 0.02
    display_min_det_f: float = 0.20
    display_min_det_f_q001: float = 0.40
    abort_min_det_f: float = 0.10
    overwrite: bool = False

    output_summary_json: Path = cherries.output("30-large-deformation-summary.json")
    output_summary_csv: Path = cherries.output("30-large-deformation-summary.csv")
    output_isometric: Path = EXPERIMENT_ROOT / "figs/30-large-deformation-isometric.png"
    output_section: Path = EXPERIMENT_ROOT / "figs/30-large-deformation-section.png"
    output_top_heatmap: Path = EXPERIMENT_ROOT / "figs/30-large-deformation-top-uy.png"
    output_report: Path = EXPERIMENT_ROOT / "docs/30-large-deformation-report.md"


@dataclass
class CaseResult:
    mesh: pv.UnstructuredGrid
    metrics: dict[str, Any]
    grid_x: np.ndarray
    grid_z: np.ndarray
    grid_u_y: np.ndarray


def load_sweep_helpers() -> ModuleType:
    path = EXPERIMENT_ROOT / "src/20-run-fat-thickness-sweep.py"
    spec = importlib.util.spec_from_file_location("fat_thickness_helpers", path)
    if spec is None or spec.loader is None:
        msg = f"could not load thickness-sweep helpers from {path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def decimal_label(value: float) -> str:
    text = format(Decimal(str(value)).normalize(), "f")
    return text.replace(".", "p").replace("-", "m")


def thickness_label(value: float) -> str:
    return f"top-fat-{decimal_label(value)}"


def pressure_label(value: float) -> str:
    return f"pressure-{decimal_label(value)}"


def case_label(*, thickness: float, pressure: float) -> str:
    return f"{thickness_label(thickness)}-{pressure_label(pressure)}"


def continuation_pressures(
    report_pressures: tuple[float, ...], step: float
) -> tuple[float, ...]:
    start = Decimal(str(min(report_pressures)))
    stop = Decimal(str(max(report_pressures)))
    delta = Decimal(str(step))
    values = {Decimal(str(value)) for value in report_pressures}
    value = start
    while value <= stop:
        values.add(value)
        value += delta
    return tuple(float(value) for value in sorted(values) if start <= value <= stop)


def validate_config(cfg: Config) -> tuple[tuple[float, ...], tuple[float, ...]]:
    thicknesses = tuple(float(value) for value in cfg.top_fat_thicknesses)
    reports = tuple(float(value) for value in cfg.report_pressures)
    if not thicknesses or any(
        not math.isfinite(value) or value <= 0.0 for value in thicknesses
    ):
        msg = "top-fat thicknesses must be finite and positive"
        raise ValueError(msg)
    if len({decimal_label(value) for value in thicknesses}) != len(thicknesses):
        msg = "top-fat thicknesses must produce unique labels"
        raise ValueError(msg)
    if not reports or any(
        not math.isfinite(value) or value <= 0.0 for value in reports
    ):
        msg = "report pressures must be finite and positive"
        raise ValueError(msg)
    if len({decimal_label(value) for value in reports}) != len(reports):
        msg = "report pressures must produce unique labels"
        raise ValueError(msg)
    if not math.isfinite(cfg.continuation_step) or cfg.continuation_step <= 0.0:
        msg = "continuation step must be finite and positive"
        raise ValueError(msg)
    if cfg.grid_size < 5:
        msg = "grid size must be at least 5"
        raise ValueError(msg)
    if not 0.0 <= cfg.grid_margin < 0.5:
        msg = "grid margin must be in [0, 0.5)"
        raise ValueError(msg)
    if not (
        0.0 < cfg.abort_min_det_f <= cfg.display_min_det_f <= cfg.display_min_det_f_q001
    ):
        msg = (
            "expected 0 < abort-min-det-f <= display-min-det-f <= "
            "display-min-det-f-q001"
        )
        raise ValueError(msg)
    return thicknesses, reports


def planned_output_paths(
    cfg: Config,
    *,
    thicknesses: tuple[float, ...],
    pressures: tuple[float, ...],
) -> list[Path]:
    data_dir = cfg.output_summary_json.parent
    paths = [
        cfg.output_summary_json,
        cfg.output_summary_csv,
        cfg.output_isometric,
        cfg.output_section,
        cfg.output_top_heatmap,
        cfg.output_report,
    ]
    for thickness in thicknesses:
        paths.append(data_dir / f"30-{thickness_label(thickness)}-input.vtu")
        for pressure in pressures:
            label = case_label(thickness=thickness, pressure=pressure)
            paths.extend(
                (
                    data_dir / f"30-{label}.vtu",
                    data_dir / f"30-{label}-top-grid.npz",
                )
            )
    return paths


def baseline_config(
    baseline: ModuleType,
    cfg: Config,
    *,
    pressure: float,
    output_input: Path,
    output: Path,
) -> Any:
    return baseline.Config(
        _cli_parse_args=False,
        output_input=output_input,
        output=output,
        E_fat=cfg.E_fat,
        nu=cfg.nu,
        smas_stiffness_ratio=cfg.smas_stiffness_ratio,
        lr=cfg.tetwild_lr,
        coarsen=cfg.coarsen,
        bottom_pressure=pressure,
        smas_prestrain=cfg.smas_prestrain,
        optimizer_max_steps=cfg.optimizer_max_steps,
        rtol_primary=cfg.rtol,
        rtol_secondary=cfg.rtol,
        boundary_atol=cfg.boundary_atol,
        bottom_atol=cfg.bottom_atol,
    )


def material_grid(
    helpers: ModuleType,
    mesh: pv.UnstructuredGrid,
    *,
    top_y: float,
    size: int,
    margin: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | int]]:
    from scipy.interpolate import griddata

    surface, ids, _ = helpers.top_surface(mesh, top_y=top_y)
    material_xz = np.asarray(surface.points, dtype=np.float64)[ids][:, [0, 2]]
    u_y = np.asarray(surface.point_data["Displacement"], dtype=np.float64)[ids, 1]
    x = np.linspace(margin, 1.0 - margin, size)
    z = np.linspace(margin, 1.0 - margin, size)
    xx, zz = np.meshgrid(x, z, indexing="xy")
    grid = griddata(material_xz, u_y, (xx, zz), method="linear")
    missing = ~np.isfinite(grid)
    if np.any(missing):
        nearest = griddata(material_xz, u_y, (xx, zz), method="nearest")
        grid[missing] = nearest[missing]
    if not np.all(np.isfinite(grid)):
        msg = "material-coordinate top-surface interpolation produced NaNs"
        raise RuntimeError(msg)

    dx = float(x[1] - x[0])
    dz = float(z[1] - z[0])
    laplacian = (grid[1:-1, 2:] - 2.0 * grid[1:-1, 1:-1] + grid[1:-1, :-2]) / dx**2 + (
        grid[2:, 1:-1] - 2.0 * grid[1:-1, 1:-1] + grid[:-2, 1:-1]
    ) / dz**2
    p05, p95 = np.quantile(grid, (0.05, 0.95))
    robust_range = float(p95 - p05)
    laplacian_rms = float(np.linalg.norm(laplacian) / math.sqrt(laplacian.size))
    length_scale = 1.0 - 2.0 * margin
    normalized_laplacian = (
        laplacian_rms * length_scale**2 / robust_range
        if robust_range > np.finfo(np.float64).eps
        else 0.0
    )
    metrics: dict[str, float | int] = {
        "top_grid/size": int(size),
        "top_grid/source_points": int(ids.size),
        "top_grid/u_y_mean": float(grid.mean()),
        "top_grid/u_y_std": float(grid.std()),
        "top_grid/u_y_min": float(grid.min()),
        "top_grid/u_y_max": float(grid.max()),
        "top_grid/u_y_range": float(np.ptp(grid)),
        "top_grid/u_y_p05": float(p05),
        "top_grid/u_y_p95": float(p95),
        "top_grid/u_y_p95_minus_p05": robust_range,
        "top_grid/laplacian_rms": laplacian_rms,
        "top_grid/laplacian_rms_normalized": float(normalized_laplacian),
    }
    return x, z, grid, metrics


def top_normal_metrics(
    helpers: ModuleType, mesh: pv.UnstructuredGrid, *, top_y: float
) -> dict[str, float | int]:
    surface, ids, _ = helpers.top_surface(mesh, top_y=top_y)
    faces = np.asarray(surface.faces, dtype=np.int64).reshape(-1, 4)[:, 1:]
    selected = np.all(np.isin(faces, ids), axis=1)
    triangles = faces[selected]
    rest = np.asarray(surface.points, dtype=np.float64)[triangles]
    displacement = np.asarray(surface.point_data["Displacement"], dtype=np.float64)
    deformed = rest + displacement[triangles]

    def normals(points: np.ndarray) -> np.ndarray:
        return np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])

    rest_normals = normals(rest)
    deformed_normals = normals(deformed)
    denominator = np.linalg.norm(rest_normals, axis=1) * np.linalg.norm(
        deformed_normals, axis=1
    )
    cosine = np.divide(
        np.einsum("ij,ij->i", rest_normals, deformed_normals),
        denominator,
        out=np.full(denominator.shape, -1.0, dtype=np.float64),
        where=denominator > 0.0,
    )
    return {
        "top_normal/triangles": int(triangles.shape[0]),
        "top_normal/cosine_min": float(cosine.min()),
        "top_normal/flipped": int(np.count_nonzero(cosine <= 0.0)),
    }


def solver_success(result_name: str) -> bool:
    return result_name == "SUCCESS" or result_name.endswith("_SUCCESS")


def solve_pressure(
    helpers: ModuleType,
    baseline: ModuleType,
    cfg: Config,
    *,
    base_body: pv.UnstructuredGrid,
    thickness: float,
    pressure: float,
    previous_u: np.ndarray | None,
    output_input: Path,
    output_mesh: Path,
    output_grid: Path,
) -> tuple[CaseResult, np.ndarray, bool]:
    from liblaf.apple.common import FORCE, GLOBAL_POINT_ID

    top_y = BOTTOM_FAT_THICKNESS + SMAS_THICKNESS + thickness
    run_cfg = baseline_config(
        baseline,
        cfg,
        pressure=pressure,
        output_input=output_input,
        output=output_mesh,
    )
    body = base_body.copy(deep=True)
    forward = helpers.build_collision_free_model(baseline, body, run_cfg)
    if previous_u is not None:
        if previous_u.shape != tuple(forward.state.u.shape):
            msg = (
                f"warm-start shape changed from {previous_u.shape} to "
                f"{tuple(forward.state.u.shape)}"
            )
            raise RuntimeError(msg)
        warm_start = torch.as_tensor(
            previous_u,
            dtype=forward.state.u.dtype,
            device=forward.state.u.device,
        )
        forward.model.update(forward.state, warm_start)

    initial_energy = baseline.tensor_scalar(forward.problem.fun(forward.state))
    initial_grad = baseline.to_numpy(forward.problem.grad(forward.state))
    initial_grad_norm = float(np.linalg.norm(initial_grad))
    solve_start = time.perf_counter()
    solution = forward.step()
    solve_elapsed = time.perf_counter() - solve_start
    final_energy = baseline.tensor_scalar(forward.problem.fun(forward.state))
    final_grad = baseline.to_numpy(forward.problem.grad(forward.state))
    final_grad_norm = float(np.linalg.norm(final_grad))
    final_u = baseline.to_numpy(forward.state.u).copy()

    global_ids = np.asarray(body.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64)
    displacement = final_u[global_ids]
    result = body.copy(deep=True)
    result.point_data["Displacement"] = displacement
    result.point_data["DisplacementNorm"] = np.linalg.norm(displacement, axis=1)
    result.point_data["DeformedPoint"] = result.points + displacement
    baseline.add_reference_fields(result, run_cfg)

    result_name = solution.result.name
    finite = bool(
        np.all(np.isfinite(displacement))
        and np.isfinite(initial_energy)
        and np.isfinite(final_energy)
        and np.all(np.isfinite(initial_grad))
        and np.all(np.isfinite(final_grad))
    )
    total_force = float(np.asarray(body.point_data[FORCE.vtk])[:, 1].sum())
    metrics: dict[str, Any] = {
        "case": case_label(thickness=thickness, pressure=pressure),
        "top_fat_thickness": float(thickness),
        "bottom_fat_thickness": BOTTOM_FAT_THICKNESS,
        "smas_thickness": SMAS_THICKNESS,
        "body_height": float(top_y),
        "pressure": float(pressure),
        "total_bottom_force": total_force,
        "warm_started": int(previous_u is not None),
        "collision_enabled": 0,
        "solver/result": result_name,
        "solver/success": int(solver_success(result_name)),
        "solver/steps": int(solution.state.step),
        "solver/elapsed_s": float(solve_elapsed),
        "solver/initial_energy": float(initial_energy),
        "solver/final_energy": float(final_energy),
        "solver/free_dofs": int(forward.model.n_free),
        "solver/fixed_dofs": int(forward.model.n_fixed),
        "solver/initial_free_grad_l2": initial_grad_norm,
        "solver/final_free_grad_l2": final_grad_norm,
        "solver/final_free_grad_rms": float(
            final_grad_norm / math.sqrt(max(1, forward.model.n_free))
        ),
        "solver/final_free_grad_linf": float(np.abs(final_grad).max()),
        "solver/final_over_initial_free_grad_l2": float(
            final_grad_norm / max(initial_grad_norm, np.finfo(np.float64).eps)
        ),
        "finite": int(finite),
        "max_displacement": float(np.linalg.norm(displacement, axis=1).max()),
        **helpers.deformation_metrics(result),
        **top_normal_metrics(helpers, result, top_y=top_y),
    }
    grid_x, grid_z, grid_u_y, grid_metrics = material_grid(
        helpers,
        result,
        top_y=top_y,
        size=cfg.grid_size,
        margin=cfg.grid_margin,
    )
    metrics.update(grid_metrics)
    metrics["display_valid"] = int(
        bool(metrics["solver/success"])
        and finite
        and int(metrics["deformation/detF_inverted"]) == 0
        and float(metrics["deformation/detF_min"]) >= cfg.display_min_det_f
        and float(metrics["deformation/detF_q001"]) >= cfg.display_min_det_f_q001
        and int(metrics["top_normal/flipped"]) == 0
    )
    continuation_valid = (
        bool(metrics["solver/success"])
        and finite
        and int(metrics["deformation/detF_inverted"]) == 0
        and float(metrics["deformation/detF_min"]) >= cfg.abort_min_det_f
        and int(metrics["top_normal/flipped"]) == 0
    )
    metrics["continuation_valid"] = int(continuation_valid)

    helpers.add_summary_fields(result, metrics)
    baseline.melon.save(result, output_mesh)
    np.savez_compressed(output_grid, x=grid_x, z=grid_z, u_y=grid_u_y)
    cherries.log_output(output_mesh)
    cherries.log_output(output_grid)
    return (
        CaseResult(
            mesh=result,
            metrics=metrics,
            grid_x=grid_x,
            grid_z=grid_z,
            grid_u_y=grid_u_y,
        ),
        final_u,
        continuation_valid,
    )


def make_body(
    baseline: ModuleType,
    cfg: Config,
    *,
    thickness: float,
    pressure: float,
    output_input: Path,
) -> pv.UnstructuredGrid:
    body_height = BOTTOM_FAT_THICKNESS + SMAS_THICKNESS + thickness
    baseline.BODY_BOUNDS = (  # type: ignore[attr-defined]
        0.0,
        1.0,
        0.0,
        body_height,
        0.0,
        1.0,
    )
    baseline.SMAS_BOUNDS = (  # type: ignore[attr-defined]
        0.0,
        1.0,
        BOTTOM_FAT_THICKNESS,
        BOTTOM_FAT_THICKNESS + SMAS_THICKNESS,
        0.0,
        1.0,
    )
    run_cfg = baseline_config(
        baseline,
        cfg,
        pressure=pressure,
        output_input=output_input,
        output=output_input.with_name(f"{output_input.stem}-unused-result.vtu"),
    )
    mesh_start = time.perf_counter()
    body = baseline.make_body_mesh(run_cfg)
    logger.info(
        "Meshed top fat %.3f in %.2f s: %d points, %d cells",
        thickness,
        time.perf_counter() - mesh_start,
        body.n_points,
        body.n_cells,
    )
    input_mesh = baseline.make_input_mesh(body.copy(deep=True), run_cfg)
    baseline.melon.save(input_mesh, output_input)
    cherries.log_output(output_input)
    return body


def find_case(
    cases: list[CaseResult], *, thickness: float, pressure: float
) -> CaseResult:
    for case in cases:
        row = case.metrics
        if math.isclose(float(row["top_fat_thickness"]), thickness) and math.isclose(
            float(row["pressure"]), pressure
        ):
            return case
    msg = f"missing thickness={thickness}, pressure={pressure}"
    raise KeyError(msg)


def select_render_pressure(
    cases: list[CaseResult],
    *,
    thicknesses: tuple[float, ...],
    report_pressures: tuple[float, ...],
) -> float:
    for pressure in sorted(report_pressures, reverse=True):
        try:
            selected = [
                find_case(cases, thickness=thickness, pressure=pressure)
                for thickness in thicknesses
            ]
        except KeyError:
            continue
        if all(bool(case.metrics["display_valid"]) for case in selected):
            return pressure
    msg = "no report pressure passed the display gate for every thickness"
    raise RuntimeError(msg)


def undeformed_surface(mesh: pv.UnstructuredGrid) -> pv.PolyData:
    return mesh.extract_surface(algorithm=None).triangulate()


def ghost_outline(mesh: pv.DataSet) -> pv.PolyData:
    """Return only visible rest-shape edges, avoiding a dense ghost wireframe."""
    return mesh.extract_feature_edges(
        boundary_edges=True,
        feature_edges=True,
        manifold_edges=False,
        non_manifold_edges=True,
        feature_angle=30.0,
    )


def render_isometric(
    helpers: ModuleType,
    cases: list[CaseResult],
    *,
    pressure: float,
    output: Path,
) -> None:
    deformed = [helpers.deformed_surface(case.mesh) for case in cases]
    all_u_y = np.concatenate(
        [
            np.asarray(surface.point_data["u_y"], dtype=np.float64)
            for surface in deformed
        ]
    )
    clim = (float(all_u_y.min()), float(all_u_y.max()))
    max_height = max(float(case.metrics["body_height"]) for case in cases)
    plotter = pv.Plotter(
        shape=(1, len(cases)),
        off_screen=True,
        window_size=[720 * len(cases), 690],
        border=False,
    )
    plotter.set_background("#0B0D10")  # type: ignore[arg-type]
    camera = [
        (1.65, max_height + 0.95, 1.65),
        (0.5, 0.5 * max_height, 0.5),
        (0.0, 1.0, 0.0),
    ]
    for index, (case, surface) in enumerate(zip(cases, deformed, strict=True)):
        plotter.subplot(0, index)
        plotter.add_mesh(
            surface,
            scalars="u_y",
            cmap="coolwarm",
            clim=clim,
            show_edges=True,
            edge_color="#30343B",
            line_width=0.2,
            show_scalar_bar=index == len(cases) - 1,
            scalar_bar_args={
                "title": "simulated u_y [model length]",
                "vertical": False,
                "position_x": 0.18,
                "position_y": 0.03,
                "width": 0.72,
                "height": 0.08,
                "color": "white",
            },
        )
        plotter.add_mesh(
            ghost_outline(undeformed_surface(case.mesh)),
            color="#F8FAFC",
            opacity=0.92,
            line_width=2.0,
            render_lines_as_tubes=True,
        )
        plotter.add_text(
            f"top fat = {float(case.metrics['top_fat_thickness']):.2f} "
            "[model length]\n"
            f"bottom pressure = {pressure:.2f} [model stress]\n"
            "gray outline = rest shape",
            position=(20, 605),
            color="white",
            font_size=15,
            shadow=True,
        )
        plotter.camera_position = camera
        plotter.enable_parallel_projection()  # type: ignore[call-arg]
        plotter.camera.parallel_scale = 0.78
        plotter.reset_camera_clipping_range()  # type: ignore[call-arg]
    plotter.screenshot(output)
    plotter.close()


def add_layer_id(mesh: pv.UnstructuredGrid) -> None:
    smas = np.asarray(mesh.cell_data["SmasFraction"], dtype=np.float64)
    centers_y = np.asarray(mesh.cell_centers().points, dtype=np.float64)[:, 1]
    layer_id = np.full(mesh.n_cells, 2, dtype=np.int8)
    layer_id[centers_y < BOTTOM_FAT_THICKNESS] = 0
    layer_id[smas > 1.0e-6] = 1
    mesh.cell_data["LayerId"] = layer_id


def render_section(cases: list[CaseResult], *, pressure: float, output: Path) -> None:
    from matplotlib.colors import ListedColormap

    colors = ListedColormap(["#F3C969", "#D95D5D", "#65B7A8"])
    max_height = max(float(case.metrics["body_height"]) for case in cases)
    plotter = pv.Plotter(
        shape=(1, len(cases)),
        off_screen=True,
        window_size=[720 * len(cases), 690],
        border=False,
    )
    plotter.set_background("#0B0D10")  # type: ignore[arg-type]
    for index, case in enumerate(cases):
        plotter.subplot(0, index)
        rest = case.mesh.copy(deep=True)
        add_layer_id(rest)
        displaced = rest.copy(deep=True)
        displaced.points = np.asarray(displaced.points) + np.asarray(
            displaced.point_data["Displacement"], dtype=np.float64
        )
        origin = (0.5, 0.5 * float(case.metrics["body_height"]), 0.5)
        rest_section = rest.slice(normal=(0.0, 0.0, 1.0), origin=origin)
        deformed_section = displaced.slice(normal=(0.0, 0.0, 1.0), origin=origin)
        plotter.add_mesh(
            deformed_section,
            scalars="LayerId",
            cmap=colors,
            clim=(-0.5, 2.5),
            categories=True,
            show_edges=True,
            edge_color="#30343B",
            line_width=0.2,
            show_scalar_bar=False,
        )
        plotter.add_mesh(
            ghost_outline(rest_section),
            color="#F8FAFC",
            opacity=0.95,
            line_width=2.2,
            render_lines_as_tubes=True,
        )
        plotter.add_text(
            f"top fat = {float(case.metrics['top_fat_thickness']):.2f} "
            "[model length]\n"
            f"bottom pressure = {pressure:.2f} [model stress]\n"
            "yellow bottom fat · red SMAS · teal top fat\n"
            "gray outline = rest section",
            position=(20, 565),
            color="white",
            font_size=14,
            shadow=True,
        )
        plotter.camera_position = [
            (0.5, 0.5 * max_height, 2.5),
            (0.5, 0.5 * max_height, 0.5),
            (0.0, 1.0, 0.0),
        ]
        plotter.enable_parallel_projection()  # type: ignore[call-arg]
        plotter.camera.parallel_scale = 0.56
        plotter.reset_camera_clipping_range()  # type: ignore[call-arg]
    plotter.screenshot(output)
    plotter.close()


def render_top_heatmap(
    cases: list[CaseResult], *, pressure: float, output: Path
) -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    from matplotlib import pyplot as plt

    limit_min = min(float(case.grid_u_y.min()) for case in cases)
    limit_max = max(float(case.grid_u_y.max()) for case in cases)
    figure, axes = plt.subplots(
        1, len(cases), figsize=(5.6 * len(cases), 5.0), constrained_layout=True
    )
    axes_array = np.atleast_1d(axes)
    image = None
    for axis, case in zip(axes_array, cases, strict=True):
        extent = (
            float(case.grid_x.min()),
            float(case.grid_x.max()),
            float(case.grid_z.min()),
            float(case.grid_z.max()),
        )
        image = axis.imshow(
            case.grid_u_y,
            origin="lower",
            extent=extent,
            cmap="viridis",
            vmin=limit_min,
            vmax=limit_max,
            interpolation="nearest",
        )
        row = case.metrics
        axis.set_title(
            f"top fat = {float(row['top_fat_thickness']):.2f} [model length]\n"
            f"p95-p05 = {float(row['top_grid/u_y_p95_minus_p05']):.4g} "
            "[model length]"
        )
        axis.set_xlabel("material x [model length]")
        axis.set_ylabel("material z [model length]")
        axis.set_aspect("equal")
    assert image is not None
    colorbar = figure.colorbar(image, ax=list(axes_array), shrink=0.82)
    colorbar.set_label("top-surface u_y [model length]")
    figure.suptitle(
        "Common material-coordinate grid, "
        f"bottom pressure = {pressure:.2f} [model stress]"
    )
    figure.savefig(output, dpi=180)
    plt.close(figure)


def write_outputs(
    cfg: Config,
    cases: list[CaseResult],
    *,
    thicknesses: tuple[float, ...],
    report_pressures: tuple[float, ...],
    continuation_levels: tuple[float, ...],
    render_pressure: float,
) -> None:
    rows = [case.metrics for case in cases]
    summary = {
        "description": (
            "Large-deformation continuation sweep for the controlled fat/SMAS "
            "sandwich. Each thickness uses one mesh and warm-starts increasing "
            "pressure levels."
        ),
        "collision_enabled": False,
        "units": {
            "status": "model units; no SI calibration is asserted",
            "length_and_displacement": "model length",
            "pressure_and_modulus": "model stress",
            "free_gradient": "model force",
            "laplacian_rms": "model displacement / model length^2",
        },
        "controlled_config": {
            "bottom_fat_thickness": BOTTOM_FAT_THICKNESS,
            "smas_thickness": SMAS_THICKNESS,
            "E_fat": cfg.E_fat,
            "E_smas": cfg.E_fat * cfg.smas_stiffness_ratio,
            "nu": cfg.nu,
            "smas_prestrain": cfg.smas_prestrain,
            "tetwild_lr": cfg.tetwild_lr,
            "coarsen": cfg.coarsen,
            "optimizer_max_steps": cfg.optimizer_max_steps,
            "rtol": cfg.rtol,
            "boundary_atol": cfg.boundary_atol,
            "bottom_atol": cfg.bottom_atol,
            "grid_size": cfg.grid_size,
            "grid_margin": cfg.grid_margin,
            "boundary_conditions": (
                "all displacement components fixed on the four vertical sides; "
                "positive-y pressure applied to the free bottom-interior surface"
            ),
        },
        "debug_local_only": os.environ.get("DEBUG") == "1",
        "report_pressures": report_pressures,
        "continuation_pressures": continuation_levels,
        "render_pressure": render_pressure,
        "display_gate": {
            "solver_success": True,
            "finite": True,
            "inverted_tets": 0,
            "min_det_f": cfg.display_min_det_f,
            "det_f_q001": cfg.display_min_det_f_q001,
            "flipped_top_triangles": 0,
        },
        "continuation_gate": {
            "solver_success": True,
            "finite": True,
            "inverted_tets": 0,
            "min_det_f": cfg.abort_min_det_f,
            "flipped_top_triangles": 0,
        },
        "cases": rows,
    }
    cfg.output_summary_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    columns = [
        "case",
        "top_fat_thickness",
        "pressure",
        "warm_started",
        "solver/result",
        "solver/success",
        "solver/steps",
        "solver/free_dofs",
        "solver/initial_free_grad_l2",
        "solver/final_free_grad_l2",
        "solver/final_free_grad_rms",
        "solver/final_free_grad_linf",
        "solver/final_over_initial_free_grad_l2",
        "max_displacement",
        "top_grid/u_y_mean",
        "top_grid/u_y_p95_minus_p05",
        "top_grid/laplacian_rms",
        "top_grid/laplacian_rms_normalized",
        "deformation/detF_min",
        "deformation/detF_q001",
        "deformation/detF_inverted",
        "top_normal/flipped",
        "display_valid",
        "continuation_valid",
    ]
    with cfg.output_summary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    selected = [
        find_case(cases, thickness=thickness, pressure=render_pressure)
        for thickness in thicknesses
    ]
    table_rows = [
        (
            f"| {float(case.metrics['top_fat_thickness']):.2f} | "
            f"{case.metrics['solver/result']} | "
            f"{int(case.metrics['solver/steps'])} | "
            f"{float(case.metrics['solver/final_free_grad_rms']):.4g} | "
            f"{float(case.metrics['max_displacement']):.5f} | "
            f"{float(case.metrics['top_grid/u_y_p95_minus_p05']):.5f} | "
            f"{float(case.metrics['top_grid/laplacian_rms']):.5g} | "
            f"{float(case.metrics['deformation/detF_min']):.3f} | "
            f"{float(case.metrics['deformation/detF_q001']):.3f} |"
        )
        for case in selected
    ]
    thin = selected[0].metrics
    thick = selected[-1].metrics
    normalized_laplacians = " → ".join(
        f"{float(case.metrics['top_grid/laplacian_rms_normalized']):.2f}"
        for case in selected
    )
    total_forces = np.asarray(
        [float(case.metrics["total_bottom_force"]) for case in selected],
        dtype=np.float64,
    )
    total_force_spread = float(np.ptp(total_forces) / total_forces.mean())

    def reduction(name: str) -> float:
        denominator = float(thin[name])
        if abs(denominator) <= np.finfo(np.float64).eps:
            return math.nan
        return 1.0 - float(thick[name]) / denominator

    command = shlex.join(sys.orig_argv)
    environment = " ".join(
        f"{name}={shlex.quote(os.environ[name])}"
        for name in ("DEBUG", "CHERRIES_NAME", "CHERRIES_TAGS")
        if name in os.environ
    )
    full_command = f"{environment} {command}".strip()
    run_logging_note = (
        "This run used DEBUG=1, so Cherries logging is local-only and has no Comet "
        "run. Inspect the completed local log for the command and runtime metadata."
        if os.environ.get("DEBUG") == "1"
        else "This non-debug run may have a Comet record; inspect the Cherries log and "
        "run snapshot for its identifier and runtime metadata."
    )
    report = "\n".join(
        [
            "# Large-deformation Fat-layer Thickness Sweep",
            "",
            "## Purpose",
            "",
            "Increase the simulated bottom pressure through continuation so that the",
            "effect of top-fat thickness is visually legible without multiplying the",
            "rendered displacement. The three reported thicknesses are 0.04, 0.08,",
            "and 0.12; the reported pressures are 0.30, 0.45, and 0.60. All quantities",
            "are model units; no SI calibration is asserted.",
            "All displacement components are fixed on the four vertical sides;",
            "positive-y pressure acts on the free bottom-interior surface. The SMAS",
            "layer uses the fixed active pre-strain listed in the controlled config.",
            "",
            "## Command",
            "",
            f"Working directory: `{Path.cwd()}`",
            "",
            "```console",
            full_command,
            "```",
            "",
            "## Selected common pressure",
            "",
            f"The highest attempted report pressure passing the heuristic display gate "
            f"for all three thicknesses was **{render_pressure:.2f}** model stress. "
            "This is not a measured maximum-safe load.",
            "",
            "| top fat [model length] | solver | steps | free-grad RMS [model force] | max displacement [model length] | top p95-p05 [model length] | top Lap RMS [model length^-1] | min detF | q0.001 detF |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            *table_rows,
            "",
            f"At bottom pressure {render_pressure:.2f}, on the common material-coordinate grid,",
            "the thinnest-to-thickest",
            f"p95-p05 reduction was {reduction('top_grid/u_y_p95_minus_p05'):.1%}, "
            "and the finite-difference Laplacian RMS reduction was "
            f"{reduction('top_grid/laplacian_rms'):.1%}.",
            f"Maximum displacement changes by only {reduction('max_displacement'):.1%}. "
            f"The normalized Laplacian is {normalized_laplacians}, so scale-normalized "
            "smoothing is not monotone from 0.08 to 0.12. The supported claim is only",
            "reduced absolute surface variation in this controlled block, not a claim",
            "about anatomical faces or scale-invariant smoothing.",
            "",
            "## Outputs",
            "",
            f"- `{cfg.output_summary_json.relative_to(EXPERIMENT_ROOT)}`",
            f"- `{cfg.output_summary_csv.relative_to(EXPERIMENT_ROOT)}`",
            f"- `{cfg.output_isometric.relative_to(EXPERIMENT_ROOT)}`: simulated outer-surface u_y plus rest outline",
            f"- `{cfg.output_section.relative_to(EXPERIMENT_ROOT)}`: layered central section plus rest outline",
            f"- `{cfg.output_top_heatmap.relative_to(EXPERIMENT_ROOT)}`: shared-scale top-surface u_y on a common x-z grid",
            "- Per-pressure VTU states and interpolated top-grid NPZ files under `data/`",
            "",
            "## Safety and limitations",
            "",
            "The display gate requires solver success, finite values, no inverted",
            f"tetrahedra, min detF >= {cfg.display_min_det_f:.2f}, q0.001 detF >= "
            f"{cfg.display_min_det_f_q001:.2f}, and no flipped top triangles. The "
            "continuation gate also requires solver success, finite values, no inversion,",
            f"no top flip, and min detF >= {cfg.abort_min_det_f:.2f}.",
            "These are heuristic geometry screens, not physical-validity guarantees.",
            "The 0.12 case has one tetrahedron below detF 0.5 at every attempted load,",
            "although it remains positive and passes the stated display thresholds.",
            "Self-collision remains disabled because the installed IPC path crashes",
            "on the empty collision set in this sandwich model. Positive local detF",
            "does not certify absence of global self-intersection, so the render must",
            "also be inspected before drawing a mechanics conclusion.",
            "",
            "Each thickness is remeshed independently, but the reported surface",
            "metrics are interpolated in undeformed material coordinates onto the same",
            f"{cfg.grid_size} x {cfg.grid_size} x-z grid. This avoids directly comparing",
            "different graph neighborhoods, but does not remove surface interpolation",
            "or independent volumetric-remeshing bias.",
            f"At bottom pressure {render_pressure:.2f}, integrated force differs by "
            f"{total_force_spread:.2%} across the independently remeshed cases.",
            "",
            run_logging_note,
        ]
    )
    cfg.output_report.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_report.write_text(report + "\n", encoding="utf-8")
    cherries.log_output(cfg.output_report)


def log_case_metrics(case: CaseResult, *, step: int) -> None:
    row = case.metrics
    cherries.set_step(step)
    cherries.log_metrics(
        {
            "case": {
                "top_fat_thickness": row["top_fat_thickness"],
                "pressure": row["pressure"],
                "warm_started": row["warm_started"],
                "display_valid": row["display_valid"],
                "continuation_valid": row["continuation_valid"],
            },
            "solver": {
                "success": row["solver/success"],
                "steps": row["solver/steps"],
                "initial_free_grad_l2": row["solver/initial_free_grad_l2"],
                "final_free_grad_l2": row["solver/final_free_grad_l2"],
                "final_free_grad_rms": row["solver/final_free_grad_rms"],
                "final_free_grad_linf": row["solver/final_free_grad_linf"],
                "final_over_initial_free_grad_l2": row[
                    "solver/final_over_initial_free_grad_l2"
                ],
            },
            "deformation": {
                "max_displacement": row["max_displacement"],
                "detF_min": row["deformation/detF_min"],
                "detF_q001": row["deformation/detF_q001"],
                "detF_inverted": row["deformation/detF_inverted"],
            },
            "top_grid": {
                "u_y_mean": row["top_grid/u_y_mean"],
                "u_y_p95_minus_p05": row["top_grid/u_y_p95_minus_p05"],
                "laplacian_rms": row["top_grid/laplacian_rms"],
                "laplacian_rms_normalized": row["top_grid/laplacian_rms_normalized"],
            },
        }
    )


def main(cfg: Config) -> None:
    thicknesses, reports = validate_config(cfg)
    pressures = continuation_pressures(reports, cfg.continuation_step)
    helpers = load_sweep_helpers()
    helpers.validate_output_paths(
        planned_output_paths(cfg, thicknesses=thicknesses, pressures=pressures),
        experiment_root=EXPERIMENT_ROOT,
        overwrite=cfg.overwrite,
    )

    baseline = helpers.load_baseline()
    baseline.configure_runtime()
    cases: list[CaseResult] = []
    data_dir = cfg.output_summary_json.parent
    global_step = 0
    for thickness in thicknesses:
        input_path = data_dir / f"30-{thickness_label(thickness)}-input.vtu"
        body = make_body(
            baseline,
            cfg,
            thickness=thickness,
            pressure=min(pressures),
            output_input=input_path,
        )
        previous_u: np.ndarray | None = None
        for pressure in pressures:
            label = case_label(thickness=thickness, pressure=pressure)
            output_mesh = data_dir / f"30-{label}.vtu"
            output_grid = data_dir / f"30-{label}-top-grid.npz"
            logger.info(
                "Solving top fat %.3f at pressure %.3f (warm start: %s)",
                thickness,
                pressure,
                previous_u is not None,
            )
            case, final_u, continuation_valid = solve_pressure(
                helpers,
                baseline,
                cfg,
                base_body=body,
                thickness=thickness,
                pressure=pressure,
                previous_u=previous_u,
                output_input=input_path,
                output_mesh=output_mesh,
                output_grid=output_grid,
            )
            cases.append(case)
            log_case_metrics(case, step=global_step)
            global_step += 1
            if not continuation_valid:
                logger.error(
                    "Stopping thickness %.3f after unsafe pressure %.3f",
                    thickness,
                    pressure,
                )
                break
            previous_u = final_u

    render_pressure = select_render_pressure(
        cases,
        thicknesses=thicknesses,
        report_pressures=reports,
    )
    selected = [
        find_case(cases, thickness=thickness, pressure=render_pressure)
        for thickness in thicknesses
    ]
    render_isometric(
        helpers,
        selected,
        pressure=render_pressure,
        output=cfg.output_isometric,
    )
    render_section(selected, pressure=render_pressure, output=cfg.output_section)
    render_top_heatmap(
        selected,
        pressure=render_pressure,
        output=cfg.output_top_heatmap,
    )
    cherries.log_output(cfg.output_isometric)
    cherries.log_output(cfg.output_section)
    cherries.log_output(cfg.output_top_heatmap)
    write_outputs(
        cfg,
        cases,
        thicknesses=thicknesses,
        report_pressures=reports,
        continuation_levels=pressures,
        render_pressure=render_pressure,
    )
    logger.info(
        "Rendered the highest common display-valid pressure %.3f", render_pressure
    )


if __name__ == "__main__":
    cherries.main(main)
