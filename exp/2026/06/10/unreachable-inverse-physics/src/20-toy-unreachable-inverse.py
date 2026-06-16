from __future__ import annotations

import contextlib
import csv
import io
import json
import logging
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pydantic_settings as ps
import pyvista as pv
import torch
import warp as wp

from liblaf import cherries, melon

logger = logging.getLogger(__name__)

FAT_FRACTION = "FatFraction"
MUSCLE_FRACTION = "MuscleFraction"
APONEUROSIS_FRACTION = "AponeurosisFraction"
SMAS_FRACTION = "SmasFraction"
BACKGROUND_FRACTION = "BackgroundFraction"
ACTIVE_FRACTION = "ActiveFraction"
SMAS_STIFFNESS_FRACTION = "SmasStiffnessFraction"
TARGET_SURFACE_MASK = "TargetSurfaceMask"
TOP_SURFACE_MASK = "TopSurfaceMask"
FIXED_BOUNDARY = "FixedBoundary"


@dataclass(frozen=True)
class ResolutionSpec:
    name: str
    x_segments: int
    y_levels: tuple[float, ...]
    z_segments: int


@dataclass(frozen=True)
class ToyCase:
    resolution: ResolutionSpec
    mode: Literal["stretch", "squash"]
    target_y: float

    @property
    def stem(self) -> str:
        return f"20-toy-{self.mode}-{self.resolution.name}"


RESOLUTION_SPECS: dict[str, ResolutionSpec] = {
    "coarse": ResolutionSpec(
        name="coarse",
        x_segments=8,
        y_levels=(0.0, 0.02, 0.04, 0.05, 0.06, 0.08, 0.10),
        z_segments=8,
    ),
    "medium": ResolutionSpec(
        name="medium",
        x_segments=14,
        y_levels=(0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10),
        z_segments=14,
    ),
    "fine": ResolutionSpec(
        name="fine",
        x_segments=20,
        y_levels=(0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10),
        z_segments=20,
    ),
}


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    output_summary: Path = cherries.output("20-toy-unreachable-inverse-summary.json")
    output_csv: Path = cherries.output("20-toy-unreachable-inverse-cases.csv")
    output_table: Path = cherries.output("20-toy-unreachable-inverse-table.md")

    resolutions: tuple[str, ...] = ("coarse", "medium", "fine")
    modes: tuple[Literal["stretch", "squash"], ...] = ("stretch", "squash")
    target_magnitude: float = 0.02

    fat_E: float = 1.0
    muscle_E: float = 1.0e2
    aponeurosis_E: float = 1.0e2
    nu: float = 0.49
    active_fraction_tol: float = 1.0e-6

    forward_rtol: float = 5.0e-4
    forward_atol: float = 0.0
    forward_max_steps: int = 5000
    inverse_max_steps: int = 120
    inverse_lr: float = 0.04
    adam_beta1: float = 0.3
    adam_beta2: float = 0.9
    adam_eps: float = 1.0e-8
    activation_inv_abs_max: float = 0.8
    activation_l2_weight: float = 1.0e-5
    series_stride: int = 5
    convergence_window: int = 20
    convergence_loss_ratio_tol: float = 0.01


def configure_runtime() -> None:
    if not torch.cuda.is_available():
        msg = "This experiment uses Warp kernels through Torch and needs CUDA."
        raise RuntimeError(msg)
    logging.getLogger("liblaf.apple.forward._forward").setLevel(logging.WARNING)
    logging.getLogger("liblaf.apple.inverse._diff_forward").setLevel(logging.WARNING)
    warnings.filterwarnings(
        "ignore",
        message=r"The \.grad attribute of a Tensor that is not a leaf Tensor.*",
        category=UserWarning,
    )
    torch.set_default_dtype(torch.float64)
    torch.set_default_device("cuda")
    wp.config.mode = "release"
    wp.init()


def lame_parameters(E: float, nu: float) -> tuple[float, float]:
    lambda_ = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    return float(lambda_), float(mu)


def grid_index(i: int, j: int, k: int, ny: int, nz: int) -> int:
    return (i * ny + j) * nz + k


def orient_tet(points: np.ndarray, tet: list[int]) -> list[int]:
    p0, p1, p2, p3 = points[np.asarray(tet)]
    signed = float(np.dot(np.cross(p1 - p0, p2 - p0), p3 - p0))
    if signed < 0.0:
        tet[2], tet[3] = tet[3], tet[2]
    return tet


def make_tet_box(spec: ResolutionSpec) -> pv.UnstructuredGrid:
    x_values = np.linspace(0.0, 1.0, spec.x_segments + 1)
    y_values = np.asarray(spec.y_levels, dtype=np.float64)
    z_values = np.linspace(0.0, 1.0, spec.z_segments + 1)
    points = np.asarray(
        [(x, y, z) for x in x_values for y in y_values for z in z_values],
        dtype=np.float64,
    )
    ny = y_values.size
    nz = z_values.size
    cells: list[int] = []
    cell_types: list[int] = []
    for i in range(spec.x_segments):
        for j in range(ny - 1):
            for k in range(spec.z_segments):
                v000 = grid_index(i, j, k, ny, nz)
                v100 = grid_index(i + 1, j, k, ny, nz)
                v010 = grid_index(i, j + 1, k, ny, nz)
                v110 = grid_index(i + 1, j + 1, k, ny, nz)
                v001 = grid_index(i, j, k + 1, ny, nz)
                v101 = grid_index(i + 1, j, k + 1, ny, nz)
                v011 = grid_index(i, j + 1, k + 1, ny, nz)
                v111 = grid_index(i + 1, j + 1, k + 1, ny, nz)
                cube_tets = (
                    [v000, v100, v110, v111],
                    [v000, v110, v010, v111],
                    [v000, v010, v011, v111],
                    [v000, v011, v001, v111],
                    [v000, v001, v101, v111],
                    [v000, v101, v100, v111],
                )
                for tet in cube_tets:
                    cells.extend([4, *orient_tet(points, list(tet))])
                    cell_types.append(int(pv.CellType.TETRA))
    mesh = pv.UnstructuredGrid(np.asarray(cells), np.asarray(cell_types), points)
    return mesh


def tetra_cells(mesh: pv.UnstructuredGrid) -> np.ndarray:
    return np.asarray(mesh.cells_dict[pv.CellType.TETRA], dtype=np.int64)


def tetra_signed_volumes(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    p0 = points[tets[:, 0]]
    p1 = points[tets[:, 1]]
    p2 = points[tets[:, 2]]
    p3 = points[tets[:, 3]]
    return np.einsum("ij,ij->i", np.cross(p1 - p0, p2 - p0), p3 - p0) / 6.0


def tetra_volumes(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    return np.abs(tetra_signed_volumes(points, tets))


def rel_change(value: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return (
        np.divide(
            value,
            reference,
            out=np.full_like(value, np.nan, dtype=np.float64),
            where=reference != 0.0,
        )
        - 1.0
    )


def add_tetra_volume_change_fields(
    mesh: pv.UnstructuredGrid,
    target: np.ndarray,
    displacement: np.ndarray,
) -> None:
    points = np.asarray(mesh.points, dtype=np.float64)
    tets = tetra_cells(mesh)
    rest_signed = tetra_signed_volumes(points, tets)
    target_signed = tetra_signed_volumes(points + target, tets)
    inverse_signed = tetra_signed_volumes(points + displacement, tets)
    rest_volume = np.abs(rest_signed)
    target_volume = np.abs(target_signed)
    inverse_volume = np.abs(inverse_signed)

    mesh.cell_data["VolumeInitial"] = rest_volume
    mesh.cell_data["VolumeTarget"] = target_volume
    mesh.cell_data["VolumeInverse"] = inverse_volume
    mesh.cell_data["VolumeTargetRelChange"] = rel_change(target_volume, rest_volume)
    mesh.cell_data["VolumeInverseRelChange"] = rel_change(inverse_volume, rest_volume)
    mesh.cell_data["SignedVolumeInitial"] = rest_signed
    mesh.cell_data["SignedVolumeTarget"] = target_signed
    mesh.cell_data["SignedVolumeInverse"] = inverse_signed
    mesh.cell_data["SignedVolumeTargetRelChange"] = rel_change(
        target_signed, rest_signed
    )
    mesh.cell_data["SignedVolumeInverseRelChange"] = rel_change(
        inverse_signed, rest_signed
    )


def triangle_areas(points: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    p0 = points[triangles[:, 0]]
    p1 = points[triangles[:, 1]]
    p2 = points[triangles[:, 2]]
    return 0.5 * np.linalg.norm(np.cross(p1 - p0, p2 - p0), axis=1)


def surface_triangles(mesh: pv.UnstructuredGrid) -> np.ndarray:
    surface = mesh.extract_surface(algorithm=None).triangulate()
    faces = np.asarray(surface.faces, dtype=np.int64).reshape(-1, 4)
    original_ids = np.asarray(surface.point_data["vtkOriginalPointIds"], dtype=np.int64)
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


def add_material_and_boundary_fields(mesh: pv.UnstructuredGrid, cfg: Config) -> None:
    from liblaf.apple.common import ACTIVATION, ACTIVATION_INV, FIXED_MASK, FIXED_VALUE

    points = np.asarray(mesh.points, dtype=np.float64)
    tets = tetra_cells(mesh)
    centers = points[tets].mean(axis=1)
    x, y, z = centers[:, 0], centers[:, 1], centers[:, 2]

    muscle = (
        (x >= 0.0) & (x <= 0.5) & (y >= 0.04) & (y <= 0.06) & (z >= 0.4) & (z <= 0.6)
    ).astype(np.float64)
    smas = ((y >= 0.04) & (y <= 0.06)).astype(np.float64)
    aponeurosis = np.maximum(0.0, smas - muscle)
    fat = np.clip(1.0 - aponeurosis - muscle, 0.0, 1.0)

    active = muscle > cfg.active_fraction_tol
    zero_activation = np.zeros((mesh.n_cells, 6), dtype=np.float64)

    mesh.cell_data[MUSCLE_FRACTION] = muscle
    mesh.cell_data[SMAS_FRACTION] = smas
    mesh.cell_data[APONEUROSIS_FRACTION] = aponeurosis
    mesh.cell_data[FAT_FRACTION] = fat
    mesh.cell_data[BACKGROUND_FRACTION] = fat
    mesh.cell_data[ACTIVE_FRACTION] = muscle
    mesh.cell_data[SMAS_STIFFNESS_FRACTION] = aponeurosis
    mesh.cell_data["ActivationMask"] = active.astype(np.int8)
    mesh.cell_data["Volume"] = tetra_volumes(points, tets)
    mesh.cell_data[ACTIVATION.vtk] = zero_activation.copy()
    mesh.cell_data[ACTIVATION_INV.vtk] = zero_activation.copy()

    point_x, point_y, point_z = points[:, 0], points[:, 1], points[:, 2]
    eps = 1.0e-10
    bottom = np.isclose(point_y, 0.0, atol=eps)
    sides = (
        np.isclose(point_x, 0.0, atol=eps)
        | np.isclose(point_x, 1.0, atol=eps)
        | np.isclose(point_z, 0.0, atol=eps)
        | np.isclose(point_z, 1.0, atol=eps)
    )
    fixed = bottom | sides
    top = np.isclose(point_y, 0.10, atol=eps)
    target = top & ~fixed
    fixed_mask = np.repeat(fixed[:, np.newaxis], 3, axis=1)
    fixed_value = np.zeros((mesh.n_points, 3), dtype=np.float64)

    mesh.point_data["FixedBottom"] = bottom.astype(np.int8)
    mesh.point_data["FixedSide"] = sides.astype(np.int8)
    mesh.point_data[FIXED_BOUNDARY] = fixed.astype(np.int8)
    mesh.point_data[TOP_SURFACE_MASK] = top.astype(np.int8)
    mesh.point_data[TARGET_SURFACE_MASK] = target.astype(np.int8)
    mesh.point_data[FIXED_MASK.vtk] = fixed_mask
    mesh.point_data[FIXED_VALUE.vtk] = fixed_value


def set_material(
    mesh: pv.UnstructuredGrid,
    *,
    E: float,
    nu: float,
    fraction: np.ndarray,
) -> None:
    from liblaf.apple.common import FRACTION, LAMBDA, MU, NU
    from liblaf.apple.common import E as YOUNG_MODULUS

    lambda_, mu = lame_parameters(E, nu)
    mesh.cell_data[YOUNG_MODULUS.vtk] = np.full(mesh.n_cells, E, dtype=np.float64)
    mesh.cell_data[NU.vtk] = np.full(mesh.n_cells, nu, dtype=np.float64)
    mesh.cell_data[LAMBDA.vtk] = np.full(mesh.n_cells, lambda_, dtype=np.float64)
    mesh.cell_data[MU.vtk] = np.full(mesh.n_cells, mu, dtype=np.float64)
    mesh.cell_data[FRACTION.vtk] = np.asarray(fraction, dtype=np.float64)


def build_forward(mesh: pv.UnstructuredGrid, cfg: Config):
    from liblaf.apple.forward import Forward, ModelBuilder
    from liblaf.apple.warp.fem import StableNeoHookean, StableNeoHookeanActive

    builder = ModelBuilder()
    builder.add_vertices(mesh)
    builder.add_fixed(mesh)

    set_material(mesh, E=cfg.fat_E, nu=cfg.nu, fraction=mesh.cell_data[FAT_FRACTION])
    builder.add_potential(StableNeoHookean.from_pyvista(mesh, name="fat"))

    set_material(
        mesh,
        E=cfg.muscle_E,
        nu=cfg.nu,
        fraction=mesh.cell_data[MUSCLE_FRACTION],
    )
    builder.add_potential(StableNeoHookeanActive.from_pyvista(mesh, name="muscle"))

    set_material(
        mesh,
        E=cfg.aponeurosis_E,
        nu=cfg.nu,
        fraction=mesh.cell_data[APONEUROSIS_FRACTION],
    )
    builder.add_potential(StableNeoHookean.from_pyvista(mesh, name="aponeurosis"))

    forward = Forward(builder.finalize())
    forward.optimizer = forward.default_optimizer(
        max_steps=cfg.forward_max_steps,
        atol=cfg.forward_atol,
        rtol=cfg.forward_rtol,
    )
    return forward


def full_activation_inv_from_active(
    active_activation_inv: torch.Tensor,
    active_ids_t: torch.Tensor,
    n_cells: int,
) -> torch.Tensor:
    full = torch.zeros(
        (n_cells, 6),
        dtype=active_activation_inv.dtype,
        device=active_activation_inv.device,
    )
    return full.index_copy(0, active_ids_t, active_activation_inv)


def material_tree(
    base_materials: dict[str, dict[str, torch.Tensor]],
    active_activation_inv: torch.Tensor,
    active_ids_t: torch.Tensor,
    n_cells: int,
) -> dict[str, dict[str, torch.Tensor]]:
    materials = {name: dict(values) for name, values in base_materials.items()}
    materials["muscle"]["activation_inv"] = full_activation_inv_from_active(
        active_activation_inv, active_ids_t, n_cells
    )
    return materials


def target_displacement(mesh: pv.UnstructuredGrid, target_y: float) -> np.ndarray:
    target_mask = np.asarray(mesh.point_data[TARGET_SURFACE_MASK], dtype=bool)
    displacement = np.zeros((mesh.n_points, 3), dtype=np.float64)
    displacement[target_mask, 1] = target_y
    return displacement


def make_target_mesh(
    mesh: pv.UnstructuredGrid, displacement: np.ndarray
) -> pv.UnstructuredGrid:
    result = mesh.copy(deep=True)
    result.point_data["Displacement"] = displacement
    result.point_data["TargetDisplacement"] = displacement
    result.point_data["TargetPoint"] = result.points + displacement
    add_tetra_volume_change_fields(result, displacement, displacement)
    return result


def make_result_mesh(
    mesh: pv.UnstructuredGrid,
    target: np.ndarray,
    displacement: np.ndarray,
    activation_inv: np.ndarray,
    metrics: dict[str, float | int | bool | str],
) -> pv.UnstructuredGrid:
    from liblaf.apple.common import ACTIVATION_INV

    result = mesh.copy(deep=True)
    error = displacement - target
    result.point_data["Displacement"] = displacement
    result.point_data["TargetDisplacement"] = target
    result.point_data["DisplacementError"] = error
    result.point_data["DisplacementErrorNorm"] = np.linalg.norm(error, axis=1)
    result.point_data["DeformedPoint"] = result.points + displacement
    result.point_data["TargetPoint"] = result.points + target
    add_tetra_volume_change_fields(result, target, displacement)
    result.cell_data[ACTIVATION_INV.vtk] = activation_inv
    result.cell_data["RecoveredActivationInv"] = activation_inv
    result.cell_data["RecoveredActivationInvNorm"] = np.linalg.norm(
        activation_inv, axis=1
    )
    for name, value in metrics.items():
        if isinstance(value, str):
            continue
        result.field_data[name] = np.asarray([value])
    return result


def forward_quiet(differentiable_forward: Any, materials: Any) -> torch.Tensor:
    with contextlib.redirect_stdout(io.StringIO()):
        return differentiable_forward.forward(materials)


def to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def point_error_stats(residual: torch.Tensor) -> dict[str, torch.Tensor]:
    point_error = torch.linalg.vector_norm(residual, dim=1)
    return {
        "mean": point_error.mean(),
        "rms": torch.linalg.vector_norm(residual) / math.sqrt(residual.shape[0]),
        "max": point_error.max(),
    }


def geometry_change(
    mesh: pv.UnstructuredGrid, displacement: np.ndarray, target_mask: np.ndarray
) -> dict[str, float]:
    points = np.asarray(mesh.points, dtype=np.float64)
    deformed = points + displacement
    tets = tetra_cells(mesh)
    rest_signed_volume = tetra_signed_volumes(points, tets)
    deformed_signed_volume = tetra_signed_volumes(deformed, tets)
    rest_volume = tetra_volumes(points, tets)
    deformed_volume = tetra_volumes(deformed, tets)
    surface = surface_triangles(mesh)
    rest_area = triangle_areas(points, surface)
    deformed_area = triangle_areas(deformed, surface)
    target_triangles = np.all(target_mask[surface], axis=1)
    volume0 = float(np.sum(rest_volume))
    volume1 = float(np.sum(deformed_volume))
    signed_volume0 = float(np.sum(rest_signed_volume))
    signed_volume1 = float(np.sum(deformed_signed_volume))
    area0 = float(np.sum(rest_area))
    area1 = float(np.sum(deformed_area))
    top_area0 = float(np.sum(rest_area[target_triangles]))
    top_area1 = float(np.sum(deformed_area[target_triangles]))
    return {
        "volume/rest": volume0,
        "volume/deformed": volume1,
        "volume/abs_rel_change": volume1 / volume0 - 1.0,
        "volume/signed_rest": signed_volume0,
        "volume/signed_deformed": signed_volume1,
        "volume/rel_change": signed_volume1 / signed_volume0 - 1.0,
        "volume/inverted_tets": float(np.sum(deformed_signed_volume <= 0.0)),
        "volume/inverted_fraction": float(np.mean(deformed_signed_volume <= 0.0)),
        "surface_area/rest": area0,
        "surface_area/deformed": area1,
        "surface_area/rel_change": area1 / area0 - 1.0,
        "target_area/rest": top_area0,
        "target_area/deformed": top_area1,
        "target_area/rel_change": top_area1 / top_area0 - 1.0
        if top_area0 > 0.0
        else math.nan,
    }


def top_roughness(
    mesh: pv.UnstructuredGrid, displacement: np.ndarray
) -> dict[str, float]:
    target_mask = np.asarray(mesh.point_data[TARGET_SURFACE_MASK], dtype=bool)
    top_ids = np.flatnonzero(target_mask)
    top_y = displacement[top_ids, 1]
    surface = surface_triangles(mesh)
    edges = unique_edges(surface)
    top_edges = edges[target_mask[edges[:, 0]] & target_mask[edges[:, 1]]]
    if top_edges.size == 0:
        edge_rms = math.nan
        edge_max = math.nan
    else:
        edge_delta = displacement[top_edges[:, 0], 1] - displacement[top_edges[:, 1], 1]
        edge_rms = float(np.linalg.norm(edge_delta) / math.sqrt(edge_delta.size))
        edge_max = float(np.abs(edge_delta).max())
    return {
        "top_y/mean": float(top_y.mean()),
        "top_y/std": float(top_y.std()),
        "top_y/min": float(top_y.min()),
        "top_y/max": float(top_y.max()),
        "top_y/range": float(top_y.max() - top_y.min()),
        "top_y/edge_rms": edge_rms,
        "top_y/edge_max": edge_max,
    }


def summarize_case(
    *,
    case: ToyCase,
    mesh: pv.UnstructuredGrid,
    target: np.ndarray,
    displacement: np.ndarray,
    activation_inv: np.ndarray,
    trace: list[dict[str, float]],
    series_frames: int,
    elapsed_s: float,
    cfg: Config,
) -> dict[str, Any]:
    target_mask = np.asarray(mesh.point_data[TARGET_SURFACE_MASK], dtype=bool)
    active_mask = np.asarray(mesh.cell_data["ActivationMask"], dtype=bool)
    error = displacement - target
    error_norm = np.linalg.norm(error[target_mask], axis=1)
    target_norm = np.linalg.norm(target[target_mask], axis=1)
    active_activation_inv = activation_inv[active_mask]
    target_geo = geometry_change(mesh, target, target_mask)
    inverse_geo = geometry_change(mesh, displacement, target_mask)
    roughness = top_roughness(mesh, displacement)
    initial_row = trace[0]
    best_row = min(trace, key=lambda row: row["loss"])
    final_row = trace[-1]
    window = trace[-min(cfg.convergence_window, len(trace)) :]
    window_best = min(window, key=lambda row: row["loss"])
    window_first = window[0]
    last_window_rel_improvement = (
        (window_first["loss"] - window_best["loss"]) / window_first["loss"]
        if window_first["loss"] > 0.0
        else math.nan
    )
    final_loss_over_best = (
        final_row["loss"] / best_row["loss"] if best_row["loss"] > 0.0 else math.nan
    )
    best_in_last_window = bool(best_row["step"] >= window_first["step"])
    converged = bool(
        (not best_in_last_window)
        and final_loss_over_best <= 1.0 + cfg.convergence_loss_ratio_tol
    )
    if best_in_last_window:
        convergence_status = "not_converged_best_in_last_window"
    elif final_loss_over_best > 1.0 + cfg.convergence_loss_ratio_tol:
        convergence_status = "drifted_after_best"
    else:
        convergence_status = "plateaued"
    summary: dict[str, Any] = {
        "case": case.stem,
        "mode": case.mode,
        "resolution": case.resolution.name,
        "x_segments": int(case.resolution.x_segments),
        "y_levels": list(case.resolution.y_levels),
        "z_segments": int(case.resolution.z_segments),
        "target_y": float(case.target_y),
        "n_points": int(mesh.n_points),
        "n_tets": int(mesh.n_cells),
        "n_active_tets": int(active_mask.sum()),
        "n_target_points": int(target_mask.sum()),
        "inverse_max_steps": int(cfg.inverse_max_steps),
        "inverse_lr": float(cfg.inverse_lr),
        "elapsed_s": float(elapsed_s),
        "n_evaluations": len(trace),
        "series_frames": int(series_frames),
        "initial/loss": float(initial_row["loss"]),
        "initial/error_rms": float(initial_row["error_rms"]),
        "initial/error_max": float(initial_row["error_max"]),
        "best/step": float(best_row["step"]),
        "best/loss": float(best_row["loss"]),
        "best/data_loss": float(best_row["data_loss"]),
        "best/reg_loss": float(best_row["reg_loss"]),
        "best/error_mean": float(error_norm.mean()),
        "best/error_rms": float(
            np.linalg.norm(error[target_mask]) / math.sqrt(target_mask.sum())
        ),
        "best/error_max": float(error_norm.max()),
        "best/error_rms_fraction_of_target": float(
            np.linalg.norm(error[target_mask]) / np.linalg.norm(target[target_mask])
        )
        if np.linalg.norm(target[target_mask]) > 0.0
        else math.nan,
        "final_step/step": float(final_row["step"]),
        "final_step/loss": float(final_row["loss"]),
        "final_step/error_rms": float(final_row["error_rms"]),
        "final_step/error_max": float(final_row["error_max"]),
        "convergence/converged": converged,
        "convergence/status": convergence_status,
        "convergence/best_in_last_window": best_in_last_window,
        "convergence/final_loss_over_best_loss": float(final_loss_over_best),
        "convergence/last_window_rel_improvement": float(last_window_rel_improvement),
        "target/displacement_rms": float(
            np.linalg.norm(target[target_mask]) / math.sqrt(target_mask.sum())
        ),
        "target/displacement_max": float(target_norm.max()),
        "activation_inv/rms": float(
            np.linalg.norm(active_activation_inv)
            / math.sqrt(max(1, active_activation_inv.size))
        ),
        "activation_inv/max_abs": float(np.abs(active_activation_inv).max()),
        "trace": trace,
    }
    summary.update({f"target/{key}": value for key, value in target_geo.items()})
    summary.update({f"inverse/{key}": value for key, value in inverse_geo.items()})
    summary.update({f"inverse/{key}": value for key, value in roughness.items()})
    return summary


def solve_case(case: ToyCase, cfg: Config) -> dict[str, Any]:  # noqa: PLR0915
    from liblaf.apple.common import GLOBAL_POINT_ID
    from liblaf.apple.inverse import DifferentiableForward

    start = time.perf_counter()
    mesh = make_tet_box(case.resolution)
    add_material_and_boundary_fields(mesh, cfg)
    target = target_displacement(mesh, case.target_y)
    data_dir = cfg.output_summary.parent
    input_path = data_dir / f"{case.stem}-input.vtu"
    target_path = data_dir / f"{case.stem}-target.vtu"
    output_path = data_dir / f"{case.stem}.vtu"
    series_path = data_dir / f"{case.stem}.vtu.series"
    melon.save(input_path, mesh)
    melon.save(target_path, make_target_mesh(mesh, target))

    inverse_mesh = mesh.copy(deep=True)
    forward = build_forward(inverse_mesh, cfg)
    differentiable_forward = DifferentiableForward(forward)
    base_materials = forward.model.get_materials()
    global_ids = np.asarray(
        inverse_mesh.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64
    )
    target_mask = np.asarray(inverse_mesh.point_data[TARGET_SURFACE_MASK], dtype=bool)
    target_ids = np.flatnonzero(target_mask).astype(np.int64)
    active_ids = np.flatnonzero(
        np.asarray(inverse_mesh.cell_data["ActivationMask"], dtype=bool)
    ).astype(np.int64)
    if active_ids.size == 0:
        msg = f"{case.stem} has no active muscle tetrahedra"
        raise ValueError(msg)

    target_t = torch.as_tensor(
        target, dtype=torch.get_default_dtype(), device=torch.get_default_device()
    )
    target_ids_t = torch.as_tensor(
        target_ids, dtype=torch.long, device=torch.get_default_device()
    )
    target_global_ids_t = torch.as_tensor(
        global_ids[target_ids], dtype=torch.long, device=torch.get_default_device()
    )
    active_ids_t = torch.as_tensor(
        active_ids, dtype=torch.long, device=torch.get_default_device()
    )
    active_activation_inv = torch.nn.Parameter(
        torch.zeros(
            (active_ids.size, 6),
            dtype=torch.get_default_dtype(),
            device=torch.get_default_device(),
        )
    )
    optimizer = torch.optim.Adam(
        [active_activation_inv],
        lr=cfg.inverse_lr,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_eps,
    )

    best_loss = math.inf
    best_displacement: np.ndarray | None = None
    best_activation_inv: np.ndarray | None = None
    trace: list[dict[str, float]] = []
    series_frames = 0
    with melon.SeriesWriter(series_path, clear=True) as series_writer:
        for step in range(cfg.inverse_max_steps + 1):
            cherries.set_step(len(trace))
            optimizer.zero_grad()
            active_clamped = torch.clamp(
                active_activation_inv,
                min=-cfg.activation_inv_abs_max,
                max=cfg.activation_inv_abs_max,
            )
            materials = material_tree(
                base_materials, active_clamped, active_ids_t, inverse_mesh.n_cells
            )
            output = forward_quiet(differentiable_forward, materials)
            residual = output[target_global_ids_t] - target_t[target_ids_t]
            data_loss = residual.square().mean()
            reg_loss = cfg.activation_l2_weight * active_clamped.square().mean()
            loss = data_loss + reg_loss
            loss.backward()
            if active_activation_inv.grad is None:
                msg = "differentiable forward did not produce activation gradients"
                raise RuntimeError(msg)
            grad = active_activation_inv.grad.detach()
            error_stats = point_error_stats(residual.detach())
            displacement = to_numpy(output)[global_ids]
            full_activation_inv = to_numpy(
                full_activation_inv_from_active(
                    active_clamped.detach(), active_ids_t, inverse_mesh.n_cells
                )
            )
            loss_value = float(loss.detach().cpu())
            row = {
                "step": float(step),
                "loss": loss_value,
                "data_loss": float(data_loss.detach().cpu()),
                "reg_loss": float(reg_loss.detach().cpu()),
                "error_mean": float(error_stats["mean"].cpu()),
                "error_rms": float(error_stats["rms"].cpu()),
                "error_max": float(error_stats["max"].cpu()),
                "activation_inv_rms": float(
                    torch.linalg.vector_norm(active_clamped.detach()).cpu()
                    / math.sqrt(active_clamped.numel())
                ),
                "activation_inv_max_abs": float(
                    active_clamped.detach().abs().max().cpu()
                ),
                "grad_norm": float(torch.linalg.vector_norm(grad).cpu()),
            }
            if loss_value < best_loss:
                best_loss = loss_value
                best_displacement = displacement
                best_activation_inv = full_activation_inv
            trace.append(row)
            if step % cfg.series_stride == 0 or step == cfg.inverse_max_steps:
                step_mesh = make_result_mesh(
                    inverse_mesh,
                    target,
                    displacement,
                    full_activation_inv,
                    row,
                )
                series_writer.append(step_mesh, time=float(step))
                series_frames += 1
            cherries.log_metrics(
                {
                    f"{case.stem}/loss": row["loss"],
                    f"{case.stem}/error_rms": row["error_rms"],
                    f"{case.stem}/error_max": row["error_max"],
                    f"{case.stem}/activation_inv_rms": row["activation_inv_rms"],
                }
            )
            logger.info(
                "%s step %03d loss %.6g rms %.6g max %.6g grad %.6g",
                case.stem,
                step,
                row["loss"],
                row["error_rms"],
                row["error_max"],
                row["grad_norm"],
            )
            if step < cfg.inverse_max_steps:
                optimizer.step()
                with torch.no_grad():
                    active_activation_inv.clamp_(
                        -cfg.activation_inv_abs_max, cfg.activation_inv_abs_max
                    )

    if best_displacement is None or best_activation_inv is None:
        msg = f"{case.stem} did not evaluate any inverse state"
        raise RuntimeError(msg)
    elapsed = time.perf_counter() - start
    summary = summarize_case(
        case=case,
        mesh=inverse_mesh,
        target=target,
        displacement=best_displacement,
        activation_inv=best_activation_inv,
        trace=trace,
        series_frames=series_frames,
        elapsed_s=elapsed,
        cfg=cfg,
    )
    result_metrics = {
        key: value
        for key, value in summary.items()
        if isinstance(value, int | float | bool)
    }
    melon.save(
        output_path,
        make_result_mesh(
            inverse_mesh,
            target,
            best_displacement,
            best_activation_inv,
            result_metrics,
        ),
    )
    cherries.log_output(input_path)
    cherries.log_output(target_path)
    cherries.log_output(output_path)
    cherries.log_output(series_path)
    return summary


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
        "| case | points | tets | active tets | best step | convergence | target signed volume change | inverse signed volume change | target inverted tets | inverse inverted tets | best error RMS | best error/target RMS | top y std | top edge RMS |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        (
            "| "
            + " | ".join(
                [
                    str(row["case"]),
                    str(row["n_points"]),
                    str(row["n_tets"]),
                    str(row["n_active_tets"]),
                    format_float(row["best/step"]),
                    str(row["convergence/status"]),
                    format_float(row["target/volume/rel_change"]),
                    format_float(row["inverse/volume/rel_change"]),
                    format_float(row["target/volume/inverted_fraction"]),
                    format_float(row["inverse/volume/inverted_fraction"]),
                    format_float(row["best/error_rms"]),
                    format_float(row["best/error_rms_fraction_of_target"]),
                    format_float(row["inverse/top_y/std"]),
                    format_float(row["inverse/top_y/edge_rms"]),
                ]
            )
            + " |"
        )
        for row in rows
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def selected_cases(cfg: Config) -> list[ToyCase]:
    cases: list[ToyCase] = []
    for resolution_name in cfg.resolutions:
        if resolution_name not in RESOLUTION_SPECS:
            msg = (
                f"unknown resolution {resolution_name!r}; "
                f"choose from {sorted(RESOLUTION_SPECS)}"
            )
            raise ValueError(msg)
        spec = RESOLUTION_SPECS[resolution_name]
        for mode in cfg.modes:
            target_y = (
                cfg.target_magnitude if mode == "stretch" else -cfg.target_magnitude
            )
            cases.append(ToyCase(resolution=spec, mode=mode, target_y=target_y))
    return cases


def main(cfg: Config) -> None:
    configure_runtime()
    rows = [solve_case(case, cfg) for case in selected_cases(cfg)]
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
