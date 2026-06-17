from __future__ import annotations

import contextlib
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

ALL_BOUNDS = (0.0, 1.0, 0.0, 0.1, 0.0, 1.0)
SMAS_BOUNDS = (0.0, 1.0, 0.04, 0.06, 0.0, 1.0)
MUSCLE_BOUNDS = (0.0, 0.5, 0.04, 0.06, 0.4, 0.6)

FAT_FRACTION = "FatFraction"
MUSCLE_FRACTION = "MuscleFraction"
APONEUROSIS_FRACTION = "AponeurosisFraction"
SMAS_FRACTION = "SmasFraction"
FRACTION_SUM = "FractionSum"
TARGET_SURFACE_MASK = "TargetSurfaceMask"
TOP_SURFACE_MASK = "TopSurfaceMask"
FIXED_BOUNDARY = "FixedBoundary"
ACTIVE_FRACTION = "ActiveFraction"

STRETCH_TARGET_MAGNITUDE = 0.1
SQUASH_TARGET_MAGNITUDE = 0.05
BOUNDARY_TOL_SCALE = 0.4
BOUNDARY_TOL_MIN = 5.0e-4
FRACTION_SAMPLES_PER_TET = 16
FRACTION_CHUNK_TETS = 20_000

FAT_E = 0.003
FAT_NU = 0.49
MUSCLE_E = 0.030
MUSCLE_NU = 0.49
APONEUROSIS_E = 0.10
APONEUROSIS_NU = 0.35
SKIN_E = 0.20
SKIN_NU = 0.49
SKIN_THICKNESS = 1.0
SKIN_PRESTRAIN = 0.10
ACTIVE_FRACTION_TOL = 1.0e-6

FORWARD_RTOL = 5.0e-4
FORWARD_ATOL = 1.0e-10
FORWARD_MAX_STEPS = 5000
ADJOINT_RTOL = 5.0e-4
ADJOINT_ATOL = 0.0
ADJOINT_MAXITER = 10_000

INVERSE_PATIENCE = 20


@dataclass(frozen=True)
class ResolutionSpec:
    name: str
    lr: float


@dataclass(frozen=True)
class LossVariant:
    name: Literal["l2", "laplacian"]
    skin_prestrain: bool
    activation_mode: Literal["per-tet", "per-tet-smooth", "shared"]

    @property
    def label(self) -> str:
        prestrain = "skin-prestrain10" if self.skin_prestrain else "skin-prestrain0"
        mode = self.activation_mode.replace("-", "_")
        return f"{self.name}-{prestrain}-activation-{mode}"

    @property
    def activation_smooth(self) -> bool:
        return self.activation_mode == "per-tet-smooth"

    @property
    def shared_activation(self) -> bool:
        return self.activation_mode == "shared"


@dataclass(frozen=True)
class ToyCase:
    resolution: ResolutionSpec
    mode: Literal["stretch", "squash"]
    variant: LossVariant
    target_y: float

    @property
    def stem(self) -> str:
        return f"20-toy-tetwild-{self.mode}-{self.resolution.name}-{self.variant.label}"


class PrepareConfig(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    output_mesh: Path = Path("data/10-toy-tetwild-lr001-prepared.vtu")
    output_summary: Path = Path("data/10-toy-tetwild-lr001-prepared-summary.json")

    tetwild_lr: float = 0.01


class InverseConfig(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    input_mesh: Path = Path("data/10-toy-tetwild-lr001-prepared.vtu")
    output_summary: Path = Path("data/20-unreachable-toy-skin-tetwild-summary.json")
    output_table: Path = Path("data/20-unreachable-toy-skin-tetwild-table.md")

    mode: str = "squash"
    loss_variant: str = "l2"
    skin_prestrain_enabled: bool = False
    activation_mode: str = "per-tet"
    compare_existing: bool = False

    inverse_lr: float = 0.03
    inverse_max_steps: int = 200
    inverse_loss_min_delta: float = 1.0e-8
    residual_laplacian_weight: float = 10.0
    activation_smooth_weight: float = 1.0e-2
    require_convergence: bool = True


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


def label_lr(lr: float) -> str:
    return f"lr{lr:g}".replace("0.", "0").replace(".", "p")


def selected_mode(cfg: InverseConfig) -> Literal["stretch", "squash"]:
    if cfg.mode not in {"stretch", "squash"}:
        msg = f"unknown mode {cfg.mode!r}; expected stretch or squash"
        raise ValueError(msg)
    mode: Literal["stretch", "squash"] = cfg.mode  # pyright: ignore[reportAssignmentType]
    return mode


def selected_case(cfg: InverseConfig, resolution: ResolutionSpec) -> ToyCase:
    mode = selected_mode(cfg)
    if cfg.loss_variant not in {"l2", "laplacian"}:
        msg = f"unknown loss variant {cfg.loss_variant!r}; expected l2 or laplacian"
        raise ValueError(msg)
    if cfg.activation_mode not in {"per-tet", "per-tet-smooth", "shared"}:
        msg = (
            f"unknown activation mode {cfg.activation_mode!r}; "
            "expected per-tet, per-tet-smooth, or shared"
        )
        raise ValueError(msg)
    loss_name: Literal["l2", "laplacian"] = cfg.loss_variant  # pyright: ignore[reportAssignmentType]
    activation_mode: Literal["per-tet", "per-tet-smooth", "shared"] = cfg.activation_mode  # pyright: ignore[reportAssignmentType]
    return ToyCase(
        resolution=resolution,
        mode=mode,
        target_y=STRETCH_TARGET_MAGNITUDE
        if mode == "stretch"
        else -SQUASH_TARGET_MAGNITUDE,
        variant=LossVariant(
            name=loss_name,
            skin_prestrain=cfg.skin_prestrain_enabled,
            activation_mode=activation_mode,
        ),
    )


def mesh_resolution(mesh: pv.UnstructuredGrid) -> ResolutionSpec:
    if "TetWildLr" not in mesh.field_data:
        return ResolutionSpec(name=label_lr(0.01), lr=0.01)
    lr = float(np.asarray(mesh.field_data["TetWildLr"]).ravel()[0])
    return ResolutionSpec(name=label_lr(lr), lr=lr)


def expected_inverse_cases(
    resolution: ResolutionSpec, mode: Literal["stretch", "squash"]
) -> list[ToyCase]:
    target_y = STRETCH_TARGET_MAGNITUDE if mode == "stretch" else -SQUASH_TARGET_MAGNITUDE
    return [
        ToyCase(
            resolution=resolution,
            mode=mode,
            target_y=target_y,
            variant=LossVariant(
                name=loss_name,
                skin_prestrain=skin_prestrain,
                activation_mode=activation_mode,
            ),
        )
        for loss_name in ("l2", "laplacian")
        for skin_prestrain in (False, True)
        for activation_mode in ("per-tet", "per-tet-smooth", "shared")
    ]


def tetwild_surface() -> pv.PolyData:
    return pv.Box(ALL_BOUNDS, quads=False).triangulate()


def tetra_cells(mesh: pv.UnstructuredGrid) -> np.ndarray:
    if pv.CellType.TETRA not in mesh.cells_dict:
        msg = "expected a tetrahedral unstructured grid"
        raise ValueError(msg)
    return np.asarray(mesh.cells_dict[pv.CellType.TETRA], dtype=np.int64)


def tetra_signed_volumes(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    p0 = points[tets[:, 0]]
    p1 = points[tets[:, 1]]
    p2 = points[tets[:, 2]]
    p3 = points[tets[:, 3]]
    return np.einsum("ij,ij->i", np.cross(p1 - p0, p2 - p0), p3 - p0) / 6.0


def tetra_volumes(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    return np.abs(tetra_signed_volumes(points, tets))


def orient_tetra_mesh(mesh: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    tets = tetra_cells(mesh).copy()
    points = np.asarray(mesh.points, dtype=np.float64)
    signed = tetra_signed_volumes(points, tets)
    flipped = signed < 0.0
    if np.any(flipped):
        tets[flipped, 2], tets[flipped, 3] = tets[flipped, 3], tets[flipped, 2].copy()
    cells = np.empty((tets.shape[0], 5), dtype=np.int64)
    cells[:, 0] = 4
    cells[:, 1:] = tets
    cell_types = np.full(tets.shape[0], int(pv.CellType.TETRA), dtype=np.uint8)
    return pv.UnstructuredGrid(cells.ravel(), cell_types, points)


def make_tetwild_mesh(spec: ResolutionSpec) -> pv.UnstructuredGrid:
    start = time.perf_counter()
    surface = tetwild_surface()
    mesh = melon.ext.tetwild(surface, edge_length_fac=spec.lr)
    if not isinstance(mesh, pv.UnstructuredGrid):
        mesh = mesh.cast_to_unstructured_grid()
    mesh = orient_tetra_mesh(mesh.clean())
    mesh.field_data["TetWildLr"] = np.asarray([spec.lr])
    mesh.field_data["TetWildLrInterpretation"] = np.asarray(
        ["relative_edge_length_fac"]
    )
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


def add_material_and_boundary_fields(
    mesh: pv.UnstructuredGrid, spec: ResolutionSpec
) -> None:
    from liblaf.apple.common import ACTIVATION, ACTIVATION_INV, FIXED_MASK, FIXED_VALUE

    points = np.asarray(mesh.points, dtype=np.float64)
    tets = tetra_cells(mesh)
    barycentric = sample_barycentric(FRACTION_SAMPLES_PER_TET)

    muscle = sampled_box_fraction(
        points=points,
        tets=tets,
        bounds=MUSCLE_BOUNDS,
        barycentric=barycentric,
        chunk_tets=FRACTION_CHUNK_TETS,
    )
    smas = sampled_box_fraction(
        points=points,
        tets=tets,
        bounds=SMAS_BOUNDS,
        barycentric=barycentric,
        chunk_tets=FRACTION_CHUNK_TETS,
    )
    muscle = np.minimum(muscle, smas)
    aponeurosis = np.maximum(0.0, smas - muscle)
    fat = np.clip(1.0 - aponeurosis - muscle, 0.0, 1.0)
    fraction_sum = aponeurosis + fat + muscle
    active = muscle > ACTIVE_FRACTION_TOL
    zero_activation = np.zeros((mesh.n_cells, 6), dtype=np.float64)

    mesh.cell_data[MUSCLE_FRACTION] = muscle
    mesh.cell_data[SMAS_FRACTION] = smas
    mesh.cell_data[APONEUROSIS_FRACTION] = aponeurosis
    mesh.cell_data[FAT_FRACTION] = fat
    mesh.cell_data[FRACTION_SUM] = fraction_sum
    mesh.cell_data[ACTIVE_FRACTION] = muscle
    mesh.cell_data["ActivationMask"] = active.astype(np.int8)
    mesh.cell_data["Volume"] = tetra_volumes(points, tets)
    mesh.cell_data[ACTIVATION.vtk] = zero_activation.copy()
    mesh.cell_data[ACTIVATION_INV.vtk] = zero_activation.copy()
    mesh.field_data["FractionSamplesPerTet"] = np.asarray(
        [FRACTION_SAMPLES_PER_TET]
    )

    tol = max(BOUNDARY_TOL_MIN, BOUNDARY_TOL_SCALE * spec.lr)
    point_x, point_y, point_z = points[:, 0], points[:, 1], points[:, 2]
    bottom = point_y <= ALL_BOUNDS[2] + tol
    sides = (
        (point_x <= ALL_BOUNDS[0] + tol)
        | (point_x >= ALL_BOUNDS[1] - tol)
        | (point_z <= ALL_BOUNDS[4] + tol)
        | (point_z >= ALL_BOUNDS[5] - tol)
    )
    fixed = bottom | sides
    top = point_y >= ALL_BOUNDS[3] - tol
    target = top & ~fixed
    if not np.any(target):
        msg = f"{spec.name} selected no free top-surface target points"
        raise ValueError(msg)

    mesh.point_data["FixedBottom"] = bottom.astype(np.int8)
    mesh.point_data["FixedSide"] = sides.astype(np.int8)
    mesh.point_data[FIXED_BOUNDARY] = fixed.astype(np.int8)
    mesh.point_data[TOP_SURFACE_MASK] = top.astype(np.int8)
    mesh.point_data[TARGET_SURFACE_MASK] = target.astype(np.int8)
    mesh.point_data[FIXED_MASK.vtk] = np.repeat(fixed[:, np.newaxis], 3, axis=1)
    mesh.point_data[FIXED_VALUE.vtk] = np.zeros((mesh.n_points, 3), dtype=np.float64)


def lame_parameters(E: float, nu: float) -> tuple[float, float]:
    lambda_ = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    return float(lambda_), float(mu)


def set_volume_material(
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


def skin_surface(mesh: pv.UnstructuredGrid, *, prestrain: float) -> pv.PolyData:
    from liblaf.apple.common import (
        ACTIVATION_INV,
        FRACTION,
        GLOBAL_POINT_ID,
        LAMBDA,
        MU,
    )

    surface = mesh.extract_surface(algorithm=None).triangulate()
    if "vtkOriginalPointIds" not in surface.point_data:
        msg = "extract_surface did not produce vtkOriginalPointIds"
        raise KeyError(msg)
    original_ids = np.asarray(surface.point_data["vtkOriginalPointIds"], dtype=np.int64)
    surface.point_data[GLOBAL_POINT_ID.vtk] = np.asarray(
        mesh.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64
    )[original_ids]
    lambda_, mu = lame_parameters(SKIN_E, SKIN_NU)
    surface.cell_data[LAMBDA.vtk] = np.full(surface.n_cells, lambda_, dtype=np.float64)
    surface.cell_data[MU.vtk] = np.full(surface.n_cells, mu, dtype=np.float64)
    surface.cell_data[FRACTION.vtk] = np.ones(surface.n_cells, dtype=np.float64)
    activation_inv = np.zeros((surface.n_cells, 3), dtype=np.float64)
    activation_inv[:, 0] = prestrain
    activation_inv[:, 1] = prestrain
    surface.cell_data[ACTIVATION_INV.vtk] = activation_inv
    surface.cell_data["SkinPrestrain"] = np.full(
        surface.n_cells, prestrain, dtype=np.float64
    )
    return surface


def build_forward(mesh: pv.UnstructuredGrid, case: ToyCase):
    from liblaf.apple.forward import Forward, ModelBuilder
    from liblaf.apple.warp.fem import Koiter, StableNeoHookean, StableNeoHookeanActive

    builder = ModelBuilder()
    builder.add_vertices(mesh)
    builder.add_fixed(mesh)

    set_volume_material(
        mesh,
        E=FAT_E,
        nu=FAT_NU,
        fraction=mesh.cell_data[FAT_FRACTION],
    )
    builder.add_potential(StableNeoHookean.from_pyvista(mesh, name="fat"))

    set_volume_material(
        mesh,
        E=MUSCLE_E,
        nu=MUSCLE_NU,
        fraction=mesh.cell_data[MUSCLE_FRACTION],
    )
    builder.add_potential(StableNeoHookeanActive.from_pyvista(mesh, name="muscle"))

    set_volume_material(
        mesh,
        E=APONEUROSIS_E,
        nu=APONEUROSIS_NU,
        fraction=mesh.cell_data[APONEUROSIS_FRACTION],
    )
    builder.add_potential(StableNeoHookean.from_pyvista(mesh, name="aponeurosis"))

    prestrain = SKIN_PRESTRAIN if case.variant.skin_prestrain else 0.0
    skin = skin_surface(mesh, prestrain=prestrain)
    builder.add_potential(
        Koiter.from_pyvista(skin, name="skin", thickness=SKIN_THICKNESS)
    )

    forward = Forward(builder.finalize())
    forward.optimizer = forward.default_optimizer(
        max_steps=FORWARD_MAX_STEPS,
        atol=FORWARD_ATOL,
        rtol=FORWARD_RTOL,
    )
    return forward, skin


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


def active_activation_inv_from_parameter(
    activation_parameter: torch.Tensor,
    *,
    n_active_tets: int,
    shared: bool,
) -> torch.Tensor:
    if shared:
        return activation_parameter.expand(n_active_tets, -1)
    return activation_parameter


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
    return result


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
    surface = mesh.extract_surface(algorithm=None).triangulate()
    if "vtkOriginalPointIds" not in surface.point_data:
        msg = "extract_surface did not produce vtkOriginalPointIds"
        raise KeyError(msg)
    original_ids = np.asarray(surface.point_data["vtkOriginalPointIds"], dtype=np.int64)
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


def top_surface_edges(mesh: pv.UnstructuredGrid) -> np.ndarray:
    triangles = surface_triangles(mesh)
    top = np.asarray(mesh.point_data[TOP_SURFACE_MASK], dtype=bool)
    top_triangles = triangles[np.all(top[triangles], axis=1)]
    if top_triangles.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    return unique_edges(top_triangles).astype(np.int64)


def bumpiness_metrics(
    mesh: pv.UnstructuredGrid, displacement: np.ndarray, target: np.ndarray
) -> dict[str, float]:
    target_mask = np.asarray(mesh.point_data[TARGET_SURFACE_MASK], dtype=bool)
    top_ids = np.flatnonzero(target_mask)
    top_y = displacement[top_ids, 1]
    residual = displacement - target
    edges = top_surface_edges(mesh)
    if edges.size == 0:
        return {
            "bumpiness/top_y_std": math.nan,
            "bumpiness/top_y_range": math.nan,
            "bumpiness/displacement_edge_rms": math.nan,
            "bumpiness/residual_edge_rms": math.nan,
            "bumpiness/displacement_laplacian_rms": math.nan,
            "bumpiness/residual_laplacian_rms": math.nan,
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
        "bumpiness/top_y_std": float(top_y.std()),
        "bumpiness/top_y_range": float(top_y.max() - top_y.min()),
        "bumpiness/displacement_edge_rms": float(
            np.linalg.norm(disp_edge) / math.sqrt(edges.shape[0])
        ),
        "bumpiness/residual_edge_rms": float(
            np.linalg.norm(residual_edge) / math.sqrt(edges.shape[0])
        ),
        "bumpiness/displacement_laplacian_rms": float(
            np.linalg.norm(disp_lap[top_ids]) / math.sqrt(top_ids.size)
        ),
        "bumpiness/residual_laplacian_rms": float(
            np.linalg.norm(residual_lap[top_ids]) / math.sqrt(top_ids.size)
        ),
    }


def to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def to_float(value: Any, default: float = math.nan) -> float:
    if value is None:
        return default
    if torch.is_tensor(value):
        return float(value.detach().cpu())
    return float(value)


def point_error_stats(residual: torch.Tensor) -> dict[str, torch.Tensor]:
    point_error = torch.linalg.vector_norm(residual, dim=1)
    return {
        "mean": point_error.mean(),
        "rms": torch.linalg.vector_norm(residual) / math.sqrt(residual.shape[0]),
        "max": point_error.max(),
    }


def residual_laplacian_loss(
    residual_full: torch.Tensor,
    edge_i: torch.Tensor,
    edge_j: torch.Tensor,
    lap_ids: torch.Tensor,
) -> torch.Tensor:
    if edge_i.numel() == 0 or lap_ids.numel() == 0:
        return torch.zeros((), dtype=residual_full.dtype, device=residual_full.device)
    neighbor_sum = torch.zeros_like(residual_full)
    counts = torch.zeros(
        (residual_full.shape[0],),
        dtype=residual_full.dtype,
        device=residual_full.device,
    )
    ones = torch.ones(edge_i.shape, dtype=residual_full.dtype, device=edge_i.device)
    neighbor_sum.index_add_(0, edge_i, residual_full[edge_j])
    neighbor_sum.index_add_(0, edge_j, residual_full[edge_i])
    counts.index_add_(0, edge_i, ones)
    counts.index_add_(0, edge_j, ones)
    lap_ids = lap_ids[counts[lap_ids] > 0.0]
    if lap_ids.numel() == 0:
        return torch.zeros((), dtype=residual_full.dtype, device=residual_full.device)
    laplacian = residual_full[lap_ids] - neighbor_sum[lap_ids] / counts[lap_ids, None]
    return laplacian.square().mean()


def active_tetra_neighbor_edges(
    mesh: pv.UnstructuredGrid, active_ids: np.ndarray
) -> np.ndarray:
    tets = tetra_cells(mesh)
    face_vertices = np.asarray(
        (
            (0, 1, 2),
            (0, 1, 3),
            (0, 2, 3),
            (1, 2, 3),
        ),
        dtype=np.int64,
    )
    faces = tets[:, face_vertices].reshape(-1, 3)
    faces.sort(axis=1)
    owners = np.repeat(np.arange(tets.shape[0], dtype=np.int64), 4)
    order = np.lexsort((faces[:, 2], faces[:, 1], faces[:, 0]))
    faces = faces[order]
    owners = owners[order]

    active_lookup = np.full(tets.shape[0], -1, dtype=np.int64)
    active_lookup[active_ids] = np.arange(active_ids.size, dtype=np.int64)
    pairs: list[tuple[int, int]] = []
    start = 0
    while start < faces.shape[0]:
        end = start + 1
        while end < faces.shape[0] and np.array_equal(faces[start], faces[end]):
            end += 1
        local_ids = active_lookup[owners[start:end]]
        local_ids = local_ids[local_ids >= 0]
        if local_ids.size >= 2:
            for i in range(local_ids.size - 1):
                for j in range(i + 1, local_ids.size):
                    a = int(local_ids[i])
                    b = int(local_ids[j])
                    pairs.append((a, b) if a < b else (b, a))
        start = end
    if not pairs:
        return np.empty((0, 2), dtype=np.int64)
    return np.unique(np.asarray(pairs, dtype=np.int64), axis=0)


def activation_smooth_loss(
    active_activation_inv: torch.Tensor,
    smooth_i: torch.Tensor,
    smooth_j: torch.Tensor,
) -> torch.Tensor:
    if smooth_i.numel() == 0:
        return torch.zeros(
            (),
            dtype=active_activation_inv.dtype,
            device=active_activation_inv.device,
        )
    diff = active_activation_inv[smooth_i] - active_activation_inv[smooth_j]
    return diff.square().mean()


def forward_quiet(differentiable_forward: Any, materials: Any) -> torch.Tensor:
    with contextlib.redirect_stdout(io.StringIO()):
        return differentiable_forward.forward(materials)


def relative_value(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        return math.nan
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else math.inf
    return numerator / denominator


def forward_solution_metrics(solution: Any) -> dict[str, Any]:
    if solution is None:
        return {
            "forward/result": "missing",
            "forward/success": False,
            "forward/steps": math.nan,
            "forward/grad_norm": math.nan,
            "forward/relative_grad_norm": math.nan,
            "forward/stagnation_count": math.nan,
        }
    convergence_state = solution.state.convergence_state
    grad_norm = to_float(convergence_state.grad_norm)
    grad_norm_first = to_float(convergence_state.grad_norm_first)
    return {
        "forward/result": str(solution.result),
        "forward/success": bool(solution.success),
        "forward/steps": int(convergence_state.step),
        "forward/grad_norm": grad_norm,
        "forward/relative_grad_norm": relative_value(grad_norm, grad_norm_first),
        "forward/grad_norm_first": grad_norm_first,
        "forward/stagnation_count": int(convergence_state.stagnation_count),
    }


def adjoint_solution_metrics(solution: Any) -> dict[str, Any]:
    if solution is None:
        return {
            "adjoint/result": "missing",
            "adjoint/success": False,
            "adjoint/solver_count": 0,
            "adjoint/best_solver": -1,
            "adjoint/absolute_residual": math.nan,
            "adjoint/relative_residual": math.nan,
        }
    state = solution.state
    best_index = int(state.best_index.detach().cpu())
    absolute_residuals = to_numpy(state.absolute_residuals)
    relative_residuals = to_numpy(state.relative_residuals)
    metrics: dict[str, Any] = {
        "adjoint/result": str(solution.result),
        "adjoint/success": bool(solution.success),
        "adjoint/solver_count": len(state.solutions),
        "adjoint/best_solver": best_index,
        "adjoint/absolute_residual": float(absolute_residuals[best_index]),
        "adjoint/relative_residual": float(relative_residuals[best_index]),
    }
    for i, solver_solution in enumerate(state.solutions):
        prefix = f"adjoint/solver_{i}"
        metrics[f"{prefix}/result"] = str(solver_solution.result)
        metrics[f"{prefix}/success"] = bool(solver_solution.success)
        metrics[f"{prefix}/steps"] = (
            -1
            if solver_solution.state.step is None
            else int(solver_solution.state.step)
        )
        metrics[f"{prefix}/absolute_residual"] = float(absolute_residuals[i])
        metrics[f"{prefix}/relative_residual"] = float(relative_residuals[i])
    return metrics


def geometry_summary(mesh: pv.UnstructuredGrid) -> dict[str, float]:
    fractions = np.column_stack(
        (
            np.asarray(mesh.cell_data[APONEUROSIS_FRACTION], dtype=np.float64),
            np.asarray(mesh.cell_data[FAT_FRACTION], dtype=np.float64),
            np.asarray(mesh.cell_data[MUSCLE_FRACTION], dtype=np.float64),
        )
    )
    volumes = np.asarray(mesh.cell_data["Volume"], dtype=np.float64)
    weighted = fractions * volumes[:, None]
    return {
        "volume/total": float(volumes.sum()),
        "volume/aponeurosis": float(weighted[:, 0].sum()),
        "volume/fat": float(weighted[:, 1].sum()),
        "volume/muscle": float(weighted[:, 2].sum()),
        "fraction_sum/min": float(fractions.sum(axis=1).min()),
        "fraction_sum/max": float(fractions.sum(axis=1).max()),
        "fraction_sum/mean": float(fractions.sum(axis=1).mean()),
    }


class RecordingDifferentiableForward:  # attrs subclasses with extra slots are brittle.
    def __init__(self, wrapped: Any, adjoint_solver: Any) -> None:
        from liblaf.apple.inverse import DifferentiableForward

        self._impl = DifferentiableForward(wrapped)
        self._impl.adjoint_solver = adjoint_solver
        self.last_forward_solution = None

    @property
    def model(self) -> Any:
        return self._impl.model

    @property
    def state(self) -> Any:
        return self._impl.state

    @property
    def last_adjoint_solution(self) -> Any:
        return self._impl.last_adjoint_solution

    def forward(self, materials: Any) -> torch.Tensor:
        return self._impl.forward(materials)

    def step(self) -> Any:
        solution = self._impl.step()
        self.last_forward_solution = solution
        return solution

    def adjoint_solve(self, u_grad: torch.Tensor) -> Any:
        return self._impl.adjoint_solve(u_grad)


def make_adjoint_solver() -> Any:
    from liblaf.peach.linalg import FallbackSolver
    from liblaf.peach.linalg.cupy import CupyCG, CupyMinRes

    return FallbackSolver(
        solvers=[
            CupyCG(
                maxiter=ADJOINT_MAXITER,
                rtol=ADJOINT_RTOL,
                atol=ADJOINT_ATOL,
            ),
            CupyMinRes(maxiter=ADJOINT_MAXITER, tol=ADJOINT_RTOL),
        ]
    )


def solve_case(  # noqa: PLR0915
    case: ToyCase, base_mesh: pv.UnstructuredGrid, cfg: InverseConfig
) -> dict[str, Any]:
    from liblaf.apple.common import GLOBAL_POINT_ID

    start = time.perf_counter()
    data_dir = cfg.output_summary.parent
    mesh = base_mesh.copy(deep=True)
    target = target_displacement(mesh, case.target_y)
    target_path = data_dir / f"{case.stem}-target.vtu"
    output_path = data_dir / f"{case.stem}.vtu"
    history_path = data_dir / f"{case.stem}-steps.vtkhdf"

    melon.save(make_target_mesh(mesh, target), target_path)

    inverse_mesh = mesh.copy(deep=True)
    forward, skin = build_forward(inverse_mesh, case)
    differentiable_forward = RecordingDifferentiableForward(
        forward, make_adjoint_solver()
    )
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
    smooth_edges = active_tetra_neighbor_edges(inverse_mesh, active_ids)

    edges = top_surface_edges(inverse_mesh)
    edge_i = global_ids[edges[:, 0]] if edges.size else np.empty(0, dtype=np.int64)
    edge_j = global_ids[edges[:, 1]] if edges.size else np.empty(0, dtype=np.int64)
    lap_ids = np.unique(np.concatenate((edge_i, edge_j))).astype(np.int64)

    target_t = torch.as_tensor(
        target, dtype=torch.get_default_dtype(), device=torch.get_default_device()
    )
    target_full_t = target_t[torch.as_tensor(global_ids, dtype=torch.long)]
    target_ids_t = torch.as_tensor(
        target_ids, dtype=torch.long, device=torch.get_default_device()
    )
    target_global_ids_t = torch.as_tensor(
        global_ids[target_ids], dtype=torch.long, device=torch.get_default_device()
    )
    active_ids_t = torch.as_tensor(
        active_ids, dtype=torch.long, device=torch.get_default_device()
    )
    edge_i_t = torch.as_tensor(
        edge_i, dtype=torch.long, device=torch.get_default_device()
    )
    edge_j_t = torch.as_tensor(
        edge_j, dtype=torch.long, device=torch.get_default_device()
    )
    lap_ids_t = torch.as_tensor(
        lap_ids, dtype=torch.long, device=torch.get_default_device()
    )
    smooth_i_t = torch.as_tensor(
        smooth_edges[:, 0], dtype=torch.long, device=torch.get_default_device()
    )
    smooth_j_t = torch.as_tensor(
        smooth_edges[:, 1], dtype=torch.long, device=torch.get_default_device()
    )

    activation_parameter = torch.nn.Parameter(
        torch.zeros(
            (1 if case.variant.shared_activation else active_ids.size, 6),
            dtype=torch.get_default_dtype(),
            device=torch.get_default_device(),
        )
    )
    optimizer = torch.optim.Adam(
        [activation_parameter],
        lr=cfg.inverse_lr,
    )

    best_step = 0
    best_loss = math.inf
    best_displacement: np.ndarray | None = None
    best_activation_inv: np.ndarray | None = None
    plateau_reference_loss = math.inf
    plateau_steps = 0
    trace: list[dict[str, Any]] = []
    history_frames = 0
    stop_reason = "step_limit"

    with melon.io.VTKHDFTemporalUnstructuredGridWriter(history_path) as history_writer:
        for step in range(cfg.inverse_max_steps + 1):
            step_start = time.perf_counter()
            optimizer.zero_grad()
            active_activation_inv = active_activation_inv_from_parameter(
                activation_parameter,
                n_active_tets=active_ids.size,
                shared=case.variant.shared_activation,
            )
            materials = material_tree(
                base_materials, active_activation_inv, active_ids_t, inverse_mesh.n_cells
            )
            forward_start = time.perf_counter()
            output = forward_quiet(differentiable_forward, materials)
            forward_elapsed = time.perf_counter() - forward_start
            residual = output[target_global_ids_t] - target_t[target_ids_t]
            data_loss = residual.square().mean()
            residual_full = output - target_full_t
            lap_loss = (
                residual_laplacian_loss(
                    residual_full=residual_full,
                    edge_i=edge_i_t,
                    edge_j=edge_j_t,
                    lap_ids=lap_ids_t,
                )
                if case.variant.name == "laplacian"
                else torch.zeros((), dtype=data_loss.dtype, device=data_loss.device)
            )
            smooth_loss = (
                activation_smooth_loss(active_activation_inv, smooth_i_t, smooth_j_t)
                if case.variant.activation_smooth
                else torch.zeros((), dtype=data_loss.dtype, device=data_loss.device)
            )
            loss = (
                data_loss
                + cfg.residual_laplacian_weight * lap_loss
                + cfg.activation_smooth_weight * smooth_loss
            )

            backward_start = time.perf_counter()
            loss.backward()
            backward_elapsed = time.perf_counter() - backward_start
            grad = activation_parameter.grad
            if grad is None:
                msg = "differentiable forward did not produce activation gradients"
                raise RuntimeError(msg)
            if not torch.isfinite(grad).all():
                nonfinite = int((~torch.isfinite(grad)).sum().detach().cpu())
                msg = f"non-finite inverse gradient at step {step}: {nonfinite} entries"
                raise FloatingPointError(msg)

            error_stats = point_error_stats(residual.detach())
            displacement = to_numpy(output)[global_ids]
            full_activation_inv = to_numpy(
                full_activation_inv_from_active(
                    active_activation_inv.detach(), active_ids_t, inverse_mesh.n_cells
                )
            )
            loss_value = float(loss.detach().cpu())
            if loss_value < best_loss:
                best_step = step
                best_loss = loss_value
                best_displacement = displacement
                best_activation_inv = full_activation_inv
            if (
                math.isinf(plateau_reference_loss)
                or loss_value <= plateau_reference_loss - cfg.inverse_loss_min_delta
            ):
                plateau_reference_loss = loss_value
                plateau_steps = 0
            else:
                plateau_steps += 1

            forward_metrics = forward_solution_metrics(
                differentiable_forward.last_forward_solution
            )
            adjoint_metrics = adjoint_solution_metrics(
                differentiable_forward.last_adjoint_solution
            )
            row: dict[str, Any] = {
                "step": float(step),
                "loss/total": loss_value,
                "loss/data": float(data_loss.detach().cpu()),
                "loss/residual_laplacian": float(lap_loss.detach().cpu()),
                "loss/activation_smooth": float(smooth_loss.detach().cpu()),
                "target/error_mean": float(error_stats["mean"].detach().cpu()),
                "target/error_rms": float(error_stats["rms"].detach().cpu()),
                "target/error_max": float(error_stats["max"].detach().cpu()),
                "activation_inv/rms": float(
                    torch.linalg.vector_norm(active_activation_inv.detach()).cpu()
                    / math.sqrt(active_activation_inv.numel())
                ),
                "activation_inv/max_abs": float(
                    active_activation_inv.detach().abs().max().cpu()
                ),
                "grad/norm": float(torch.linalg.vector_norm(grad).detach().cpu()),
                "best/step": float(best_step),
                "best/loss": float(best_loss),
                "plateau/steps": float(plateau_steps),
                "time/forward_s": forward_elapsed,
                "time/backward_s": backward_elapsed,
                "time/step_s": time.perf_counter() - step_start,
                **forward_metrics,
                **adjoint_metrics,
            }
            trace.append(row)
            cherries.set_step(len(trace) - 1)
            cherries.log_metrics(
                {
                    f"{case.stem}/loss": row["loss/total"],
                    f"{case.stem}/data_loss": row["loss/data"],
                    f"{case.stem}/laplacian_loss": row["loss/residual_laplacian"],
                    f"{case.stem}/activation_smooth_loss": row[
                        "loss/activation_smooth"
                    ],
                    f"{case.stem}/error_rms": row["target/error_rms"],
                    f"{case.stem}/error_max": row["target/error_max"],
                    f"{case.stem}/activation_inv_rms": row["activation_inv/rms"],
                }
            )
            logger.info(
                "%s step %03d loss %.6g data %.6g lap %.6g act %.6g rms %.6g grad %.6g",
                case.stem,
                step,
                row["loss/total"],
                row["loss/data"],
                row["loss/residual_laplacian"],
                row["loss/activation_smooth"],
                row["target/error_rms"],
                row["grad/norm"],
            )

            history_start = time.perf_counter()
            step_mesh = make_result_mesh(
                inverse_mesh,
                target,
                displacement,
                full_activation_inv,
                {
                    "inverse/step": step,
                    "inverse/loss": row["loss/total"],
                    "inverse/activation_smooth_loss": row["loss/activation_smooth"],
                    "inverse/error_rms": row["target/error_rms"],
                },
            )
            history_writer.append(make_history_mesh(step_mesh), time=float(step))
            history_frames += 1
            row["time/history_s"] = time.perf_counter() - history_start

            if plateau_steps >= INVERSE_PATIENCE:
                stop_reason = f"loss_plateau_{INVERSE_PATIENCE}_steps"
                break
            if step == cfg.inverse_max_steps:
                break
            optimizer.step()

    if best_displacement is None or best_activation_inv is None:
        msg = f"{case.stem} did not evaluate any inverse state"
        raise RuntimeError(msg)

    elapsed_s = time.perf_counter() - start
    target_error = best_displacement[target_ids] - target[target_ids]
    target_norm = np.linalg.norm(target[target_ids], axis=1)
    target_error_norm = np.linalg.norm(target_error, axis=1)
    active_activation_inv_np = best_activation_inv[active_ids]
    initial = trace[0]
    final = trace[-1]
    converged = stop_reason.startswith("loss_plateau")
    summary: dict[str, Any] = {
        "case": case.stem,
        "input_mesh": str(cfg.input_mesh),
        "mode": case.mode,
        "loss_variant": case.variant.name,
        "skin/prestrain_enabled": bool(case.variant.skin_prestrain),
        "skin/prestrain": float(
            SKIN_PRESTRAIN if case.variant.skin_prestrain else 0.0
        ),
        "skin/n_triangles": int(skin.n_cells),
        "resolution": case.resolution.name,
        "tetwild/lr": float(case.resolution.lr),
        "tetwild/lr_interpretation": "relative_edge_length_fac",
        "target_y": float(case.target_y),
        "n_points": int(inverse_mesh.n_points),
        "n_tets": int(inverse_mesh.n_cells),
        "n_active_tets": int(active_ids.size),
        "n_activation_parameters": int(1 if case.variant.shared_activation else active_ids.size),
        "n_activation_parameter_dofs": int(activation_parameter.numel()),
        "n_target_points": int(target_ids.size),
        "n_top_laplacian_edges": int(edge_i.size),
        "n_activation_smooth_edges": int(smooth_edges.shape[0]),
        "elapsed_s": float(elapsed_s),
        "inverse/max_steps": int(cfg.inverse_max_steps),
        "inverse/lr": float(cfg.inverse_lr),
        "inverse/patience": int(INVERSE_PATIENCE),
        "inverse/loss_min_delta": float(cfg.inverse_loss_min_delta),
        "inverse/stop_reason": stop_reason,
        "inverse/converged": bool(converged),
        "inverse/evaluations": len(trace),
        "history/format": "VTKHDFTemporalUnstructuredGrid",
        "history/path": history_path.name,
        "history/frames": int(history_frames),
        "initial/loss": float(initial["loss/total"]),
        "initial/error_rms": float(initial["target/error_rms"]),
        "best/step": int(best_step),
        "best/loss": float(best_loss),
        "best/error_mean": float(target_error_norm.mean()),
        "best/error_rms": float(
            np.linalg.norm(target_error) / math.sqrt(target_ids.size)
        ),
        "best/error_max": float(target_error_norm.max()),
        "best/error_rms_fraction_of_target": float(
            np.linalg.norm(target_error) / np.linalg.norm(target[target_ids])
        )
        if np.linalg.norm(target[target_ids]) > 0.0
        else math.nan,
        "final/step": float(final["step"]),
        "final/loss": float(final["loss/total"]),
        "final/error_rms": float(final["target/error_rms"]),
        "final/plateau_steps": int(final["plateau/steps"]),
        "target/displacement_rms": float(
            np.linalg.norm(target[target_ids]) / math.sqrt(target_ids.size)
        ),
        "target/displacement_max": float(target_norm.max()),
        "activation/mode": case.variant.activation_mode,
        "activation/shared": bool(case.variant.shared_activation),
        "activation_inv/rms": float(
            np.linalg.norm(active_activation_inv_np)
            / math.sqrt(max(1, active_activation_inv_np.size))
        ),
        "activation_inv/max_abs": float(np.abs(active_activation_inv_np).max()),
        "fat/E_MPa": float(FAT_E),
        "fat/nu": float(FAT_NU),
        "muscle/E_MPa": float(MUSCLE_E),
        "muscle/nu": float(MUSCLE_NU),
        "aponeurosis/E_MPa": float(APONEUROSIS_E),
        "aponeurosis/nu": float(APONEUROSIS_NU),
        "skin/E_MPa": float(SKIN_E),
        "skin/nu": float(SKIN_NU),
        "skin/thickness": float(SKIN_THICKNESS),
        "loss/residual_laplacian_enabled": case.variant.name == "laplacian",
        "loss/residual_laplacian_weight": float(cfg.residual_laplacian_weight),
        "loss/activation_smooth_enabled": bool(case.variant.activation_smooth),
        "loss/activation_smooth_weight": float(cfg.activation_smooth_weight),
        "trace": trace,
        **geometry_summary(inverse_mesh),
        **bumpiness_metrics(inverse_mesh, best_displacement, target),
        **{
            f"last/{key}": value
            for key, value in forward_solution_metrics(
                differentiable_forward.last_forward_solution
            ).items()
        },
        **{
            f"last/{key}": value
            for key, value in adjoint_solution_metrics(
                differentiable_forward.last_adjoint_solution
            ).items()
        },
    }
    result_metrics = {
        key: value
        for key, value in summary.items()
        if isinstance(value, int | float | bool)
    }
    result = make_result_mesh(
        inverse_mesh,
        target,
        best_displacement,
        best_activation_inv,
        result_metrics,
    )
    melon.save(result, output_path)
    case_summary_path = data_dir / f"{case.stem}-summary.json"
    case_summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    cherries.log_output(target_path)
    cherries.log_output(output_path)
    cherries.log_output(case_summary_path)
    cherries.log_output(history_path)
    return summary


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
        "| case | lr | mode | residual lap | activation mode | skin prestrain | tets | active | params | target pts | stop | best step | best loss | error RMS | error/target | top y std | residual edge RMS | residual lap RMS |",
        "| --- | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        (
            "| "
            + " | ".join(
                [
                    str(row["case"]),
                    format_float(row["tetwild/lr"]),
                    str(row["mode"]),
                    format_float(row["loss/residual_laplacian_enabled"]),
                    str(row["activation/mode"]),
                    format_float(row["skin/prestrain_enabled"]),
                    str(row["n_tets"]),
                    str(row["n_active_tets"]),
                    str(row["n_activation_parameters"]),
                    str(row["n_target_points"]),
                    str(row["inverse/stop_reason"]),
                    format_float(row["best/step"]),
                    format_float(row["best/loss"]),
                    format_float(row["best/error_rms"]),
                    format_float(row["best/error_rms_fraction_of_target"]),
                    format_float(row["bumpiness/top_y_std"]),
                    format_float(row["bumpiness/residual_edge_rms"]),
                    format_float(row["bumpiness/residual_laplacian_rms"]),
                ]
            )
            + " |"
        )
        for row in rows
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_mesh(cfg: PrepareConfig) -> None:
    resolution = ResolutionSpec(name=label_lr(cfg.tetwild_lr), lr=cfg.tetwild_lr)
    mesh = make_tetwild_mesh(resolution)
    add_material_and_boundary_fields(mesh, resolution)
    cfg.output_mesh.parent.mkdir(parents=True, exist_ok=True)
    melon.save(mesh, cfg.output_mesh)

    target_mask = np.asarray(mesh.point_data[TARGET_SURFACE_MASK], dtype=bool)
    active_mask = np.asarray(mesh.cell_data["ActivationMask"], dtype=bool)
    summary: dict[str, Any] = {
        "mesh": cfg.output_mesh.name,
        "resolution": resolution.name,
        "tetwild/lr": float(resolution.lr),
        "tetwild/lr_interpretation": "relative_edge_length_fac",
        "n_points": int(mesh.n_points),
        "n_tets": int(mesh.n_cells),
        "n_active_tets": int(active_mask.sum()),
        "n_target_points": int(target_mask.sum()),
        "target/stretch_y": float(STRETCH_TARGET_MAGNITUDE),
        "target/squash_y": float(-SQUASH_TARGET_MAGNITUDE),
        "fat/E_MPa": float(FAT_E),
        "fat/nu": float(FAT_NU),
        "muscle/E_MPa": float(MUSCLE_E),
        "muscle/nu": float(MUSCLE_NU),
        "aponeurosis/E_MPa": float(APONEUROSIS_E),
        "aponeurosis/nu": float(APONEUROSIS_NU),
        "skin/E_MPa": float(SKIN_E),
        "skin/nu": float(SKIN_NU),
        "skin/prestrain": float(SKIN_PRESTRAIN),
        **geometry_summary(mesh),
    }
    cfg.output_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    cherries.log_output(cfg.output_mesh)
    cherries.log_output(cfg.output_summary)
    logger.info("Wrote %s", cfg.output_mesh)
    logger.info("Wrote %s", cfg.output_summary)


def load_existing_case_summaries(cfg: InverseConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(
            cfg.output_summary.parent.glob("20-toy-tetwild-*-summary.json")
        )
    ]
    if not rows:
        msg = f"no per-case summaries found under {cfg.output_summary.parent}"
        raise FileNotFoundError(msg)
    return rows


def summarize_existing_cases(
    rows: list[dict[str, Any]],
    resolution: ResolutionSpec,
    mode: Literal["stretch", "squash"],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    expected_stems = [case.stem for case in expected_inverse_cases(resolution, mode)]
    rows_by_case = {str(row["case"]): row for row in rows}
    ordered_rows = [rows_by_case[stem] for stem in expected_stems if stem in rows_by_case]
    extra_stems = sorted(set(rows_by_case) - set(expected_stems))
    ordered_rows.extend(rows_by_case[stem] for stem in extra_stems)
    missing_stems = [stem for stem in expected_stems if stem not in rows_by_case]
    converged = [
        bool(row.get("inverse/converged", False))
        for row in ordered_rows
        if str(row["case"]) in expected_stems
    ]
    summary = {
        "complete": not missing_stems and not extra_stems and all(converged),
        "cases": ordered_rows,
        "expected/cases": expected_stems,
        "expected/n_cases": len(expected_stems),
        "extra/cases": extra_stems,
        "extra/n_cases": len(extra_stems),
        "missing/cases": missing_stems,
        "missing/n_cases": len(missing_stems),
        "mode": mode,
        "observed/n_cases": len(ordered_rows),
        "observed/n_expected_cases": len(ordered_rows) - len(extra_stems),
    }
    return summary, ordered_rows


def run_inverse(cfg: InverseConfig) -> None:
    cfg.output_summary.parent.mkdir(parents=True, exist_ok=True)
    if cfg.compare_existing:
        rows = load_existing_case_summaries(cfg)
        mesh = pv.read(cfg.input_mesh)
        if not isinstance(mesh, pv.UnstructuredGrid):
            mesh = mesh.cast_to_unstructured_grid()
        resolution = mesh_resolution(mesh)
        mode = selected_mode(cfg)
        summary, rows = summarize_existing_cases(rows, resolution, mode)
        cfg.output_summary.write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )
        write_table(cfg.output_table, rows)
        cherries.log_output(cfg.output_summary)
        cherries.log_output(cfg.output_table)
        logger.info(
            "Compared %d existing %s case summaries; %d of %d expected cases missing",
            len(rows),
            mode,
            summary["missing/n_cases"],
            summary["expected/n_cases"],
        )
        return

    configure_runtime()
    mesh = pv.read(cfg.input_mesh)
    if not isinstance(mesh, pv.UnstructuredGrid):
        mesh = mesh.cast_to_unstructured_grid()
    resolution = mesh_resolution(mesh)
    case = selected_case(cfg, resolution)
    rows = [solve_case(case, mesh, cfg)]

    cfg.output_summary.write_text(
        json.dumps({"cases": rows}, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_table(cfg.output_table, rows)
    cherries.log_output(cfg.output_summary)
    cherries.log_output(cfg.output_table)
    logger.info("Wrote %s", cfg.output_summary)
    logger.info("Wrote %s", cfg.output_table)

    failed = [row["case"] for row in rows if not row["inverse/converged"]]
    if failed and cfg.require_convergence:
        msg = "inverse cases did not hit the 20-step loss plateau: " + ", ".join(failed)
        raise RuntimeError(msg)
