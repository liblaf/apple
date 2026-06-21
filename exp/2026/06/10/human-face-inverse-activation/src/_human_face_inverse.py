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

SOURCE_MESH = Path(
    "/home/liblaf/github/liblaf/melon/exp/2026/05/27/head/data/62-tetmesh-3191k.vtu"
)

APONEUROSIS_FRACTION = "AponeurosisFraction"
FAT_FRACTION = "FatFraction"
MUSCLE_FRACTION = "MuscleFraction"
FRACTION_SUM = "FractionSum"
ACTIVE_FRACTION = "ActiveFraction"
TARGET_FINITE = "TargetFinite"
SMILE_LOSS_MASK = "SmileLossMask"
TOP_SURFACE_MASK = "TopSurfaceMask"
SMILE_TARGET = "Smile"
IS_FACE = "IsFace"
IS_FIXED = "IsFixed"

FAT_E = 0.003
FAT_NU = 0.49
MUSCLE_E = 0.030
MUSCLE_NU = 0.49
APONEUROSIS_E = 0.10
APONEUROSIS_NU = 0.35
SKIN_E = 0.20
SKIN_NU = 0.49
SKIN_THICKNESS = 0.001
SKIN_PRESTRAIN = 0.0
ACTIVE_FRACTION_TOL = 1.0e-6

FORWARD_RTOL = 5.0e-4
FORWARD_ATOL = 1.0e-10
FORWARD_MAX_STEPS = 5000
ADJOINT_RTOL = 5.0e-4
ADJOINT_ATOL = 0.0
ADJOINT_MAXITER = 10_000

INVERSE_PATIENCE = 20


@dataclass(frozen=True)
class InverseCase:
    target: Literal["smile", "top-y"]
    lr: float
    label: str = ""

    @property
    def stem(self) -> str:
        stem = f"20-human-face-{self.target}-{label_lr(self.lr)}"
        if self.label:
            stem = f"{stem}-{self.label}"
        return stem


class PrepareConfig(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    input_mesh: Path = cherries.input(SOURCE_MESH)
    output_mesh: Path = cherries.output("10-human-face-prepared.vtu", mkdir=True)
    output_summary: Path = cherries.output(
        "10-human-face-prepared-summary.json", mkdir=True
    )


class InverseConfig(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    input_mesh: Path = cherries.input("10-human-face-prepared.vtu")
    output_summary: Path = cherries.output("20-inverse-summary.json", mkdir=True)
    output_table: Path = cherries.output("20-inverse-table.md", mkdir=True)

    target: str = "smile"
    case_label: str = ""
    initial_activation_mesh: Path | None = None
    inverse_lr: float = 0.03
    inverse_max_steps: int = 300
    inverse_loss_min_delta: float = 1.0e-9
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


def selected_cases(cfg: InverseConfig) -> list[InverseCase]:
    targets = ["smile"] if cfg.target == "all" else [cfg.target]
    invalid = sorted(set(targets) - {"smile"})
    if invalid:
        msg = f"unknown target(s) {invalid}; expected smile or all"
        raise ValueError(msg)
    return [
        InverseCase(target=target, lr=cfg.inverse_lr, label=cfg.case_label)  # pyright: ignore[reportArgumentType]
        for target in targets
    ]


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


def orient_tetra_mesh(mesh: pv.UnstructuredGrid) -> tuple[pv.UnstructuredGrid, int]:
    tets = tetra_cells(mesh).copy()
    points = np.asarray(mesh.points, dtype=np.float64)
    signed = tetra_signed_volumes(points, tets)
    flipped = signed < 0.0
    n_flipped = int(flipped.sum())
    if n_flipped:
        tets[flipped, 2], tets[flipped, 3] = tets[flipped, 3], tets[flipped, 2].copy()
    cells = np.empty((tets.shape[0], 5), dtype=np.int64)
    cells[:, 0] = 4
    cells[:, 1:] = tets
    cell_types = np.full(tets.shape[0], int(pv.CellType.TETRA), dtype=np.uint8)
    oriented = pv.UnstructuredGrid(cells.ravel(), cell_types, points)
    for name, values in mesh.point_data.items():
        oriented.point_data[name] = np.asarray(values)
    for name, values in mesh.cell_data.items():
        oriented.cell_data[name] = np.asarray(values)
    for name, values in mesh.field_data.items():
        oriented.field_data[name] = np.asarray(values)
    return oriented, n_flipped


def lame_parameters(E: float, nu: float) -> tuple[float, float]:
    lambda_ = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    return float(lambda_), float(mu)


def finite_vec3(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        msg = f"expected vec3 data, got shape {values.shape}"
        raise ValueError(msg)
    return np.isfinite(values).all(axis=1)


def surface_original_ids(mesh: pv.UnstructuredGrid) -> tuple[pv.PolyData, np.ndarray]:
    surface = mesh.extract_surface(algorithm=None).triangulate()
    if "vtkOriginalPointIds" not in surface.point_data:
        msg = "extract_surface did not produce vtkOriginalPointIds"
        raise KeyError(msg)
    original_ids = np.asarray(surface.point_data["vtkOriginalPointIds"], dtype=np.int64)
    return surface, original_ids


def add_required_fields(mesh: pv.UnstructuredGrid) -> dict[str, Any]:
    from liblaf.apple.common import ACTIVATION, ACTIVATION_INV, FIXED_MASK, FIXED_VALUE

    for name in (APONEUROSIS_FRACTION, FAT_FRACTION, MUSCLE_FRACTION):
        if name not in mesh.cell_data:
            msg = f"missing required cell data {name!r}"
            raise KeyError(msg)
        mesh.cell_data[name] = np.nan_to_num(
            np.asarray(mesh.cell_data[name], dtype=np.float64), nan=0.0
        )

    points = np.asarray(mesh.points, dtype=np.float64)
    tets = tetra_cells(mesh)
    volumes = tetra_volumes(points, tets)
    fractions = np.column_stack(
        (
            np.asarray(mesh.cell_data[APONEUROSIS_FRACTION], dtype=np.float64),
            np.asarray(mesh.cell_data[FAT_FRACTION], dtype=np.float64),
            np.asarray(mesh.cell_data[MUSCLE_FRACTION], dtype=np.float64),
        )
    )
    fraction_sum = fractions.sum(axis=1)
    active = fractions[:, 2] > ACTIVE_FRACTION_TOL

    mesh.cell_data["Volume"] = volumes
    mesh.cell_data[FRACTION_SUM] = fraction_sum
    mesh.cell_data[ACTIVE_FRACTION] = fractions[:, 2]
    mesh.cell_data["ActivationMask"] = active.astype(np.int8)
    mesh.cell_data[ACTIVATION.vtk] = np.zeros((mesh.n_cells, 6), dtype=np.float64)
    mesh.cell_data[ACTIVATION_INV.vtk] = np.zeros((mesh.n_cells, 6), dtype=np.float64)

    if IS_FIXED not in mesh.point_data:
        msg = f"missing required point data {IS_FIXED!r}"
        raise KeyError(msg)
    fixed = np.asarray(mesh.point_data[IS_FIXED], dtype=bool)
    mesh.point_data[FIXED_MASK.vtk] = np.repeat(fixed[:, None], 3, axis=1)
    mesh.point_data[FIXED_VALUE.vtk] = np.zeros((mesh.n_points, 3), dtype=np.float64)

    if SMILE_TARGET not in mesh.point_data:
        msg = f"missing required point data {SMILE_TARGET!r}"
        raise KeyError(msg)
    if IS_FACE not in mesh.point_data:
        msg = f"missing required point data {IS_FACE!r}"
        raise KeyError(msg)
    smile = np.asarray(mesh.point_data[SMILE_TARGET], dtype=np.float64)
    finite = finite_vec3(smile)
    face = np.asarray(mesh.point_data[IS_FACE], dtype=bool)
    smile_loss = face & finite
    mesh.point_data[TARGET_FINITE] = finite.astype(np.int8)
    mesh.point_data[SMILE_LOSS_MASK] = smile_loss.astype(np.int8)

    surface, original_ids = surface_original_ids(mesh)

    return {
        "n_points": int(mesh.n_points),
        "n_tets": int(mesh.n_cells),
        "n_surface_points": int(np.unique(original_ids).size),
        "n_surface_triangles": int(surface.n_cells),
        "n_fixed_points": int(fixed.sum()),
        "n_smile_finite_points": int(finite.sum()),
        "n_smile_nan_points": int((~finite).sum()),
        "n_smile_loss_points": int(smile_loss.sum()),
        "n_active_tets": int(active.sum()),
    }


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


def skin_surface(mesh: pv.UnstructuredGrid) -> pv.PolyData:
    from liblaf.apple.common import (
        ACTIVATION_INV,
        FRACTION,
        GLOBAL_POINT_ID,
        LAMBDA,
        MU,
    )

    surface, original_ids = surface_original_ids(mesh)
    surface.point_data[GLOBAL_POINT_ID.vtk] = np.asarray(
        mesh.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64
    )[original_ids]
    lambda_, mu = lame_parameters(SKIN_E, SKIN_NU)
    surface.cell_data[LAMBDA.vtk] = np.full(surface.n_cells, lambda_, dtype=np.float64)
    surface.cell_data[MU.vtk] = np.full(surface.n_cells, mu, dtype=np.float64)
    surface.cell_data[FRACTION.vtk] = np.ones(surface.n_cells, dtype=np.float64)
    surface.cell_data[ACTIVATION_INV.vtk] = np.zeros(
        (surface.n_cells, 3), dtype=np.float64
    )
    surface.cell_data["SkinPrestrain"] = np.full(
        surface.n_cells, SKIN_PRESTRAIN, dtype=np.float64
    )
    return surface


def build_forward(mesh: pv.UnstructuredGrid):
    from liblaf.apple.forward import Forward, ModelBuilder
    from liblaf.apple.warp.fem import Koiter, StableNeoHookean, StableNeoHookeanActive

    builder = ModelBuilder()
    builder.add_vertices(mesh)
    builder.add_fixed(mesh)

    set_volume_material(
        mesh,
        E=APONEUROSIS_E,
        nu=APONEUROSIS_NU,
        fraction=np.asarray(mesh.cell_data[APONEUROSIS_FRACTION], dtype=np.float64),
    )
    builder.add_potential(StableNeoHookean.from_pyvista(mesh, name="aponeurosis"))

    set_volume_material(
        mesh,
        E=FAT_E,
        nu=FAT_NU,
        fraction=np.asarray(mesh.cell_data[FAT_FRACTION], dtype=np.float64),
    )
    builder.add_potential(StableNeoHookean.from_pyvista(mesh, name="fat"))

    set_volume_material(
        mesh,
        E=MUSCLE_E,
        nu=MUSCLE_NU,
        fraction=np.asarray(mesh.cell_data[MUSCLE_FRACTION], dtype=np.float64),
    )
    builder.add_potential(StableNeoHookeanActive.from_pyvista(mesh, name="muscle"))

    skin = skin_surface(mesh)
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


def initial_active_activation_inv(
    cfg: InverseConfig, active_ids: np.ndarray, n_cells: int
) -> np.ndarray:
    if cfg.initial_activation_mesh is None:
        return np.zeros((active_ids.size, 6), dtype=np.float64)

    from liblaf.apple.common import ACTIVATION_INV

    mesh = pv.read(cfg.initial_activation_mesh)
    activation_inv = np.asarray(mesh.cell_data[ACTIVATION_INV.vtk], dtype=np.float64)
    expected = (n_cells, 6)
    if activation_inv.shape != expected:
        msg = (
            f"{cfg.initial_activation_mesh} {ACTIVATION_INV.vtk} must have shape "
            f"{expected}, got {activation_inv.shape}"
        )
        raise ValueError(msg)
    return activation_inv[active_ids].copy()


def initial_forward_displacement(
    cfg: InverseConfig, n_points: int
) -> np.ndarray | None:
    if cfg.initial_activation_mesh is None:
        return None

    mesh = pv.read(cfg.initial_activation_mesh)
    if "Displacement" not in mesh.point_data:
        return None
    displacement = np.asarray(mesh.point_data["Displacement"], dtype=np.float64)
    expected = (n_points, 3)
    if displacement.shape != expected:
        msg = (
            f"{cfg.initial_activation_mesh} Displacement must have shape "
            f"{expected}, got {displacement.shape}"
        )
        raise ValueError(msg)
    return displacement.copy()


def target_displacement_and_mask(
    mesh: pv.UnstructuredGrid, case: InverseCase
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if case.target == "smile":
        target = np.nan_to_num(
            np.asarray(mesh.point_data[SMILE_TARGET], dtype=np.float64),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        mask = np.asarray(mesh.point_data[SMILE_LOSS_MASK], dtype=bool)
    else:
        msg = f"unknown target {case.target!r}"
        raise ValueError(msg)
    if not np.any(mask):
        msg = f"{case.target} selected no loss points"
        raise ValueError(msg)
    fixed = np.asarray(mesh.point_data[IS_FIXED], dtype=bool)
    target_norm = np.linalg.norm(target[mask], axis=1)
    return (
        target,
        mask,
        {
            "target/name": case.target,
            "target/loss_points": int(mask.sum()),
            "target/fixed_overlap_points": int((mask & fixed).sum()),
            "target/displacement_rms": float(
                np.linalg.norm(target[mask]) / math.sqrt(mask.sum())
            ),
            "target/displacement_max": float(target_norm.max()),
        },
    )


def make_target_mesh(
    mesh: pv.UnstructuredGrid,
    target: np.ndarray,
    mask: np.ndarray,
) -> pv.UnstructuredGrid:
    result = mesh.copy(deep=True)
    result.point_data["TargetDisplacement"] = target
    result.point_data["LossMask"] = mask.astype(np.int8)
    result.point_data["TargetPoint"] = result.points + target
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
    mask: np.ndarray,
    displacement: np.ndarray,
    activation_inv: np.ndarray,
    metrics: dict[str, float | int | bool | str],
) -> pv.UnstructuredGrid:
    from liblaf.apple.common import ACTIVATION_INV

    result = mesh.copy(deep=True)
    error = displacement - target
    result.point_data["Displacement"] = displacement
    result.point_data["TargetDisplacement"] = target
    result.point_data["LossMask"] = mask.astype(np.int8)
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
    surface, original_ids = surface_original_ids(mesh)
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


def surface_edges_for_mask(mesh: pv.UnstructuredGrid, mask: np.ndarray) -> np.ndarray:
    triangles = surface_triangles(mesh)
    selected = triangles[np.all(mask[triangles], axis=1)]
    if selected.size == 0:
        edges = unique_edges(triangles)
        edges = edges[np.all(mask[edges], axis=1)]
        return edges.astype(np.int64)
    return unique_edges(selected).astype(np.int64)


def bumpiness_metrics(
    *,
    mask: np.ndarray,
    edges: np.ndarray,
    displacement: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    ids = np.flatnonzero(mask)
    residual = displacement - target
    if edges.size == 0:
        return {
            "bumpiness/displacement_edge_rms": math.nan,
            "bumpiness/residual_edge_rms": math.nan,
            "bumpiness/displacement_laplacian_rms": math.nan,
            "bumpiness/residual_laplacian_rms": math.nan,
            "bumpiness/displacement_norm_std": math.nan,
            "bumpiness/residual_norm_std": math.nan,
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
        "bumpiness/displacement_edge_rms": float(
            np.linalg.norm(disp_edge) / math.sqrt(edges.shape[0])
        ),
        "bumpiness/residual_edge_rms": float(
            np.linalg.norm(residual_edge) / math.sqrt(edges.shape[0])
        ),
        "bumpiness/displacement_laplacian_rms": float(
            np.linalg.norm(disp_lap[ids]) / math.sqrt(ids.size)
        ),
        "bumpiness/residual_laplacian_rms": float(
            np.linalg.norm(residual_lap[ids]) / math.sqrt(ids.size)
        ),
        "bumpiness/displacement_norm_std": float(
            np.linalg.norm(displacement[ids], axis=1).std()
        ),
        "bumpiness/residual_norm_std": float(
            np.linalg.norm(residual[ids], axis=1).std()
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


class RecordingDifferentiableForward:
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
        output = self._impl.forward(materials)
        self.last_forward_solution = self._impl.last_solution
        return output

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
            CupyCG(maxiter=ADJOINT_MAXITER, rtol=ADJOINT_RTOL, atol=ADJOINT_ATOL),
            CupyMinRes(maxiter=ADJOINT_MAXITER, tol=ADJOINT_RTOL),
        ]
    )


def remove_stale_case_outputs(data_dir: Path, stem: str) -> None:
    for suffix in ("-target.vtu", ".vtu", "-summary.json", "-steps.vtkhdf"):
        path = data_dir / f"{stem}{suffix}"
        if path.exists():
            path.unlink()


def solve_case(  # noqa: C901, PLR0915
    case: InverseCase, base_mesh: pv.UnstructuredGrid, cfg: InverseConfig
) -> dict[str, Any]:
    from liblaf.apple.common import GLOBAL_POINT_ID

    start = time.perf_counter()
    data_dir = cfg.output_summary.parent
    remove_stale_case_outputs(data_dir, case.stem)

    mesh = base_mesh.copy(deep=True)
    target, loss_mask, target_metrics = target_displacement_and_mask(mesh, case)
    target_path = data_dir / f"{case.stem}-target.vtu"
    output_path = data_dir / f"{case.stem}.vtu"
    history_path = data_dir / f"{case.stem}-steps.vtkhdf"
    melon.save(make_target_mesh(mesh, target, loss_mask), target_path)

    inverse_mesh = mesh.copy(deep=True)
    forward, skin = build_forward(inverse_mesh)
    initial_displacement = initial_forward_displacement(cfg, inverse_mesh.n_points)
    if initial_displacement is not None:
        initial_displacement_t = torch.as_tensor(
            initial_displacement,
            dtype=torch.get_default_dtype(),
            device=torch.get_default_device(),
        )
        forward.model.update(forward.state, initial_displacement_t)
    differentiable_forward = RecordingDifferentiableForward(
        forward, make_adjoint_solver()
    )
    base_materials = forward.model.get_materials()

    global_ids = np.asarray(
        inverse_mesh.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64
    )
    target_ids = np.flatnonzero(loss_mask).astype(np.int64)
    active_ids = np.flatnonzero(
        np.asarray(inverse_mesh.cell_data["ActivationMask"], dtype=bool)
    ).astype(np.int64)
    if active_ids.size == 0:
        msg = f"{case.stem} has no active muscle tetrahedra"
        raise ValueError(msg)

    bump_edges = surface_edges_for_mask(inverse_mesh, loss_mask)
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

    initial_activation = initial_active_activation_inv(
        cfg, active_ids, inverse_mesh.n_cells
    )
    activation_parameter = torch.nn.Parameter(
        torch.as_tensor(
            initial_activation,
            dtype=torch.get_default_dtype(),
            device=torch.get_default_device(),
        )
    )
    optimizer = torch.optim.Adam([activation_parameter], lr=cfg.inverse_lr)

    best_step = 0
    best_loss = math.inf
    best_displacement: np.ndarray | None = None
    best_activation_inv: np.ndarray | None = None
    plateau_steps = 0
    trace: list[dict[str, Any]] = []
    history_frames = 0
    stop_reason = "step_limit"

    with melon.io.VTKHDFTemporalUnstructuredGridWriter(history_path) as history_writer:
        for step in range(cfg.inverse_max_steps + 1):
            step_start = time.perf_counter()
            optimizer.zero_grad()
            materials = material_tree(
                base_materials, activation_parameter, active_ids_t, inverse_mesh.n_cells
            )
            forward_start = time.perf_counter()
            output = forward_quiet(differentiable_forward, materials)
            forward_elapsed = time.perf_counter() - forward_start
            residual = output[target_global_ids_t] - target_t[target_ids_t]
            loss = residual.square().mean()

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
                    activation_parameter.detach(), active_ids_t, inverse_mesh.n_cells
                )
            )
            loss_value = float(loss.detach().cpu())
            prev_best_loss = best_loss
            if loss_value < best_loss:
                best_step = step
                best_loss = loss_value
                best_displacement = displacement
                best_activation_inv = full_activation_inv
            if math.isinf(prev_best_loss) or loss_value <= (
                prev_best_loss - cfg.inverse_loss_min_delta
            ):
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
                "loss/data": loss_value,
                "target/error_mean": float(error_stats["mean"].detach().cpu()),
                "target/error_rms": float(error_stats["rms"].detach().cpu()),
                "target/error_max": float(error_stats["max"].detach().cpu()),
                "activation_inv/rms": float(
                    torch.linalg.vector_norm(activation_parameter.detach()).cpu()
                    / math.sqrt(activation_parameter.numel())
                ),
                "activation_inv/max_abs": float(
                    activation_parameter.detach().abs().max().cpu()
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
                    f"{case.stem}/error_rms": row["target/error_rms"],
                    f"{case.stem}/error_max": row["target/error_max"],
                    f"{case.stem}/activation_inv_rms": row["activation_inv/rms"],
                    f"{case.stem}/grad_norm": row["grad/norm"],
                }
            )
            logger.info(
                "%s step %03d loss %.6g rms %.6g grad %.6g best %.6g plateau %d",
                case.stem,
                step,
                row["loss/total"],
                row["target/error_rms"],
                row["grad/norm"],
                row["best/loss"],
                int(row["plateau/steps"]),
            )

            history_start = time.perf_counter()
            step_mesh = make_result_mesh(
                inverse_mesh,
                target,
                loss_mask,
                displacement,
                full_activation_inv,
                {
                    "inverse/step": step,
                    "inverse/loss": row["loss/total"],
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
    target_norm = np.linalg.norm(target[target_ids])
    target_error_norm = np.linalg.norm(target_error, axis=1)
    active_activation_inv_np = best_activation_inv[active_ids]
    initial = trace[0]
    final = trace[-1]
    converged = stop_reason.startswith("loss_plateau")
    summary: dict[str, Any] = {
        "case": case.stem,
        "input_mesh": str(cfg.input_mesh),
        "initial_activation_mesh": None
        if cfg.initial_activation_mesh is None
        else str(cfg.initial_activation_mesh),
        "target/name": case.target,
        "case/label": case.label,
        "n_points": int(inverse_mesh.n_points),
        "n_tets": int(inverse_mesh.n_cells),
        "n_active_tets": int(active_ids.size),
        "n_activation_parameters": int(active_ids.size),
        "n_activation_parameter_dofs": int(activation_parameter.numel()),
        "n_skin_triangles": int(skin.n_cells),
        "n_bumpiness_edges": int(bump_edges.shape[0]),
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
            np.linalg.norm(target_error) / target_norm
        )
        if target_norm > 0.0
        else math.nan,
        "final/step": float(final["step"]),
        "final/loss": float(final["loss/total"]),
        "final/error_rms": float(final["target/error_rms"]),
        "final/plateau_steps": int(final["plateau/steps"]),
        "activation/mode": "per-muscle-tet-6dof",
        "activation/shared": False,
        "activation/range_clamping": False,
        "activation_inv/initial_rms": float(
            np.linalg.norm(initial_activation)
            / math.sqrt(max(1, initial_activation.size))
        ),
        "activation_inv/initial_max_abs": float(np.abs(initial_activation).max()),
        "initial_displacement/enabled": initial_displacement is not None,
        "initial_displacement/rms": math.nan
        if initial_displacement is None
        else float(
            np.linalg.norm(initial_displacement)
            / math.sqrt(max(1, initial_displacement.shape[0]))
        ),
        "initial_displacement/max": math.nan
        if initial_displacement is None
        else float(np.linalg.norm(initial_displacement, axis=1).max()),
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
        "skin/prestrain": float(SKIN_PRESTRAIN),
        "solver/forward": "PNCG",
        "solver/forward/rtol": float(FORWARD_RTOL),
        "solver/forward/atol": float(FORWARD_ATOL),
        "solver/forward/max_steps": int(FORWARD_MAX_STEPS),
        "solver/adjoint": "FallbackSolver(CupyCG,CupyMinRes)",
        "solver/adjoint/rtol": float(ADJOINT_RTOL),
        "solver/adjoint/atol": float(ADJOINT_ATOL),
        "loss/type": "point-to-point L2",
        "trace": trace,
        **target_metrics,
        **geometry_summary(inverse_mesh),
        **bumpiness_metrics(
            mask=loss_mask,
            edges=bump_edges,
            displacement=best_displacement,
            target=target,
        ),
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
        loss_mask,
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
        "| case | target | tets | active | params | loss pts | stop | best step | best loss | error RMS | error/target | disp edge RMS | residual edge RMS | disp lap RMS | residual lap RMS |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        (
            "| "
            + " | ".join(
                [
                    str(row["case"]),
                    str(row["target/name"]),
                    str(row["n_tets"]),
                    str(row["n_active_tets"]),
                    str(row["n_activation_parameter_dofs"]),
                    str(row["target/loss_points"]),
                    str(row["inverse/stop_reason"]),
                    format_float(row["best/step"]),
                    format_float(row["best/loss"]),
                    format_float(row["best/error_rms"]),
                    format_float(row["best/error_rms_fraction_of_target"]),
                    format_float(row["bumpiness/displacement_edge_rms"]),
                    format_float(row["bumpiness/residual_edge_rms"]),
                    format_float(row["bumpiness/displacement_laplacian_rms"]),
                    format_float(row["bumpiness/residual_laplacian_rms"]),
                ]
            )
            + " |"
        )
        for row in rows
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_mesh(cfg: PrepareConfig) -> None:
    mesh = pv.read(cfg.input_mesh)
    if not isinstance(mesh, pv.UnstructuredGrid):
        mesh = mesh.cast_to_unstructured_grid()
    mesh, n_flipped = orient_tetra_mesh(mesh)
    field_summary = add_required_fields(mesh)
    cfg.output_mesh.parent.mkdir(parents=True, exist_ok=True)
    melon.save(mesh, cfg.output_mesh)

    active_mask = np.asarray(mesh.cell_data["ActivationMask"], dtype=bool)
    summary: dict[str, Any] = {
        "source_mesh": str(cfg.input_mesh),
        "mesh": str(cfg.output_mesh),
        "n_oriented_tets_flipped": int(n_flipped),
        "n_active_tets": int(active_mask.sum()),
        "n_activation_parameter_dofs": int(active_mask.sum() * 6),
        "fat/E_MPa": float(FAT_E),
        "fat/nu": float(FAT_NU),
        "muscle/E_MPa": float(MUSCLE_E),
        "muscle/nu": float(MUSCLE_NU),
        "aponeurosis/E_MPa": float(APONEUROSIS_E),
        "aponeurosis/nu": float(APONEUROSIS_NU),
        "skin/E_MPa": float(SKIN_E),
        "skin/nu": float(SKIN_NU),
        "skin/thickness": float(SKIN_THICKNESS),
        "skin/prestrain": float(SKIN_PRESTRAIN),
        **field_summary,
        **geometry_summary(mesh),
    }
    cfg.output_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    cherries.log_output(cfg.output_mesh)
    cherries.log_output(cfg.output_summary)
    logger.info("Wrote %s", cfg.output_mesh)
    logger.info("Wrote %s", cfg.output_summary)


def run_inverse(cfg: InverseConfig) -> None:
    cfg.output_summary.parent.mkdir(parents=True, exist_ok=True)
    configure_runtime()
    mesh = pv.read(cfg.input_mesh)
    if not isinstance(mesh, pv.UnstructuredGrid):
        mesh = mesh.cast_to_unstructured_grid()
    rows = [solve_case(case, mesh, cfg) for case in selected_cases(cfg)]
    summary = {
        "complete": all(row["inverse/converged"] for row in rows),
        "cases": rows,
        "target/requested": cfg.target,
        "inverse/lr": float(cfg.inverse_lr),
        "inverse/max_steps": int(cfg.inverse_max_steps),
        "inverse/loss_min_delta": float(cfg.inverse_loss_min_delta),
    }
    cfg.output_summary.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
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
