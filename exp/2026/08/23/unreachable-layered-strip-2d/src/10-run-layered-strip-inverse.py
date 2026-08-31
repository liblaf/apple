from __future__ import annotations

# ruff: noqa: EM101, EM102, PLR0915, TRY003
import csv
import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pydantic_settings as ps
import pyvista as pv
import scipy.ndimage as ndi
import scipy.optimize as spo
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from liblaf import cherries

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
DESIGN = "unreachable-layered-plane-strain-strip-inverse"
CASE_ORDER = ("baseline-per-cell", "smoothed-per-cell", "shared-muscle")
MATERIAL_NAMES = ("fat", "SMAS", "muscle")

CaseName = Literal["baseline-per-cell", "smoothed-per-cell", "shared-muscle"]


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)

    output_dir: Path = cherries.output("10-layered-strip-inverse", mkdir=True)
    nx: int = 100
    ny: int = 10
    steps: int = 500
    learning_rate: float = 0.05
    final_learning_rate: float = 3.0e-4
    activation_bound: float = 1.5
    activation_smooth_weight: float = 1.0e-3
    target_y: float = 0.1
    bump_filter_width: float = 0.02
    full_step_history: bool = False


@dataclass(frozen=True)
class FEMModel:
    nx: int
    ny: int
    points: np.ndarray
    triangles: np.ndarray
    material_id: np.ndarray
    young_modulus: np.ndarray
    poisson_ratio: np.ndarray
    areas: np.ndarray
    strain_matrices: tuple[np.ndarray, ...]
    elasticity_matrices: tuple[np.ndarray, ...]
    element_dofs: np.ndarray
    fixed_nodes: np.ndarray
    target_nodes: np.ndarray
    free_dofs: np.ndarray
    free_lookup: np.ndarray
    stiffness_free: sp.csc_matrix
    activation_load: np.ndarray
    response_free: np.ndarray
    response_top: np.ndarray
    muscle_elements: np.ndarray
    muscle_adjacency: np.ndarray
    control_difference: sp.csr_matrix
    target: np.ndarray
    diagnostics: dict[str, float | int]

    @property
    def n_controls(self) -> int:
        return int(self.response_top.shape[1])

    @property
    def n_muscle_elements(self) -> int:
        return int(self.muscle_elements.size)


@dataclass(frozen=True)
class InverseResult:
    name: CaseName
    label: str
    basis: Literal["per-cell", "shared"]
    smooth_weight: float
    steps: np.ndarray
    controls: np.ndarray
    trace: tuple[dict[str, float], ...]
    best_index: int
    reference_controls: np.ndarray
    reference_metrics: dict[str, float | int | str | bool]
    certificate: dict[str, float | int | bool | list[float]]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_config(cfg: Config) -> None:
    if cfg.output_dir.exists() and any(cfg.output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite nonempty output: {cfg.output_dir}"
        )
    if cfg.nx < 20 or cfg.ny < 5:
        raise ValueError("the mesh must have at least 20 x 5 rectangles")
    if cfg.steps < 1:
        raise ValueError("steps must be positive")
    if not 0.0 < cfg.final_learning_rate <= cfg.learning_rate:
        raise ValueError("learning rates must satisfy 0 < final <= initial")
    if cfg.activation_bound <= 0.0:
        raise ValueError("activation_bound must be positive")
    if cfg.activation_smooth_weight <= 0.0:
        raise ValueError("activation_smooth_weight must be positive")
    if cfg.target_y <= 0.0 or cfg.bump_filter_width <= 0.0:
        raise ValueError("target_y and bump_filter_width must be positive")


def plane_strain_elasticity(young: float, poisson: float) -> np.ndarray:
    lame_lambda = young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    lame_mu = young / (2.0 * (1.0 + poisson))
    return np.array(
        [
            [lame_lambda + 2.0 * lame_mu, lame_lambda, 0.0],
            [lame_lambda, lame_lambda + 2.0 * lame_mu, 0.0],
            [0.0, 0.0, lame_mu],
        ],
        dtype=np.float64,
    )


def triangle_strain_matrix(points: np.ndarray) -> tuple[float, np.ndarray]:
    signed_double_area = float(
        np.linalg.det(
            np.array(
                [
                    [1.0, points[0, 0], points[0, 1]],
                    [1.0, points[1, 0], points[1, 1]],
                    [1.0, points[2, 0], points[2, 1]],
                ]
            )
        )
    )
    if signed_double_area <= 0.0:
        raise ValueError("triangle orientation must be positive")
    area = 0.5 * signed_double_area
    bx = (
        np.array(
            [
                points[1, 1] - points[2, 1],
                points[2, 1] - points[0, 1],
                points[0, 1] - points[1, 1],
            ]
        )
        / signed_double_area
    )
    cy = (
        np.array(
            [
                points[2, 0] - points[1, 0],
                points[0, 0] - points[2, 0],
                points[1, 0] - points[0, 0],
            ]
        )
        / signed_double_area
    )
    strain = np.zeros((3, 6), dtype=np.float64)
    strain[0, 0::2] = bx
    strain[1, 1::2] = cy
    strain[2, 0::2] = cy
    strain[2, 1::2] = bx
    return area, strain


def structured_triangles(nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0.0, 1.0, nx + 1)
    y = np.linspace(0.0, 0.1, ny + 1)
    points = np.array([(xx, yy) for yy in y for xx in x], dtype=np.float64)
    triangles: list[tuple[int, int, int]] = []
    for row in range(ny):
        for column in range(nx):
            lower_left = row * (nx + 1) + column
            lower_right = lower_left + 1
            upper_left = (row + 1) * (nx + 1) + column
            upper_right = upper_left + 1
            if (row + column) % 2 == 0:
                triangles.extend(
                    (
                        (lower_left, lower_right, upper_right),
                        (lower_left, upper_right, upper_left),
                    )
                )
            else:
                triangles.extend(
                    (
                        (lower_left, lower_right, upper_left),
                        (lower_right, upper_right, upper_left),
                    )
                )
    return points, np.asarray(triangles, dtype=np.int64)


def material_at(
    centroid: np.ndarray,
    *,
    smas_young: float,
    near_incompressible_poisson: float,
) -> tuple[int, float, float]:
    in_smas = 0.04 - 1.0e-12 <= centroid[1] <= 0.06 + 1.0e-12
    in_muscle = in_smas and 0.05 - 1.0e-12 <= centroid[0] <= 0.22 + 1.0e-12
    if in_muscle:
        return 2, 0.030, near_incompressible_poisson
    if in_smas:
        return 1, smas_young, 0.35
    return 0, 0.003, near_incompressible_poisson


def muscle_graph(
    triangles: np.ndarray, muscle_elements: np.ndarray
) -> tuple[np.ndarray, sp.csr_matrix]:
    edge_owner: dict[tuple[int, int], int] = {}
    adjacency: list[tuple[int, int]] = []
    for local_index, element_index in enumerate(muscle_elements):
        triangle = triangles[element_index]
        for a, b in ((0, 1), (1, 2), (2, 0)):
            edge = tuple(sorted((int(triangle[a]), int(triangle[b]))))
            if edge in edge_owner:
                adjacency.append((edge_owner[edge], local_index))
            else:
                edge_owner[edge] = local_index
    adjacency_array = np.asarray(adjacency, dtype=np.int64).reshape(-1, 2)
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for edge_index, (left, right) in enumerate(adjacency_array):
        for component in range(3):
            row = 3 * edge_index + component
            rows.extend((row, row))
            columns.extend((3 * int(left) + component, 3 * int(right) + component))
            values.extend((1.0, -1.0))
    difference = sp.coo_matrix(
        (values, (rows, columns)),
        shape=(3 * adjacency_array.shape[0], 3 * muscle_elements.size),
        dtype=np.float64,
    ).tocsr()
    return adjacency_array, difference


def build_model(
    nx: int,
    ny: int,
    target_y: float,
    *,
    smas_young: float = 0.100,
    near_incompressible_poisson: float = 0.49,
) -> FEMModel:
    points, triangles = structured_triangles(nx, ny)
    n_dofs = 2 * points.shape[0]
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    material_id = np.empty(triangles.shape[0], dtype=np.int32)
    young_modulus = np.empty(triangles.shape[0], dtype=np.float64)
    poisson_ratio = np.empty(triangles.shape[0], dtype=np.float64)
    areas = np.empty(triangles.shape[0], dtype=np.float64)
    element_dofs = np.empty((triangles.shape[0], 6), dtype=np.int64)
    strain_matrices: list[np.ndarray] = []
    elasticity_matrices: list[np.ndarray] = []

    for element_index, triangle in enumerate(triangles):
        triangle_points = points[triangle]
        area, strain = triangle_strain_matrix(triangle_points)
        centroid = np.mean(triangle_points, axis=0)
        material, young, poisson = material_at(
            centroid,
            smas_young=smas_young,
            near_incompressible_poisson=near_incompressible_poisson,
        )
        elasticity = plane_strain_elasticity(young, poisson)
        dofs = np.ravel(np.column_stack((2 * triangle, 2 * triangle + 1)))
        element_stiffness = area * strain.T @ elasticity @ strain
        element_rows, element_columns = np.meshgrid(dofs, dofs, indexing="ij")
        rows.extend(int(value) for value in element_rows.ravel())
        columns.extend(int(value) for value in element_columns.ravel())
        values.extend(float(value) for value in element_stiffness.ravel())
        material_id[element_index] = material
        young_modulus[element_index] = young
        poisson_ratio[element_index] = poisson
        areas[element_index] = area
        element_dofs[element_index] = dofs
        strain_matrices.append(strain)
        elasticity_matrices.append(elasticity)

    stiffness = sp.coo_matrix(
        (values, (rows, columns)), shape=(n_dofs, n_dofs), dtype=np.float64
    ).tocsr()
    fixed_nodes = np.flatnonzero(
        np.isclose(points[:, 1], 0.0)
        | np.isclose(points[:, 0], 0.0)
        | np.isclose(points[:, 0], 1.0)
    )
    fixed_dofs = np.ravel(np.column_stack((2 * fixed_nodes, 2 * fixed_nodes + 1)))
    free_dofs = np.setdiff1d(np.arange(n_dofs, dtype=np.int64), fixed_dofs)
    free_lookup = np.full(n_dofs, -1, dtype=np.int64)
    free_lookup[free_dofs] = np.arange(free_dofs.size)
    stiffness_free = stiffness[free_dofs][:, free_dofs].tocsc()

    muscle_elements = np.flatnonzero(material_id == 2)
    activation_load = np.zeros(
        (free_dofs.size, 3 * muscle_elements.size), dtype=np.float64
    )
    for muscle_local, element_index in enumerate(muscle_elements):
        active_force = (
            areas[element_index]
            * strain_matrices[element_index].T
            @ elasticity_matrices[element_index]
        )
        for element_local, dof in enumerate(element_dofs[element_index]):
            free_index = free_lookup[dof]
            if free_index >= 0:
                activation_load[
                    free_index, 3 * muscle_local : 3 * muscle_local + 3
                ] += active_force[element_local]

    factor = spla.splu(stiffness_free)
    response_free = -np.asarray(factor.solve(activation_load), dtype=np.float64)
    target_nodes = np.flatnonzero(
        np.isclose(points[:, 1], 0.1) & (points[:, 0] > 0.0) & (points[:, 0] < 1.0)
    )
    target_dofs = np.ravel(np.column_stack((2 * target_nodes, 2 * target_nodes + 1)))
    target_free = free_lookup[target_dofs]
    if np.any(target_free < 0):
        raise RuntimeError("a target degree of freedom is fixed")
    response_top = response_free[target_free]
    target = np.tile(np.array([0.0, target_y]), target_nodes.size)
    muscle_adjacency, control_difference = muscle_graph(triangles, muscle_elements)

    symmetry_error = float(
        spla.norm(stiffness_free - stiffness_free.T)
        / max(spla.norm(stiffness_free), np.finfo(float).tiny)
    )
    smallest_eigenvalue = float(
        spla.eigsh(
            stiffness_free,
            k=1,
            which="SM",
            return_eigenvectors=False,
            tol=1.0e-8,
        )[0]
    )
    diagnostics: dict[str, float | int] = {
        "n_points": int(points.shape[0]),
        "n_triangles": int(triangles.shape[0]),
        "n_free_dofs": int(free_dofs.size),
        "n_target_points": int(target_nodes.size),
        "n_muscle_triangles": int(muscle_elements.size),
        "n_per_cell_controls": int(response_top.shape[1]),
        "n_muscle_adjacency_edges": int(muscle_adjacency.shape[0]),
        "stiffness_symmetry_relative_error": symmetry_error,
        "stiffness_smallest_eigenvalue": smallest_eigenvalue,
    }
    return FEMModel(
        nx=nx,
        ny=ny,
        points=points,
        triangles=triangles,
        material_id=material_id,
        young_modulus=young_modulus,
        poisson_ratio=poisson_ratio,
        areas=areas,
        strain_matrices=tuple(strain_matrices),
        elasticity_matrices=tuple(elasticity_matrices),
        element_dofs=element_dofs,
        fixed_nodes=fixed_nodes,
        target_nodes=target_nodes,
        free_dofs=free_dofs,
        free_lookup=free_lookup,
        stiffness_free=stiffness_free,
        activation_load=activation_load,
        response_free=response_free,
        response_top=response_top,
        muscle_elements=muscle_elements,
        muscle_adjacency=muscle_adjacency,
        control_difference=control_difference,
        target=target,
        diagnostics=diagnostics,
    )


def response_for_basis(
    model: FEMModel, basis: Literal["per-cell", "shared"]
) -> np.ndarray:
    if basis == "per-cell":
        return model.response_top
    return np.sum(
        model.response_top.reshape(
            model.response_top.shape[0], model.n_muscle_elements, 3
        ),
        axis=1,
    )


def cell_controls(
    model: FEMModel,
    controls: np.ndarray,
    basis: Literal["per-cell", "shared"],
) -> np.ndarray:
    if basis == "per-cell":
        return controls.reshape(model.n_muscle_elements, 3)
    return np.repeat(controls.reshape(1, 3), model.n_muscle_elements, axis=0)


def projection_certificate(response: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    singular_values = np.linalg.svd(response, compute_uv=False)
    tolerance = 1.0e-10 * singular_values[0]
    rank = int(np.count_nonzero(singular_values > tolerance))
    projection_controls = np.linalg.lstsq(response, target, rcond=1.0e-10)[0]
    projection_residual = target - response @ projection_controls
    orthogonality_absolute = float(
        np.max(np.abs(response.T @ projection_residual), initial=0.0)
    )
    orthogonality_relative = float(
        np.linalg.norm(response.T @ projection_residual)
        / max(
            np.linalg.norm(response) * np.linalg.norm(projection_residual),
            np.finfo(float).tiny,
        )
    )
    projection_rms = float(np.sqrt(np.mean(projection_residual**2)))
    target_rms = float(np.sqrt(np.mean(target**2)))
    smallest_retained = float(singular_values[rank - 1]) if rank else 0.0
    condition = float(singular_values[0] / smallest_retained) if rank else math.inf
    return {
        "n_outputs": int(response.shape[0]),
        "n_controls": int(response.shape[1]),
        "rank_tolerance": float(tolerance),
        "effective_rank": rank,
        "effective_nullity": int(response.shape[1] - rank),
        "largest_singular_value": float(singular_values[0]),
        "smallest_retained_singular_value": smallest_retained,
        "retained_condition_number": condition,
        "projection_residual_rms": projection_rms,
        "target_rms": target_rms,
        "projection_residual_fraction_of_target": projection_rms / target_rms,
        "orthogonality_max_abs": orthogonality_absolute,
        "orthogonality_relative": orthogonality_relative,
        "unreachable_certified": bool(
            projection_rms > 1.0e-6 and orthogonality_relative < 1.0e-8
        ),
        "singular_values": [float(value) for value in singular_values],
    }


def augmented_system(
    response: np.ndarray,
    target: np.ndarray,
    difference: sp.csr_matrix | None,
    smooth_weight: float,
) -> tuple[np.ndarray, np.ndarray]:
    blocks = [response / math.sqrt(response.shape[0])]
    targets = [target / math.sqrt(response.shape[0])]
    if smooth_weight > 0.0:
        if difference is None or difference.shape[0] == 0:
            raise ValueError("smooth regularization requires a nonempty difference")
        blocks.append(
            math.sqrt(smooth_weight / difference.shape[0]) * difference.toarray()
        )
        targets.append(np.zeros(difference.shape[0], dtype=np.float64))
    return np.vstack(blocks), np.concatenate(targets)


def scalar_metrics(
    model: FEMModel,
    response: np.ndarray,
    controls: np.ndarray,
    basis: Literal["per-cell", "shared"],
    difference: sp.csr_matrix | None,
    smooth_weight: float,
    activation_bound: float,
    bump_filter_width: float,
) -> dict[str, float]:
    predicted = response @ controls
    residual = predicted - model.target
    top_vectors = predicted.reshape(-1, 2)
    top_y = top_vectors[:, 1]
    dx = 1.0 / model.nx
    high_pass = top_y - ndi.gaussian_filter1d(
        top_y,
        sigma=bump_filter_width / dx,
        mode="nearest",
    )
    top_x = model.points[model.target_nodes, 0]
    near = top_x <= 0.30
    far = top_x >= 0.50
    cell = cell_controls(model, controls, basis)
    if model.muscle_adjacency.size:
        jumps = cell[model.muscle_adjacency[:, 0]] - cell[model.muscle_adjacency[:, 1]]
        jump_rms = float(np.sqrt(np.mean(jumps**2)))
    else:
        jump_rms = 0.0
    data_mse = float(np.mean(residual**2))
    if smooth_weight > 0.0 and difference is not None:
        difference_values = difference @ controls
        smooth_mse = float(np.mean(difference_values**2))
    else:
        smooth_mse = 0.0
    return {
        "objective_total": data_mse + smooth_weight * smooth_mse,
        "data_mse": data_mse,
        "error_mae": float(np.mean(np.abs(residual))),
        "error_rms": float(np.sqrt(data_mse)),
        "error_max": float(np.max(np.abs(residual))),
        "error_rms_fraction_of_target": float(
            np.sqrt(data_mse) / np.sqrt(np.mean(model.target**2))
        ),
        "top_y_mean": float(np.mean(top_y)),
        "top_y_std": float(np.std(top_y)),
        "top_y_min": float(np.min(top_y)),
        "top_y_max": float(np.max(top_y)),
        "top_y_range": float(np.ptp(top_y)),
        "top_y_near_muscle_mean": float(np.mean(top_y[near])),
        "top_y_far_mean": float(np.mean(top_y[far])),
        "top_y_highpass_rms": float(np.sqrt(np.mean(high_pass**2))),
        "top_y_highpass_peak_to_valley": float(np.ptp(high_pass)),
        "top_y_first_difference_rms": float(np.sqrt(np.mean(np.diff(top_y) ** 2))),
        "top_y_second_difference_rms": float(
            np.sqrt(np.mean(np.diff(top_y, n=2) ** 2))
        ),
        "activation_rms": float(np.sqrt(np.mean(cell**2))),
        "activation_max_abs": float(np.max(np.abs(cell))),
        "activation_neighbor_jump_rms": jump_rms,
        "activation_bound_hits": float(
            np.count_nonzero(
                np.isclose(
                    np.abs(controls),
                    activation_bound,
                    rtol=1.0e-6,
                    atol=1.0e-8,
                )
            )
        ),
        "activation_smooth_mse": smooth_mse,
    }


def full_state_metrics(
    model: FEMModel,
    controls: np.ndarray,
    basis: Literal["per-cell", "shared"],
) -> dict[str, float]:
    cell = cell_controls(model, controls, basis)
    flat_cell = cell.ravel()
    displacement_free = model.response_free @ flat_cell
    displacement_dofs = np.zeros(2 * model.points.shape[0], dtype=np.float64)
    displacement_dofs[model.free_dofs] = displacement_free
    equilibrium_residual = (
        model.stiffness_free @ displacement_free + model.activation_load @ flat_cell
    )
    force_norm = np.linalg.norm(model.activation_load @ flat_cell)
    max_principal = 0.0
    max_abs_volumetric = 0.0
    max_elastic_principal = 0.0
    for element_index, dofs in enumerate(model.element_dofs):
        strain_vector = model.strain_matrices[element_index] @ displacement_dofs[dofs]
        strain_tensor = np.array(
            [
                [strain_vector[0], 0.5 * strain_vector[2]],
                [0.5 * strain_vector[2], strain_vector[1]],
            ]
        )
        max_principal = max(
            max_principal, float(np.max(np.abs(np.linalg.eigvalsh(strain_tensor))))
        )
        max_abs_volumetric = max(
            max_abs_volumetric, float(abs(strain_vector[0] + strain_vector[1]))
        )
        elastic_vector = strain_vector.copy()
        if model.material_id[element_index] == 2:
            muscle_local = int(np.searchsorted(model.muscle_elements, element_index))
            elastic_vector += cell[muscle_local]
        elastic_tensor = np.array(
            [
                [elastic_vector[0], 0.5 * elastic_vector[2]],
                [0.5 * elastic_vector[2], elastic_vector[1]],
            ]
        )
        max_elastic_principal = max(
            max_elastic_principal,
            float(np.max(np.abs(np.linalg.eigvalsh(elastic_tensor)))),
        )
    return {
        "equilibrium_relative_residual": float(
            np.linalg.norm(equilibrium_residual) / max(force_norm, np.finfo(float).tiny)
        ),
        "maximum_abs_principal_displacement_strain": max_principal,
        "maximum_abs_volumetric_displacement_strain": max_abs_volumetric,
        "maximum_abs_principal_elastic_strain": max_elastic_principal,
        "small_strain_limit_exceeded": bool(
            max(max_principal, max_abs_volumetric, max_elastic_principal) > 0.10
        ),
    }


def bounded_reference(
    model: FEMModel,
    response: np.ndarray,
    basis: Literal["per-cell", "shared"],
    difference: sp.csr_matrix | None,
    smooth_weight: float,
    activation_bound: float,
    bump_filter_width: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    augmented_response, augmented_target = augmented_system(
        response, model.target, difference, smooth_weight
    )
    solved = spo.lsq_linear(
        augmented_response,
        augmented_target,
        bounds=(-activation_bound, activation_bound),
        method="trf",
        tol=1.0e-12,
        lsq_solver="exact",
        max_iter=2000,
        verbose=0,
    )
    if not solved.success:
        raise RuntimeError(f"bounded least-squares reference failed: {solved.message}")
    controls = np.asarray(solved.x, dtype=np.float64)
    metrics: dict[str, Any] = scalar_metrics(
        model,
        response,
        controls,
        basis,
        difference,
        smooth_weight,
        activation_bound,
        bump_filter_width,
    )
    metrics.update(full_state_metrics(model, controls, basis))
    metrics.update(
        {
            "solver": "scipy.optimize.lsq_linear/trf/exact",
            "solver_status": int(solved.status),
            "solver_message": str(solved.message),
            "solver_optimality": float(solved.optimality),
            "solver_active_mask_count": int(np.count_nonzero(solved.active_mask)),
            "solver_success": bool(solved.success),
        }
    )
    return controls, metrics


def run_adam(
    name: CaseName,
    label: str,
    model: FEMModel,
    basis: Literal["per-cell", "shared"],
    smooth_weight: float,
    cfg: Config,
) -> InverseResult:
    response = response_for_basis(model, basis)
    difference = model.control_difference if smooth_weight > 0.0 else None
    certificate = projection_certificate(response, model.target)
    if not bool(certificate["unreachable_certified"]):
        raise RuntimeError(f"unreachability certificate failed for {name}")
    reference_controls, reference_metrics = bounded_reference(
        model,
        response,
        basis,
        difference,
        smooth_weight,
        cfg.activation_bound,
        cfg.bump_filter_width,
    )

    torch.set_default_dtype(torch.float64)
    response_torch = torch.from_numpy(response)
    target_torch = torch.from_numpy(model.target)
    difference_torch = (
        torch.from_numpy(model.control_difference.toarray())
        if smooth_weight > 0.0
        else None
    )
    controls = torch.zeros(response.shape[1], requires_grad=True)
    optimizer = torch.optim.Adam([controls], lr=cfg.learning_rate)
    saved_controls: list[np.ndarray] = []
    trace: list[dict[str, float]] = []

    for step in range(cfg.steps + 1):
        with torch.no_grad():
            controls_numpy = controls.detach().cpu().numpy().copy()
        metrics = scalar_metrics(
            model,
            response,
            controls_numpy,
            basis,
            difference,
            smooth_weight,
            cfg.activation_bound,
            cfg.bump_filter_width,
        )
        if step == 0:
            learning_rate = cfg.learning_rate
        else:
            learning_rate = float(optimizer.param_groups[0]["lr"])
        trace.append(
            {
                "step": float(step),
                "learning_rate": learning_rate,
                **metrics,
            }
        )
        saved_controls.append(controls_numpy)
        if step == cfg.steps:
            break

        phase = (step - 1) / max(cfg.steps - 1, 1)
        learning_rate = cfg.final_learning_rate + 0.5 * (
            cfg.learning_rate - cfg.final_learning_rate
        ) * (1.0 + math.cos(math.pi * phase))
        optimizer.param_groups[0]["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        residual = response_torch @ controls - target_torch
        loss = torch.mean(residual**2)
        if smooth_weight > 0.0:
            if difference_torch is None:
                raise AssertionError("missing smoothness operator")
            loss = loss + smooth_weight * torch.mean((difference_torch @ controls) ** 2)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            controls.clamp_(-cfg.activation_bound, cfg.activation_bound)

    objective = np.array([row["objective_total"] for row in trace])
    best_index = int(np.argmin(objective))
    return InverseResult(
        name=name,
        label=label,
        basis=basis,
        smooth_weight=smooth_weight,
        steps=np.arange(cfg.steps + 1, dtype=np.int64),
        controls=np.asarray(saved_controls),
        trace=tuple(trace),
        best_index=best_index,
        reference_controls=reference_controls,
        reference_metrics=reference_metrics,
        certificate=certificate,
    )


def displacement_for_controls(
    model: FEMModel,
    controls: np.ndarray,
    basis: Literal["per-cell", "shared"],
) -> np.ndarray:
    flat_cell = cell_controls(model, controls, basis).ravel()
    free = model.response_free @ flat_cell
    displacement = np.zeros((model.points.shape[0], 3), dtype=np.float64)
    displacement_dofs = np.zeros(2 * model.points.shape[0], dtype=np.float64)
    displacement_dofs[model.free_dofs] = free
    displacement[:, :2] = displacement_dofs.reshape(-1, 2)
    return displacement


def make_grid(
    model: FEMModel,
    controls: np.ndarray,
    basis: Literal["per-cell", "shared"],
    step: int,
    target_y: float,
) -> pv.UnstructuredGrid:
    cells = np.column_stack(
        (
            np.full(model.triangles.shape[0], 3, dtype=np.int64),
            model.triangles,
        )
    ).ravel()
    cell_types = np.full(
        model.triangles.shape[0], int(pv.CellType.TRIANGLE), dtype=np.uint8
    )
    points_3d = np.column_stack(
        (model.points, np.zeros(model.points.shape[0], dtype=np.float64))
    )
    grid = pv.UnstructuredGrid(cells, cell_types, points_3d)
    displacement = displacement_for_controls(model, controls, basis)
    target_displacement = np.zeros_like(displacement)
    target_displacement[model.target_nodes, 1] = target_y
    target_mask = np.zeros(model.points.shape[0], dtype=np.uint8)
    target_mask[model.target_nodes] = 1
    fixed_mask = np.zeros(model.points.shape[0], dtype=np.uint8)
    fixed_mask[model.fixed_nodes] = 1
    cell = cell_controls(model, controls, basis)
    activation = np.zeros((model.triangles.shape[0], 3), dtype=np.float64)
    activation[model.muscle_elements] = cell
    grid.point_data["Displacement"] = displacement
    grid.point_data["DisplacementY"] = displacement[:, 1]
    grid.point_data["TargetDisplacement"] = target_displacement
    grid.point_data["TargetMask"] = target_mask
    grid.point_data["FixedMask"] = fixed_mask
    grid.cell_data["MaterialId"] = model.material_id
    grid.cell_data["YoungModulusMPa"] = model.young_modulus
    grid.cell_data["PoissonRatio"] = model.poisson_ratio
    grid.cell_data["ActivationXX"] = activation[:, 0]
    grid.cell_data["ActivationYY"] = activation[:, 1]
    grid.cell_data["ActivationXYEngineering"] = activation[:, 2]
    grid.cell_data["ActivationNorm"] = np.linalg.norm(activation, axis=1)
    grid.cell_data["MuscleMask"] = (model.material_id == 2).astype(np.uint8)
    grid.cell_data["SMASMask"] = (model.material_id == 1).astype(np.uint8)
    grid.field_data["InverseStep"] = np.array([step], dtype=np.int64)
    return grid


def write_polyline(path: Path, points: np.ndarray, name: str) -> None:
    points_3d = np.column_stack((points, np.zeros(points.shape[0], dtype=np.float64)))
    polyline = pv.PolyData(points_3d)
    polyline.lines = np.concatenate(
        (np.array([points.shape[0]], dtype=np.int64), np.arange(points.shape[0]))
    )
    polyline.point_data[name] = np.ones(points.shape[0], dtype=np.uint8)
    polyline.save(path)


def write_trace(path: Path, result: InverseResult) -> None:
    keys = list(result.trace[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(result.trace)


def write_case(
    output_dir: Path,
    model: FEMModel,
    result: InverseResult,
    cfg: Config,
) -> dict[str, Any]:
    case_dir = output_dir / result.name
    case_dir.mkdir(parents=False, exist_ok=False)
    frames_dir = case_dir / "frames"
    frames_dir.mkdir()
    write_trace(case_dir / "trace.csv", result)
    np.savez_compressed(
        case_dir / "history.npz",
        steps=result.steps,
        controls=result.controls,
        reference_controls=result.reference_controls,
        best_index=np.array([result.best_index], dtype=np.int64),
    )

    if cfg.full_step_history and result.name == "baseline-per-cell":
        selected_steps = list(range(cfg.steps + 1))
        history_sampling = "every-step"
    else:
        requested_steps = {
            0,
            1,
            2,
            5,
            10,
            20,
            30,
            50,
            75,
            100,
            150,
            200,
            300,
            400,
            cfg.steps,
            int(result.steps[result.best_index]),
        }
        selected_steps = sorted(step for step in requested_steps if step <= cfg.steps)
        history_sampling = "selected-checkpoints"
    files: list[dict[str, float | str]] = []
    frame_paths: dict[int, Path] = {}
    for step in selected_steps:
        frame_path = frames_dir / f"step-{step:04d}.vtu"
        make_grid(
            model,
            result.controls[step],
            result.basis,
            step,
            cfg.target_y,
        ).save(frame_path)
        frame_paths[step] = frame_path
        files.append(
            {
                "name": str(frame_path.relative_to(case_dir)),
                "time": float(step),
            }
        )
    series_path = case_dir / "history.vtu.series"
    write_json(series_path, {"file-series-version": "1.0", "files": files})

    best_step = int(result.steps[result.best_index])
    best_metrics: dict[str, Any] = dict(result.trace[result.best_index])
    best_metrics.update(
        full_state_metrics(model, result.controls[result.best_index], result.basis)
    )
    reference_objective = float(result.reference_metrics["objective_total"])
    best_metrics["objective_gap_to_bounded_reference_fraction"] = float(
        (float(best_metrics["objective_total"]) - reference_objective)
        / max(reference_objective, np.finfo(float).tiny)
    )
    best_frame = frame_paths[best_step]
    return {
        "name": result.name,
        "label": result.label,
        "basis": result.basis,
        "smooth_weight": result.smooth_weight,
        "steps": cfg.steps,
        "best_step": best_step,
        "history_sampling": history_sampling,
        "selected_steps": selected_steps,
        "series": str(series_path.relative_to(output_dir)),
        "series_sha256": sha256(series_path),
        "best_frame": str(best_frame.relative_to(output_dir)),
        "best_frame_sha256": sha256(best_frame),
        "trace": str((case_dir / "trace.csv").relative_to(output_dir)),
        "history": str((case_dir / "history.npz").relative_to(output_dir)),
        "certificate": result.certificate,
        "best": best_metrics,
        "bounded_reference": result.reference_metrics,
    }


def reference_ablation(
    name: str,
    model: FEMModel,
    cfg: Config,
    *,
    note: str,
) -> dict[str, Any]:
    response = model.response_top
    certificate = projection_certificate(response, model.target)
    controls, metrics = bounded_reference(
        model,
        response,
        "per-cell",
        None,
        0.0,
        cfg.activation_bound,
        cfg.bump_filter_width,
    )
    return {
        "name": name,
        "note": note,
        "mesh": model.diagnostics,
        "certificate": certificate,
        "bounded_reference": metrics,
        "control_count": int(controls.size),
    }


def write_results_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Layered 2D unreachable-target inverse results",
        "",
        "| case | target RMS fraction | top range | high-pass RMS | control jump RMS | reference gap |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for case in summary["cases"]:
        best = case["best"]
        lines.append(
            "| {label} | {error:.3%} | {top_range:.6f} | {bump:.6f} | "
            "{jump:.6f} | {gap:.3%} |".format(
                label=case["label"],
                error=best["error_rms_fraction_of_target"],
                top_range=best["top_y_range"],
                bump=best["top_y_highpass_rms"],
                jump=best["activation_neighbor_jump_rms"],
                gap=best["objective_gap_to_bounded_reference_fraction"],
            )
        )
    baseline = summary["cases"][0]
    lines.extend(
        [
            "",
            "The entire free top targets +0.1 in y. The muscle occupies only the left "
            "part of the stiff middle layer. The unrestricted projection residual is "
            f"{baseline['certificate']['projection_residual_fraction_of_target']:.2%} "
            "of the target RMS, which certifies that the uniform target is outside the "
            "linear response span.",
            "",
            "All geometry and field images are rendered separately by ParaView. This "
            "runner writes only VTK inputs, traces, and numerical summaries.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    started = time.perf_counter()
    cfg = Config()
    validate_config(cfg)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("building %d x %d plane-strain FEM", cfg.nx, cfg.ny)
    model = build_model(cfg.nx, cfg.ny, cfg.target_y)

    results = (
        run_adam(
            "baseline-per-cell",
            "Per-cell tensor, no regularization",
            model,
            "per-cell",
            0.0,
            cfg,
        ),
        run_adam(
            "smoothed-per-cell",
            "Per-cell tensor + neighbor smoothing",
            model,
            "per-cell",
            cfg.activation_smooth_weight,
            cfg,
        ),
        run_adam(
            "shared-muscle",
            "One tensor shared by the muscle",
            model,
            "shared",
            0.0,
            cfg,
        ),
    )

    top_points = np.column_stack(
        (np.linspace(0.0, 1.0, cfg.nx + 1), np.full(cfg.nx + 1, 0.1))
    )
    write_polyline(cfg.output_dir / "rest-top.vtp", top_points, "RestTop")
    target_points = top_points.copy()
    target_points[:, 1] += cfg.target_y
    write_polyline(cfg.output_dir / "target-top.vtp", target_points, "TargetTop")

    case_summaries = [
        write_case(cfg.output_dir, model, result, cfg) for result in results
    ]
    ablations = [
        reference_ablation(
            "soft-middle-layer",
            build_model(
                cfg.nx,
                cfg.ny,
                cfg.target_y,
                smas_young=0.003,
            ),
            cfg,
            note="SMAS Young's modulus set equal to fat; all else unchanged",
        ),
        reference_ablation(
            "poisson-0.45",
            build_model(
                cfg.nx,
                cfg.ny,
                cfg.target_y,
                near_incompressible_poisson=0.45,
            ),
            cfg,
            note="fat and muscle Poisson ratio reduced from 0.49 to 0.45",
        ),
        reference_ablation(
            "coarse-x",
            build_model(
                cfg.nx // 2,
                cfg.ny,
                cfg.target_y,
            ),
            cfg,
            note="x resolution halved; per-cell control basis changes with the mesh",
        ),
    ]

    baseline = case_summaries[0]
    smoothed = case_summaries[1]
    shared = case_summaries[2]
    checks = {
        "stiffness_is_symmetric": bool(
            model.diagnostics["stiffness_symmetry_relative_error"] < 1.0e-12
        ),
        "stiffness_is_positive_definite": bool(
            model.diagnostics["stiffness_smallest_eigenvalue"] > 0.0
        ),
        "uniform_target_is_certified_unreachable": bool(
            baseline["certificate"]["unreachable_certified"]
        ),
        "all_adam_histories_are_finite": bool(
            all(
                np.all(np.isfinite(result.controls))
                and all(
                    np.isfinite(value) for row in result.trace for value in row.values()
                )
                for result in results
            )
        ),
        "smoothing_reduces_control_jumps": bool(
            smoothed["best"]["activation_neighbor_jump_rms"]
            < baseline["best"]["activation_neighbor_jump_rms"]
        ),
        "shared_control_reduces_control_jumps": bool(
            shared["best"]["activation_neighbor_jump_rms"]
            < baseline["best"]["activation_neighbor_jump_rms"]
        ),
        "bounded_reference_is_also_bumpy": bool(
            baseline["bounded_reference"]["top_y_highpass_rms"] > 1.0e-5
        ),
    }
    complete = all(checks.values())
    summary: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "design": DESIGN,
        "complete": complete,
        "status": "ok" if complete else "failed-checks",
        "scope": (
            "small-strain linear plane-strain mechanism surrogate of the current "
            "3D active-solid experiment; not native Apple nonlinear 2D FEM"
        ),
        "renderer_contract": "ParaView 6.1.1 renders all geometry and field images",
        "geometry": {
            "width": 1.0,
            "height": 0.1,
            "smas_y_bounds": [0.04, 0.06],
            "muscle_x_bounds": [0.05, 0.22],
            "muscle_y_bounds": [0.04, 0.06],
            "skin_enabled": False,
        },
        "materials": {
            "fat": {"young_modulus_MPa": 0.003, "poisson_ratio": 0.49},
            "SMAS": {"young_modulus_MPa": 0.100, "poisson_ratio": 0.35},
            "muscle": {"young_modulus_MPa": 0.030, "poisson_ratio": 0.49},
        },
        "boundary": "bottom, left, and right fixed in x and y",
        "target": "every free top node has displacement (0, +0.1)",
        "activation_semantics": (
            "symmetric active strain [xx, yy, engineering-xy], entering the "
            "linearized energy as B u + activation"
        ),
        "activation_bound": [-cfg.activation_bound, cfg.activation_bound],
        "model": model.diagnostics,
        "cases": case_summaries,
        "reference_ablations": ablations,
        "checks": checks,
        "elapsed_seconds": float(time.perf_counter() - started),
        "runtime": {
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "torch": torch.__version__,
            "pyvista": pv.__version__,
        },
    }
    write_json(cfg.output_dir / "summary.json", summary)
    write_results_markdown(cfg.output_dir / "results.md", summary)

    contract = {
        "schema_version": SCHEMA_VERSION,
        "design": DESIGN,
        "complete": complete,
        "required_paraview_version": "6.1.1",
        "image_resolution": [1600, 900],
        "comparison_resolution": [1600, 1200],
        "camera": {
            "position": [0.5, 0.105, 3.0],
            "focal_point": [0.5, 0.105, 0.0],
            "view_up": [0.0, 1.0, 0.0],
            "parallel_scale": 0.16,
        },
        "displacement_y_range": [
            min(
                0.0,
                *(float(case["best"]["top_y_min"]) for case in case_summaries),
            ),
            max(float(case["best"]["top_y_max"]) for case in case_summaries),
        ],
        "activation_norm_range": [0.0, math.sqrt(3.0) * cfg.activation_bound],
        "rest_top": "rest-top.vtp",
        "rest_top_sha256": sha256(cfg.output_dir / "rest-top.vtp"),
        "target_top": "target-top.vtp",
        "target_top_sha256": sha256(cfg.output_dir / "target-top.vtp"),
        "cases": [
            {
                "name": case["name"],
                "label": case["label"],
                "steps": case["steps"],
                "series": case["series"],
                "series_sha256": case["series_sha256"],
                "history_sampling": case["history_sampling"],
                "selected_steps": case["selected_steps"],
                "best_frame": case["best_frame"],
                "best_frame_sha256": case["best_frame_sha256"],
                "best_step": case["best_step"],
                "best": case["best"],
                "bounded_reference": case["bounded_reference"],
            }
            for case in case_summaries
        ],
    }
    write_json(cfg.output_dir / "paraview-contract.json", contract)
    if not complete:
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"experiment validation failed: {failed}")
    logger.info("completed in %.2f s", summary["elapsed_seconds"])


if __name__ == "__main__":
    main()
