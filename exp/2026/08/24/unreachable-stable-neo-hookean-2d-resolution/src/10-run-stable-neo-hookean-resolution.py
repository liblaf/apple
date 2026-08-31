# Copyright (c) 2026 liblaf
from __future__ import annotations

# ruff: noqa: B007, C901, EM101, EM102, PLR0912, PLR0915, TRY003
import csv
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
import pyvista as pv
import scipy.ndimage as ndi
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from liblaf import cherries

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
MATERIAL_NAMES = ("fat", "SMAS", "muscle")
YOUNG_MPA = np.array([0.003, 0.100, 0.030], dtype=np.float64)
POISSON = 0.49
ACTIVATION_BASES = np.array(
    [
        [[1.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [0.0, 1.0]],
        [[0.0, 1.0], [1.0, 0.0]],
    ],
    dtype=np.float64,
)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)

    output_dir: Path = cherries.output("10-stable-neo-hookean-resolution", mkdir=True)
    resolutions: str = "50x5,100x10,200x20"
    variants: str = "free,tied,regularized"
    max_steps: int = 160
    learning_rate: float = 0.05
    patience: int = 20
    minimum_delta: float = 5.0e-6
    regularization_weight: float = 1.0e-4
    tied_nx: int = 50
    tied_ny: int = 5
    target_y: float = 0.1
    poisson_ratio: float = 0.49
    bump_filter_width: float = 0.02
    forward_tolerance: float = 1.0e-9
    forward_max_iterations: int = 80
    save_every: int = 1
    validate_derivatives: bool = True


@dataclass(frozen=True)
class Mesh:
    nx: int
    ny: int
    points: np.ndarray
    triangles: np.ndarray
    gradients: np.ndarray
    areas: np.ndarray
    material_id: np.ndarray
    young: np.ndarray
    lame_lambda: np.ndarray
    lame_mu: np.ndarray
    element_dofs: np.ndarray
    fixed_nodes: np.ndarray
    free_dofs: np.ndarray
    free_lookup: np.ndarray
    top_nodes: np.ndarray
    muscle_elements: np.ndarray
    muscle_local: np.ndarray
    matrix_element: np.ndarray
    matrix_local_row: np.ndarray
    matrix_local_col: np.ndarray
    matrix_rows: np.ndarray
    matrix_cols: np.ndarray
    muscle_edges: tuple[tuple[int, int, float], ...]

    @property
    def n_free(self) -> int:
        return int(self.free_dofs.size)


@dataclass(frozen=True)
class ControlMap:
    variant: str
    element_group: np.ndarray
    n_groups: int
    regularizer: sp.csr_matrix

    @property
    def n_controls(self) -> int:
        return 3 * self.n_groups


@dataclass
class ForwardState:
    u_free: np.ndarray
    energy: float
    residual: np.ndarray
    hessian: sp.csc_matrix
    det_deformation: np.ndarray
    det_elastic: np.ndarray
    det_active_inv: np.ndarray
    min_singular_active_inv: np.ndarray
    iterations: int
    converged: bool
    line_search_failures: int


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def parse_resolutions(text: str) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    for token in text.split(","):
        fields = token.strip().lower().split("x")
        if len(fields) != 2:
            raise ValueError(f"invalid resolution {token!r}; expected NXxNY")
        nx, ny = map(int, fields)
        if nx < 10 or ny < 2:
            raise ValueError("each resolution must be at least 10x2")
        result.append((nx, ny))
    if not result:
        raise ValueError("at least one resolution is required")
    return result


def cofactor_2d(matrix: np.ndarray) -> np.ndarray:
    result = np.empty_like(matrix)
    result[..., 0, 0] = matrix[..., 1, 1]
    result[..., 0, 1] = -matrix[..., 1, 0]
    result[..., 1, 0] = -matrix[..., 0, 1]
    result[..., 1, 1] = matrix[..., 0, 0]
    return result


def build_mesh(nx: int, ny: int, poisson_ratio: float = POISSON) -> Mesh:
    xs = np.linspace(0.0, 1.0, nx + 1)
    ys = np.linspace(0.0, 0.1, ny + 1)
    xx, yy = np.meshgrid(xs, ys)
    points = np.column_stack((xx.ravel(), yy.ravel()))

    triangles: list[list[int]] = []
    for j in range(ny):
        for i in range(nx):
            n00 = j * (nx + 1) + i
            n10 = n00 + 1
            n01 = n00 + nx + 1
            n11 = n01 + 1
            triangles.extend(([n00, n10, n11], [n00, n11, n01]))
    tri = np.asarray(triangles, dtype=np.int64)

    gradients = np.empty((tri.shape[0], 3, 2), dtype=np.float64)
    areas = np.empty(tri.shape[0], dtype=np.float64)
    for element, nodes in enumerate(tri):
        x = points[nodes]
        dm = np.column_stack((x[1] - x[0], x[2] - x[0]))
        signed_double_area = float(np.linalg.det(dm))
        if signed_double_area <= 0.0:
            raise ValueError("mesh contains a non-positive reference triangle")
        areas[element] = 0.5 * signed_double_area
        inv_dm = np.linalg.inv(dm)
        gradients[element, 1] = inv_dm[0]
        gradients[element, 2] = inv_dm[1]
        gradients[element, 0] = -gradients[element, 1:].sum(axis=0)

    centroids = points[tri].mean(axis=1)
    in_smas = (centroids[:, 1] >= 0.04) & (centroids[:, 1] <= 0.06)
    in_muscle = in_smas & (centroids[:, 0] >= 0.06) & (centroids[:, 0] <= 0.22)
    material_id = np.zeros(tri.shape[0], dtype=np.uint8)
    material_id[in_smas] = 1
    material_id[in_muscle] = 2
    young = YOUNG_MPA[material_id]
    lame_mu = young / (2.0 * (1.0 + poisson_ratio))
    lame_lambda = (
        young * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    )

    element_dofs = np.empty((tri.shape[0], 6), dtype=np.int64)
    element_dofs[:, 0::2] = 2 * tri
    element_dofs[:, 1::2] = 2 * tri + 1
    fixed_nodes = np.flatnonzero(
        np.isclose(points[:, 1], 0.0)
        | np.isclose(points[:, 0], 0.0)
        | np.isclose(points[:, 0], 1.0)
    )
    fixed_dofs = np.concatenate((2 * fixed_nodes, 2 * fixed_nodes + 1))
    all_dofs = np.arange(2 * points.shape[0], dtype=np.int64)
    free_dofs = np.setdiff1d(all_dofs, fixed_dofs, assume_unique=False)
    free_lookup = np.full(all_dofs.size, -1, dtype=np.int64)
    free_lookup[free_dofs] = np.arange(free_dofs.size)

    local_free = free_lookup[element_dofs]
    element_index, local_row, local_col, rows, cols = [], [], [], [], []
    for element in range(tri.shape[0]):
        for i in range(6):
            if local_free[element, i] < 0:
                continue
            for j in range(6):
                if local_free[element, j] < 0:
                    continue
                element_index.append(element)
                local_row.append(i)
                local_col.append(j)
                rows.append(local_free[element, i])
                cols.append(local_free[element, j])

    muscle_elements = np.flatnonzero(material_id == 2)
    muscle_local = np.full(tri.shape[0], -1, dtype=np.int64)
    muscle_local[muscle_elements] = np.arange(muscle_elements.size)
    edge_owner: dict[tuple[int, int], int] = {}
    muscle_edges: list[tuple[int, int, float]] = []
    for local, element in enumerate(muscle_elements):
        nodes = tri[element]
        for a, b in ((nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[2], nodes[0])):
            edge = (min(int(a), int(b)), max(int(a), int(b)))
            if edge in edge_owner:
                other = edge_owner.pop(edge)
                length = float(np.linalg.norm(points[edge[1]] - points[edge[0]]))
                distance = float(
                    np.linalg.norm(
                        centroids[element] - centroids[muscle_elements[other]]
                    )
                )
                muscle_edges.append((other, local, length / distance))
            else:
                edge_owner[edge] = local

    top_nodes = np.flatnonzero(
        np.isclose(points[:, 1], 0.1)
        & ~np.isclose(points[:, 0], 0.0)
        & ~np.isclose(points[:, 0], 1.0)
    )
    return Mesh(
        nx=nx,
        ny=ny,
        points=points,
        triangles=tri,
        gradients=gradients,
        areas=areas,
        material_id=material_id,
        young=young,
        lame_lambda=lame_lambda,
        lame_mu=lame_mu,
        element_dofs=element_dofs,
        fixed_nodes=fixed_nodes,
        free_dofs=free_dofs,
        free_lookup=free_lookup,
        top_nodes=top_nodes,
        muscle_elements=muscle_elements,
        muscle_local=muscle_local,
        matrix_element=np.asarray(element_index, dtype=np.int64),
        matrix_local_row=np.asarray(local_row, dtype=np.int64),
        matrix_local_col=np.asarray(local_col, dtype=np.int64),
        matrix_rows=np.asarray(rows, dtype=np.int64),
        matrix_cols=np.asarray(cols, dtype=np.int64),
        muscle_edges=tuple(muscle_edges),
    )


def build_control_map(
    mesh: Mesh, variant: str, tied_nx: int, tied_ny: int
) -> ControlMap:
    n_muscle = mesh.muscle_elements.size
    if variant not in {"free", "tied", "regularized"}:
        raise ValueError(f"unknown variant {variant!r}")
    if variant == "tied":
        centroids = mesh.points[mesh.triangles[mesh.muscle_elements]].mean(axis=1)
        cell_x = np.minimum((centroids[:, 0] * tied_nx).astype(int), tied_nx - 1)
        cell_y = np.minimum((centroids[:, 1] / 0.1 * tied_ny).astype(int), tied_ny - 1)
        local_x = centroids[:, 0] * tied_nx - cell_x
        local_y = centroids[:, 1] / 0.1 * tied_ny - cell_y
        half = (local_y > local_x).astype(int)
        raw = 2 * (cell_y * tied_nx + cell_x) + half
        _, element_group = np.unique(raw, return_inverse=True)
    else:
        element_group = np.arange(n_muscle, dtype=np.int64)
    n_groups = int(element_group.max() + 1) if element_group.size else 0

    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    for left, right, weight in mesh.muscle_edges:
        for component in range(3):
            i = 3 * int(element_group[left]) + component
            j = 3 * int(element_group[right]) + component
            rows.extend((i, i, j, j))
            cols.extend((i, j, i, j))
            values.extend((weight, -weight, -weight, weight))
    size = 3 * n_groups
    regularizer = sp.coo_matrix((values, (rows, cols)), shape=(size, size)).tocsr()
    return ControlMap(variant, element_group, n_groups, regularizer)


def unpack_displacement(mesh: Mesh, u_free: np.ndarray) -> np.ndarray:
    u = np.zeros(2 * mesh.points.shape[0], dtype=np.float64)
    u[mesh.free_dofs] = u_free
    return u.reshape((-1, 2))


def element_activation(
    mesh: Mesh, controls: np.ndarray, cmap: ControlMap
) -> np.ndarray:
    activation = np.zeros((mesh.triangles.shape[0], 3), dtype=np.float64)
    if cmap.n_groups:
        activation[mesh.muscle_elements] = controls.reshape((-1, 3))[cmap.element_group]
    return activation


def constitutive(
    mesh: Mesh,
    u_free: np.ndarray,
    controls: np.ndarray,
    cmap: ControlMap,
    *,
    need_hessian: bool,
    need_mixed: bool,
) -> tuple[
    float,
    np.ndarray,
    sp.csc_matrix,
    sp.csc_matrix,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    u = unpack_displacement(mesh, u_free)
    x = mesh.points + u
    element_x = x[mesh.triangles]
    deformation = np.einsum("eia,eib->eab", element_x, mesh.gradients)
    activation = element_activation(mesh, controls, cmap)
    active_inv = np.broadcast_to(np.eye(2), deformation.shape).copy()
    active_inv[:, 0, 0] += activation[:, 0]
    active_inv[:, 1, 1] += activation[:, 1]
    active_inv[:, 0, 1] += activation[:, 2]
    active_inv[:, 1, 0] += activation[:, 2]
    elastic = deformation @ active_inv
    determinant = np.linalg.det(elastic)
    det_deformation = np.linalg.det(deformation)
    det_active_inv = np.linalg.det(active_inv)
    min_singular_active_inv = np.linalg.svd(active_inv, compute_uv=False)[:, -1]
    cof = cofactor_2d(elastic)
    kappa = -mesh.lame_mu + mesh.lame_lambda * (determinant - 1.0)
    p_elastic = mesh.lame_mu[:, None, None] * elastic + kappa[:, None, None] * cof
    p_deformation = p_elastic @ np.swapaxes(active_inv, 1, 2)
    density = (
        0.5 * mesh.lame_mu * (np.sum(elastic * elastic, axis=(1, 2)) - 2.0)
        - mesh.lame_mu * (determinant - 1.0)
        + 0.5 * mesh.lame_lambda * (determinant - 1.0) ** 2
    )
    energy = float(np.dot(mesh.areas, density))
    local_residual = mesh.areas[:, None, None] * np.einsum(
        "eab,eib->eia", p_deformation, mesh.gradients
    )
    residual = np.zeros(mesh.n_free, dtype=np.float64)
    local_free = mesh.free_lookup[mesh.element_dofs]
    mask = local_free >= 0
    np.add.at(residual, local_free[mask], local_residual.reshape((-1, 6))[mask])

    hessian = sp.csc_matrix((mesh.n_free, mesh.n_free), dtype=np.float64)
    if need_hessian:
        local_hessian = np.empty((mesh.triangles.shape[0], 6, 6), dtype=np.float64)
        for node in range(3):
            for component in range(2):
                column = 2 * node + component
                d_deformation = np.zeros_like(deformation)
                d_deformation[:, component, :] = mesh.gradients[:, node, :]
                d_elastic = d_deformation @ active_inv
                d_determinant = np.sum(cof * d_elastic, axis=(1, 2))
                d_p_elastic = (
                    mesh.lame_mu[:, None, None] * d_elastic
                    + mesh.lame_lambda[:, None, None]
                    * d_determinant[:, None, None]
                    * cof
                    + kappa[:, None, None] * cofactor_2d(d_elastic)
                )
                d_p_deformation = d_p_elastic @ np.swapaxes(active_inv, 1, 2)
                d_residual = mesh.areas[:, None, None] * np.einsum(
                    "eab,eib->eia", d_p_deformation, mesh.gradients
                )
                local_hessian[:, :, column] = d_residual.reshape((-1, 6))
        local_hessian = 0.5 * (local_hessian + np.swapaxes(local_hessian, 1, 2))
        values = local_hessian[
            mesh.matrix_element, mesh.matrix_local_row, mesh.matrix_local_col
        ]
        hessian = sp.coo_matrix(
            (values, (mesh.matrix_rows, mesh.matrix_cols)),
            shape=(mesh.n_free, mesh.n_free),
        ).tocsc()

    mixed = sp.csc_matrix((mesh.n_free, cmap.n_controls), dtype=np.float64)
    if need_mixed and cmap.n_controls:
        rows: list[int] = []
        cols: list[int] = []
        values: list[float] = []
        for local, element in enumerate(mesh.muscle_elements):
            group = int(cmap.element_group[local])
            f = deformation[element]
            a = active_inv[element]
            c = cof[element]
            p_g = p_elastic[element]
            for component, d_a in enumerate(ACTIVATION_BASES):
                d_g = f @ d_a
                d_j = float(np.sum(c * d_g))
                d_p_g = (
                    mesh.lame_mu[element] * d_g
                    + mesh.lame_lambda[element] * d_j * c
                    + kappa[element] * cofactor_2d(d_g)
                )
                d_p_f = d_p_g @ a.T + p_g @ d_a.T
                d_local = mesh.areas[element] * np.einsum(
                    "ab,ib->ia", d_p_f, mesh.gradients[element]
                )
                for local_dof, global_dof in enumerate(local_free[element]):
                    if global_dof >= 0:
                        rows.append(int(global_dof))
                        cols.append(3 * group + component)
                        values.append(float(d_local.reshape(6)[local_dof]))
        mixed = sp.coo_matrix(
            (values, (rows, cols)), shape=(mesh.n_free, cmap.n_controls)
        ).tocsc()
    return (
        energy,
        residual,
        hessian,
        mixed,
        det_deformation,
        determinant,
        det_active_inv,
        min_singular_active_inv,
        deformation,
        activation,
    )


def solve_linear(matrix: sp.csc_matrix, rhs: np.ndarray) -> np.ndarray:
    solution = spla.spsolve(matrix, rhs)
    if not np.all(np.isfinite(solution)):
        raise RuntimeError("sparse solve produced non-finite values")
    return np.asarray(solution)


def solve_forward(
    mesh: Mesh,
    controls: np.ndarray,
    cmap: ControlMap,
    initial: np.ndarray,
    tolerance: float,
    max_iterations: int,
) -> ForwardState:
    u = initial.copy()
    line_search_failures = 0
    converged = False
    iteration = -1
    for iteration in range(max_iterations):
        energy, residual, hessian, _, det_f, det_g, det_a, min_sv_a, _, _ = (
            constitutive(mesh, u, controls, cmap, need_hessian=True, need_mixed=False)
        )
        residual_norm = float(np.linalg.norm(residual) / math.sqrt(max(mesh.n_free, 1)))
        if residual_norm <= tolerance:
            converged = True
            break
        diagonal_scale = max(float(np.max(np.abs(hessian.diagonal()))), 1.0e-12)
        damping = 0.0
        direction: np.ndarray | None = None
        for _ in range(12):
            system = (
                hessian
                if damping == 0.0
                else hessian + damping * sp.eye(mesh.n_free, format="csc")
            )
            try:
                candidate = solve_linear(system, -residual)
            except Exception:  # noqa: BLE001
                candidate = np.full_like(residual, np.nan)
            if (
                np.all(np.isfinite(candidate))
                and float(np.dot(candidate, residual)) < 0.0
            ):
                direction = candidate
                break
            damping = diagonal_scale * (
                1.0e-9 if damping == 0.0 else 10.0 * damping / diagonal_scale
            )
        if direction is None:
            raise RuntimeError("could not find a descent Newton direction")
        directional = float(np.dot(residual, direction))
        step = 1.0
        accepted = False
        for _ in range(30):
            trial = u + step * direction
            trial_energy, *_ = constitutive(
                mesh, trial, controls, cmap, need_hessian=False, need_mixed=False
            )
            if (
                np.isfinite(trial_energy)
                and trial_energy <= energy + 1.0e-4 * step * directional
            ):
                u = trial
                accepted = True
                break
            step *= 0.5
        if not accepted:
            line_search_failures += 1
            break
    energy, residual, hessian, _, det_f, det_g, det_a, min_sv_a, _, _ = constitutive(
        mesh, u, controls, cmap, need_hessian=True, need_mixed=False
    )
    residual_norm = float(np.linalg.norm(residual) / math.sqrt(max(mesh.n_free, 1)))
    converged = converged or residual_norm <= tolerance
    return ForwardState(
        u_free=u,
        energy=energy,
        residual=residual,
        hessian=hessian,
        det_deformation=det_f,
        det_elastic=det_g,
        det_active_inv=det_a,
        min_singular_active_inv=min_sv_a,
        iterations=iteration + 1,
        converged=converged,
        line_search_failures=line_search_failures,
    )


def target_loss_and_gradient(
    mesh: Mesh, u_free: np.ndarray, target_y: float
) -> tuple[float, np.ndarray]:
    u = unpack_displacement(mesh, u_free)
    top = u[mesh.top_nodes]
    error = top.copy()
    error[:, 1] -= target_y
    # Match the current 3D experiment: the 2D residual is embedded as (ux, uy, 0)
    # and mean squared error is taken over all three vector components.
    loss = float(np.sum(error * error) / (3 * mesh.top_nodes.size))
    gradient = np.zeros(mesh.n_free, dtype=np.float64)
    scale = 2.0 / (3 * mesh.top_nodes.size)
    for node, node_error in zip(mesh.top_nodes, error, strict=True):
        for component in range(2):
            free = mesh.free_lookup[2 * node + component]
            if free >= 0:
                gradient[free] = scale * node_error[component]
    return loss, gradient


def inverse_evaluation(
    mesh: Mesh,
    controls: np.ndarray,
    cmap: ControlMap,
    u_initial: np.ndarray,
    cfg: Config,
) -> tuple[ForwardState, float, float, np.ndarray]:
    forward = solve_forward(
        mesh,
        controls,
        cmap,
        u_initial,
        cfg.forward_tolerance,
        cfg.forward_max_iterations,
    )
    loss_data, dloss_du = target_loss_and_gradient(mesh, forward.u_free, cfg.target_y)
    _, _, _, mixed, _, _, _, _, _, _ = constitutive(
        mesh,
        forward.u_free,
        controls,
        cmap,
        need_hessian=False,
        need_mixed=True,
    )
    adjoint = solve_linear(forward.hessian, -dloss_du)
    gradient = np.asarray(mixed.T @ adjoint).ravel()
    loss_regularizer = 0.0
    if cmap.variant == "regularized":
        q_controls = np.asarray(cmap.regularizer @ controls).ravel()
        loss_regularizer = (
            0.5 * cfg.regularization_weight * float(np.dot(controls, q_controls))
        )
        gradient += cfg.regularization_weight * q_controls
    return forward, loss_data, loss_regularizer, gradient


def top_metrics(
    mesh: Mesh, u_free: np.ndarray, target_y: float, filter_width: float
) -> dict[str, float]:
    u = unpack_displacement(mesh, u_free)
    top = u[mesh.top_nodes]
    x = mesh.points[mesh.top_nodes, 0]
    spacing = float(np.mean(np.diff(x))) if x.size > 1 else 1.0
    sigma = filter_width / spacing
    smooth = ndi.gaussian_filter1d(top[:, 1], sigma=max(sigma, 0.5), mode="nearest")
    highpass = top[:, 1] - smooth
    slope = np.gradient(top[:, 1], x)
    curvature = np.gradient(slope, x)
    return {
        "top_uy_mean": float(np.mean(top[:, 1])),
        "top_uy_range": float(np.ptp(top[:, 1])),
        "top_error_rms": float(
            np.sqrt(np.mean((top[:, 1] - target_y) ** 2 + top[:, 0] ** 2))
        ),
        "top_highpass_rms": float(np.sqrt(np.mean(highpass**2))),
        "top_slope_rms": float(np.sqrt(np.mean(slope**2))),
        "top_curvature_rms": float(np.sqrt(np.mean(curvature**2))),
    }


def control_metrics(
    mesh: Mesh, controls: np.ndarray, cmap: ControlMap
) -> dict[str, float]:
    per_element = controls.reshape((-1, 3))[cmap.element_group]
    jumps = [
        per_element[left] - per_element[right] for left, right, _ in mesh.muscle_edges
    ]
    jump_rms = float(np.sqrt(np.mean(np.square(jumps)))) if jumps else 0.0
    h1 = (
        float(controls @ (cmap.regularizer @ controls)) if cmap.regularizer.nnz else 0.0
    )
    return {
        "activation_l2_rms": float(np.sqrt(np.mean(controls**2)))
        if controls.size
        else 0.0,
        "activation_max_abs": float(np.max(np.abs(controls))) if controls.size else 0.0,
        "activation_neighbor_jump_rms": jump_rms,
        "activation_physical_h1_squared": h1,
    }


def make_grid(
    mesh: Mesh, controls: np.ndarray, cmap: ControlMap, forward: ForwardState, step: int
) -> pv.UnstructuredGrid:
    u = unpack_displacement(mesh, forward.u_free)
    points = np.column_stack((mesh.points, np.zeros(mesh.points.shape[0])))
    cells = np.column_stack(
        (np.full(mesh.triangles.shape[0], 3, dtype=np.int64), mesh.triangles)
    ).ravel()
    grid = pv.UnstructuredGrid(
        cells,
        np.full(mesh.triangles.shape[0], pv.CellType.TRIANGLE, dtype=np.uint8),
        points,
    )
    activation = element_activation(mesh, controls, cmap)
    grid.point_data["Displacement"] = np.column_stack((u, np.zeros(u.shape[0])))
    grid.point_data["DisplacementY"] = u[:, 1]
    grid.point_data["DisplacementMagnitude"] = np.linalg.norm(u, axis=1)
    grid.cell_data["MaterialId"] = mesh.material_id
    grid.cell_data["YoungModulusMPa"] = mesh.young
    poisson = mesh.young / (2.0 * mesh.lame_mu) - 1.0
    grid.cell_data["PoissonRatio"] = poisson
    grid.cell_data["DetF"] = forward.det_deformation
    grid.cell_data["DetG"] = forward.det_elastic
    grid.cell_data["DetAinv"] = forward.det_active_inv
    grid.cell_data["MinSingularAinv"] = forward.min_singular_active_inv
    grid.cell_data["ActivationXX"] = activation[:, 0]
    grid.cell_data["ActivationYY"] = activation[:, 1]
    grid.cell_data["ActivationXY"] = activation[:, 2]
    grid.cell_data["ActivationNorm"] = np.linalg.norm(activation, axis=1)
    grid.cell_data["MuscleMask"] = (mesh.material_id == 2).astype(np.uint8)
    grid.field_data["InverseStep"] = np.array([step], dtype=np.int64)
    return grid


def write_profile_and_spectrum(
    case_dir: Path,
    mesh: Mesh,
    forward: ForwardState,
    target_y: float,
    *,
    profile_name: str = "top-profile.csv",
    spectrum_name: str = "top-spectrum.csv",
) -> None:
    u = unpack_displacement(mesh, forward.u_free)
    x = mesh.points[mesh.top_nodes, 0]
    ux = u[mesh.top_nodes, 0]
    uy = u[mesh.top_nodes, 1]
    smooth = ndi.gaussian_filter1d(
        uy, sigma=max(0.02 / float(np.mean(np.diff(x))), 0.5), mode="nearest"
    )
    with (case_dir / profile_name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("x", "ux", "uy", "target_uy", "smooth_uy", "highpass_uy"))
        writer.writerows(
            zip(x, ux, uy, np.full_like(x, target_y), smooth, uy - smooth, strict=True)
        )
    spacing = float(np.mean(np.diff(x)))
    centered = uy - np.mean(uy)
    frequency = np.fft.rfftfreq(uy.size, spacing)
    power = np.abs(np.fft.rfft(centered)) ** 2 / max(uy.size**2, 1)
    power = np.maximum(power, np.finfo(np.float64).tiny)
    with (case_dir / spectrum_name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("cycles_per_unit_length", "wavenumber", "power"))
        writer.writerows(
            zip(frequency[1:], 2.0 * np.pi * frequency[1:], power[1:], strict=True)
        )


def run_inverse(
    mesh: Mesh, cmap: ControlMap, case_dir: Path, cfg: Config
) -> dict[str, Any]:
    case_dir.mkdir(parents=True, exist_ok=False)
    frames_dir = case_dir / "frames"
    frames_dir.mkdir()
    controls = np.zeros(cmap.n_controls, dtype=np.float64)
    first_moment = np.zeros_like(controls)
    second_moment = np.zeros_like(controls)
    u = np.zeros(mesh.n_free, dtype=np.float64)
    trace: list[dict[str, Any]] = []
    series: list[dict[str, float | str]] = []
    best_loss = math.inf
    plateau_loss = math.inf
    best_step = 0
    best_valid_loss = math.inf
    best_valid_step = 0
    best_prefix_loss = math.inf
    best_prefix_step = 0
    stale = 0
    best_payload: tuple[np.ndarray, ForwardState] | None = None
    best_valid_payload: tuple[np.ndarray, ForwardState] | None = None
    best_prefix_payload: tuple[np.ndarray, ForwardState] | None = None
    first_invalid_step: int | None = None

    for step in range(cfg.max_steps + 1):
        forward, loss_data, loss_reg, gradient = inverse_evaluation(
            mesh, controls, cmap, u, cfg
        )
        u = forward.u_free
        total = loss_data + loss_reg
        orientation_preserving = bool(
            np.min(forward.det_deformation) > 0.0
            and np.min(forward.det_elastic) > 0.0
            and np.min(forward.det_active_inv) > 0.0
        )
        residual_rms = float(
            np.linalg.norm(forward.residual) / math.sqrt(max(mesh.n_free, 1))
        )
        numerically_equilibrated = bool(
            forward.converged and residual_rms <= cfg.forward_tolerance
        )
        verified_admissible = bool(orientation_preserving and numerically_equilibrated)
        row: dict[str, Any] = {
            "step": step,
            "objective_total": total,
            "objective_data": loss_data,
            "objective_regularizer": loss_reg,
            "gradient_rms": float(
                np.linalg.norm(gradient) / math.sqrt(max(gradient.size, 1))
            ),
            "equilibrium_residual_rms": residual_rms,
            "forward_iterations": forward.iterations,
            "forward_converged": int(forward.converged),
            "numerically_equilibrated": int(numerically_equilibrated),
            "min_det_f": float(np.min(forward.det_deformation)),
            "min_det_g": float(np.min(forward.det_elastic)),
            "min_det_ainv": float(np.min(forward.det_active_inv)),
            "min_singular_ainv": float(np.min(forward.min_singular_active_inv)),
            "orientation_preserving": int(orientation_preserving),
            "verified_admissible": int(verified_admissible),
            **top_metrics(mesh, u, cfg.target_y, cfg.bump_filter_width),
            **control_metrics(mesh, controls, cmap),
        }
        trace.append(row)
        cherries.set_step(step)
        cherries.log_metrics(
            {
                f"{mesh.nx}x{mesh.ny}/{cmap.variant}/objective": total,
                f"{mesh.nx}x{mesh.ny}/{cmap.variant}/equilibrium-residual-rms": residual_rms,
                f"{mesh.nx}x{mesh.ny}/{cmap.variant}/top-highpass-rms": row[
                    "top_highpass_rms"
                ],
            }
        )
        if step % cfg.save_every == 0 or step == cfg.max_steps:
            frame = frames_dir / f"step-{step:04d}.vtu"
            make_grid(mesh, controls, cmap, forward, step).save(frame)
            series.append(
                {"name": str(frame.relative_to(case_dir)), "time": float(step)}
            )
        if total < best_loss:
            best_loss = total
            best_step = step
            best_payload = (controls.copy(), forward)
        if orientation_preserving and total < best_valid_loss:
            best_valid_loss = total
            best_valid_step = step
            best_valid_payload = (controls.copy(), forward)
        if (
            first_invalid_step is None
            and verified_admissible
            and total < best_prefix_loss
        ):
            best_prefix_loss = total
            best_prefix_step = step
            best_prefix_payload = (controls.copy(), forward)
        if not orientation_preserving and first_invalid_step is None:
            first_invalid_step = step
        if total < plateau_loss - cfg.minimum_delta:
            plateau_loss = total
            stale = 0
        else:
            stale += 1
        if step == cfg.max_steps or (step > 0 and stale >= cfg.patience):
            break
        first_moment = 0.9 * first_moment + 0.1 * gradient
        second_moment = 0.999 * second_moment + 0.001 * gradient**2
        mhat = first_moment / (1.0 - 0.9 ** (step + 1))
        vhat = second_moment / (1.0 - 0.999 ** (step + 1))
        controls = controls - cfg.learning_rate * mhat / (np.sqrt(vhat) + 1.0e-8)

    if best_payload is None:
        raise RuntimeError("inverse run did not produce a best state")
    if best_valid_payload is None:
        raise RuntimeError(
            "inverse run did not produce an orientation-preserving state"
        )
    if best_prefix_payload is None:
        raise RuntimeError("inverse run did not produce an admissible-prefix state")
    best_controls, best_forward = best_payload
    best_valid_controls, best_valid_forward = best_valid_payload
    best_prefix_controls, best_prefix_forward = best_prefix_payload
    write_json(
        case_dir / "history.vtu.series", {"file-series-version": "1.0", "files": series}
    )
    with (case_dir / "trace.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(trace[0]))
        writer.writeheader()
        writer.writerows(trace)
    np.savez_compressed(
        case_dir / "best-state.npz",
        controls=best_controls,
        displacement_free=best_forward.u_free,
        best_step=np.array([best_step], dtype=np.int64),
    )
    np.savez_compressed(
        case_dir / "best-valid-state.npz",
        controls=best_valid_controls,
        displacement_free=best_valid_forward.u_free,
        best_step=np.array([best_valid_step], dtype=np.int64),
    )
    np.savez_compressed(
        case_dir / "best-admissible-prefix-state.npz",
        controls=best_prefix_controls,
        displacement_free=best_prefix_forward.u_free,
        best_step=np.array([best_prefix_step], dtype=np.int64),
    )
    best_grid = make_grid(mesh, best_controls, cmap, best_forward, best_step)
    best_grid.save(case_dir / "best.vtu")
    best_grid.save(case_dir / "final.vtu")
    write_profile_and_spectrum(case_dir, mesh, best_forward, cfg.target_y)
    make_grid(
        mesh,
        best_valid_controls,
        cmap,
        best_valid_forward,
        best_valid_step,
    ).save(case_dir / "best-valid.vtu")
    write_profile_and_spectrum(
        case_dir,
        mesh,
        best_valid_forward,
        cfg.target_y,
        profile_name="top-profile-best-valid.csv",
        spectrum_name="top-spectrum-best-valid.csv",
    )
    make_grid(
        mesh,
        best_prefix_controls,
        cmap,
        best_prefix_forward,
        best_prefix_step,
    ).save(case_dir / "best-admissible-prefix.vtu")
    write_profile_and_spectrum(
        case_dir,
        mesh,
        best_prefix_forward,
        cfg.target_y,
        profile_name="top-profile-best-admissible-prefix.csv",
        spectrum_name="top-spectrum-best-admissible-prefix.csv",
    )
    best_row = trace[best_step]
    best_valid_row = trace[best_valid_step]
    best_prefix_row = trace[best_prefix_step]
    return {
        "resolution": [mesh.nx, mesh.ny],
        "variant": cmap.variant,
        "n_nodes": int(mesh.points.shape[0]),
        "n_triangles": int(mesh.triangles.shape[0]),
        "n_muscle_triangles": int(mesh.muscle_elements.size),
        "n_control_groups": cmap.n_groups,
        "n_activation_dofs": cmap.n_controls,
        "n_observed_components": int(3 * mesh.top_nodes.size),
        "best_step": best_step,
        "best_valid_step": best_valid_step,
        "best_admissible_prefix_step": best_prefix_step,
        "first_invalid_step": first_invalid_step,
        "evaluations": len(trace),
        "stop_reason": "patience" if stale >= cfg.patience else "maximum-steps",
        "best": best_row,
        "best_valid": best_valid_row,
        "best_admissible_prefix": best_prefix_row,
        "best_admissible_prefix_is_verified": bool(
            best_prefix_row["verified_admissible"]
        ),
        "global_best_is_orientation_preserving": bool(
            best_row["orientation_preserving"]
        ),
        "all_states_orientation_preserving": first_invalid_step is None,
        "best_valid_objective_gap_to_global_best": float(
            best_valid_row["objective_total"] - best_row["objective_total"]
        ),
        "best_admissible_prefix_objective_gap_to_global_best": float(
            best_prefix_row["objective_total"] - best_row["objective_total"]
        ),
        "all_forward_solves_converged": bool(
            all(row["forward_converged"] for row in trace)
        ),
        "all_pre_inversion_forward_solves_equilibrated": bool(
            all(
                row["numerically_equilibrated"]
                for row in trace
                if first_invalid_step is None or row["step"] < first_invalid_step
            )
        ),
        "maximum_equilibrium_residual_rms": float(
            max(row["equilibrium_residual_rms"] for row in trace)
        ),
        "minimum_det_f_over_history": float(min(row["min_det_f"] for row in trace)),
        "minimum_det_g_over_history": float(min(row["min_det_g"] for row in trace)),
        "minimum_det_ainv_over_history": float(
            min(row["min_det_ainv"] for row in trace)
        ),
        "minimum_singular_ainv_over_history": float(
            min(row["min_singular_ainv"] for row in trace)
        ),
        "paths": {
            "series": "history.vtu.series",
            "trace": "trace.csv",
            "best_vtu": "best.vtu",
            "final_vtu": "final.vtu",
            "best_valid_vtu": "best-valid.vtu",
            "best_valid_state": "best-valid-state.npz",
            "best_admissible_prefix_vtu": "best-admissible-prefix.vtu",
            "best_admissible_prefix_state": "best-admissible-prefix-state.npz",
            "profile": "top-profile.csv",
            "spectrum": "top-spectrum.csv",
            "best_valid_profile": "top-profile-best-valid.csv",
            "best_valid_spectrum": "top-spectrum-best-valid.csv",
            "best_admissible_prefix_profile": "top-profile-best-admissible-prefix.csv",
            "best_admissible_prefix_spectrum": "top-spectrum-best-admissible-prefix.csv",
        },
    }


def element_derivative_validation(poisson_ratio: float) -> dict[str, float | bool]:
    # This deliberately exercises a deformed active muscle element away from identity.
    mu = 0.03 / (2.0 * (1.0 + poisson_ratio))
    lame = 0.03 * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio))
    f = np.array([[1.08, 0.13], [-0.04, 0.93]])
    alpha = np.array([0.07, -0.03, 0.025])
    a = np.eye(2) + np.tensordot(alpha, ACTIVATION_BASES, axes=1)
    direction_f = np.array([[0.17, -0.11], [0.09, 0.05]])
    direction_a = (
        ACTIVATION_BASES[0] - 0.4 * ACTIVATION_BASES[1] + 0.3 * ACTIVATION_BASES[2]
    )

    def energy(ff: np.ndarray, aa: np.ndarray) -> float:
        g = ff @ aa
        j = float(np.linalg.det(g))
        return (
            0.5 * mu * (float(np.sum(g * g)) - 2.0)
            - mu * (j - 1.0)
            + 0.5 * lame * (j - 1.0) ** 2
        )

    def stress(ff: np.ndarray, aa: np.ndarray) -> np.ndarray:
        g = ff @ aa
        j = float(np.linalg.det(g))
        p_g = mu * g + (-mu + lame * (j - 1.0)) * cofactor_2d(g)
        return p_g @ aa.T

    def tangent(ff: np.ndarray, aa: np.ndarray, dff: np.ndarray) -> np.ndarray:
        g = ff @ aa
        dg = dff @ aa
        c = cofactor_2d(g)
        j = float(np.linalg.det(g))
        dj = float(np.sum(c * dg))
        dpg = mu * dg + lame * dj * c + (-mu + lame * (j - 1.0)) * cofactor_2d(dg)
        return dpg @ aa.T

    def mixed(ff: np.ndarray, aa: np.ndarray, daa: np.ndarray) -> np.ndarray:
        g = ff @ aa
        dg = ff @ daa
        c = cofactor_2d(g)
        j = float(np.linalg.det(g))
        dj = float(np.sum(c * dg))
        pg = mu * g + (-mu + lame * (j - 1.0)) * c
        dpg = mu * dg + lame * dj * c + (-mu + lame * (j - 1.0)) * cofactor_2d(dg)
        return dpg @ aa.T + pg @ daa.T

    eps = 1.0e-6
    p = stress(f, a)
    fd_energy = (
        energy(f + eps * direction_f, a) - energy(f - eps * direction_f, a)
    ) / (2 * eps)
    analytic_energy = float(np.sum(p * direction_f))
    fd_tangent = (
        stress(f + eps * direction_f, a) - stress(f - eps * direction_f, a)
    ) / (2 * eps)
    analytic_tangent = tangent(f, a, direction_f)
    fd_mixed = (stress(f, a + eps * direction_a) - stress(f, a - eps * direction_a)) / (
        2 * eps
    )
    analytic_mixed = mixed(f, a, direction_a)

    def relative(actual: np.ndarray | float, expected: np.ndarray | float) -> float:
        return float(
            np.linalg.norm(np.asarray(actual) - np.asarray(expected))
            / max(np.linalg.norm(np.asarray(expected)), 1.0e-14)
        )

    errors = {
        "energy_gradient_relative_error": relative(analytic_energy, fd_energy),
        "hessian_action_relative_error": relative(analytic_tangent, fd_tangent),
        "mixed_activation_derivative_relative_error": relative(
            analytic_mixed, fd_mixed
        ),
    }
    return {**errors, "passed": bool(max(errors.values()) < 1.0e-7)}


def assembled_adjoint_validation(cfg: Config) -> dict[str, float | int | bool]:
    mesh = build_mesh(50, 5, cfg.poisson_ratio)
    cmap = build_control_map(mesh, "free", cfg.tied_nx, cfg.tied_ny)
    controls = 0.003 * np.sin(0.37 * np.arange(cmap.n_controls))
    forward, loss, _, gradient = inverse_evaluation(
        mesh, controls, cmap, np.zeros(mesh.n_free), cfg
    )
    component = int(np.argmax(np.abs(gradient)))
    epsilon = 1.0e-5
    losses: list[float] = []
    residuals: list[float] = []
    for sign in (-1.0, 1.0):
        perturbed = controls.copy()
        perturbed[component] += sign * epsilon
        state = solve_forward(
            mesh,
            perturbed,
            cmap,
            forward.u_free,
            cfg.forward_tolerance,
            cfg.forward_max_iterations,
        )
        perturbed_loss, _ = target_loss_and_gradient(mesh, state.u_free, cfg.target_y)
        losses.append(perturbed_loss)
        residuals.append(
            float(np.linalg.norm(state.residual) / math.sqrt(max(mesh.n_free, 1)))
        )
    finite_difference = (losses[1] - losses[0]) / (2.0 * epsilon)
    analytic = float(gradient[component])
    relative_error = abs(analytic - finite_difference) / max(
        abs(analytic), abs(finite_difference), 1.0e-14
    )
    return {
        "resolution_nx": mesh.nx,
        "resolution_ny": mesh.ny,
        "activation_component": component,
        "base_loss": loss,
        "analytic_gradient": analytic,
        "finite_difference_gradient": finite_difference,
        "relative_error": relative_error,
        "maximum_equilibrium_residual_rms": max(residuals),
        "passed": bool(
            relative_error < 1.0e-4 and max(residuals) <= 10.0 * cfg.forward_tolerance
        ),
    }


def tangent_identifiability_certificate(
    mesh: Mesh, cfg: Config
) -> tuple[dict[str, Any], np.ndarray]:
    cmap = build_control_map(mesh, "free", cfg.tied_nx, cfg.tied_ny)
    controls = np.zeros(cmap.n_controls, dtype=np.float64)
    u = np.zeros(mesh.n_free, dtype=np.float64)
    _, residual, hessian, mixed, *_ = constitutive(
        mesh, u, controls, cmap, need_hessian=True, need_mixed=True
    )
    top_free = np.array(
        [
            mesh.free_lookup[2 * node + component]
            for node in mesh.top_nodes
            for component in range(2)
        ],
        dtype=np.int64,
    )
    factor = spla.splu(hessian)
    response = np.empty((top_free.size, cmap.n_controls), dtype=np.float64)
    for start in range(0, cmap.n_controls, 64):
        stop = min(start + 64, cmap.n_controls)
        du = factor.solve(-mixed[:, start:stop].toarray())
        response[:, start:stop] = du[top_free]
    singular_values = np.linalg.svd(response, compute_uv=False)
    largest = float(singular_values[0]) if singular_values.size else 0.0
    rank_tolerance = (
        largest * max(response.shape) * np.finfo(np.float64).eps
        if largest > 0.0
        else 0.0
    )
    rank = int(np.count_nonzero(singular_values > rank_tolerance))
    smallest_nonzero = float(singular_values[rank - 1]) if rank else 0.0
    target = np.tile(np.array([0.0, cfg.target_y]), mesh.top_nodes.size)
    coefficients, *_ = np.linalg.lstsq(response, target, rcond=None)
    projection = response @ coefficients
    projection_residual = float(
        np.linalg.norm(projection - target) / max(np.linalg.norm(target), 1.0e-30)
    )
    weak_threshold = largest * 1.0e-6
    return (
        {
            "scope": (
                "local tangent at undeformed zero-activation equilibrium; this is an "
                "identifiability certificate, not a global nonlinear unreachability proof"
            ),
            "resolution": [mesh.nx, mesh.ny],
            "top_in_plane_output_components": int(top_free.size),
            "top_loss_components_after_3d_embedding": int(3 * mesh.top_nodes.size),
            "muscle_triangles": int(mesh.muscle_elements.size),
            "activation_dofs": cmap.n_controls,
            "numerical_rank": rank,
            "control_nullity_lower_bound": max(cmap.n_controls - rank, 0),
            "output_cokernel_dimension": max(top_free.size - rank, 0),
            "rank_tolerance": rank_tolerance,
            "largest_singular_value": largest,
            "smallest_nonzero_singular_value": smallest_nonzero,
            "nonzero_condition_number": (
                largest / smallest_nonzero if smallest_nonzero > 0.0 else None
            ),
            "weak_responsive_singular_values_below_1e-6_relative": int(
                np.count_nonzero(
                    (singular_values > rank_tolerance)
                    & (singular_values < weak_threshold)
                )
            ),
            "projection_residual_fraction_of_target_l2": projection_residual,
            "equilibrium_residual_rms": float(
                np.linalg.norm(residual) / math.sqrt(max(mesh.n_free, 1))
            ),
        },
        singular_values,
    )


def validate_config(cfg: Config) -> tuple[list[tuple[int, int]], list[str]]:
    if cfg.output_dir.exists() and any(cfg.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output {cfg.output_dir}")
    resolutions = parse_resolutions(cfg.resolutions)
    variants = [item.strip() for item in cfg.variants.split(",") if item.strip()]
    if not variants or any(
        item not in {"free", "tied", "regularized"} for item in variants
    ):
        raise ValueError(
            "variants must be a comma-separated subset of free,tied,regularized"
        )
    if cfg.max_steps < 1 or cfg.patience < 1 or cfg.save_every < 1:
        raise ValueError("step, patience, and save intervals must be positive")
    if cfg.learning_rate <= 0.0 or cfg.regularization_weight < 0.0:
        raise ValueError(
            "learning rate must be positive and regularization nonnegative"
        )
    if not 0.0 <= cfg.poisson_ratio < 0.5:
        raise ValueError("poisson ratio must satisfy 0 <= nu < 0.5")
    return resolutions, variants


def main(cfg: Config) -> None:
    started = time.perf_counter()
    resolutions, variants = validate_config(cfg)
    if cfg.output_dir.exists() and any(cfg.output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output {cfg.output_dir}")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    validation: dict[str, Any]
    if cfg.validate_derivatives:
        element = element_derivative_validation(cfg.poisson_ratio)
        assembled = assembled_adjoint_validation(cfg)
        validation = {
            "element": element,
            "assembled_implicit_adjoint": assembled,
            "passed": bool(element["passed"] and assembled["passed"]),
        }
    else:
        validation = {"skipped": True, "passed": True}
    if cfg.validate_derivatives and not validation["passed"]:
        raise RuntimeError(f"analytic derivative validation failed: {validation}")
    cases: list[dict[str, Any]] = []
    certificates: list[dict[str, Any]] = []
    for nx, ny in resolutions:
        mesh = build_mesh(nx, ny, cfg.poisson_ratio)
        logger.info(
            "resolution %dx%d: %d triangles, %d muscle triangles, %d free dofs",
            nx,
            ny,
            mesh.triangles.shape[0],
            mesh.muscle_elements.size,
            mesh.n_free,
        )
        certificate, singular_values = tangent_identifiability_certificate(mesh, cfg)
        certificates.append(certificate)
        spectrum_path = cfg.output_dir / f"tangent-{nx}x{ny}-singular-values.csv"
        with spectrum_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("index", "singular_value", "relative_to_largest"))
            largest = float(singular_values[0]) if singular_values.size else 1.0
            writer.writerows(
                (index, value, value / largest)
                for index, value in enumerate(singular_values)
            )
        certificate["singular_values_csv"] = spectrum_path.name
        for variant in variants:
            cmap = build_control_map(mesh, variant, cfg.tied_nx, cfg.tied_ny)
            case_name = f"{nx}x{ny}-{variant}"
            logger.info(
                "running %s with %d activation DoFs", case_name, cmap.n_controls
            )
            cases.append(run_inverse(mesh, cmap, cfg.output_dir / case_name, cfg))

    execution_checks = {
        "derivative_validations_pass": bool(validation.get("passed", True)),
        "all_comparison_states_verified_admissible": bool(
            all(case["best_admissible_prefix_is_verified"] for case in cases)
        ),
        "three_dofs_per_free_muscle_triangle": bool(
            all(
                case["n_activation_dofs"] == 3 * case["n_muscle_triangles"]
                for case in cases
                if case["variant"] in {"free", "regularized"}
            )
        ),
    }
    solver_diagnostics = {
        "all_forward_solves_converged": bool(
            all(case["all_forward_solves_converged"] for case in cases)
        ),
        "all_equilibrium_residuals_below_tolerance": bool(
            all(
                case["maximum_equilibrium_residual_rms"] <= cfg.forward_tolerance
                for case in cases
            )
        ),
        "all_pre_inversion_forward_solves_equilibrated": bool(
            all(case["all_pre_inversion_forward_solves_equilibrated"] for case in cases)
        ),
        "interpretation": (
            "False values identify near-singular or inverted transition evaluations; "
            "they do not invalidate the separately verified comparison states."
        ),
    }
    configuration_checks = {
        "primary_poisson_ratio_is_0_49": bool(cfg.poisson_ratio == 0.49),
    }
    validity_checks = {
        "all_states_orientation_preserving": bool(
            all(case["all_states_orientation_preserving"] for case in cases)
        ),
        "all_global_best_states_orientation_preserving": bool(
            all(case["global_best_is_orientation_preserving"] for case in cases)
        ),
        "all_deformation_determinants_positive": bool(
            all(case["minimum_det_f_over_history"] > 0.0 for case in cases)
        ),
        "all_elastic_determinants_positive": bool(
            all(case["minimum_det_g_over_history"] > 0.0 for case in cases)
        ),
        "all_active_inverse_determinants_positive": bool(
            all(case["minimum_det_ainv_over_history"] > 0.0 for case in cases)
        ),
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "design": "exact-plane-strain-stable-neo-hookean-active-resolution-study",
        "complete": all(execution_checks.values())
        and all(configuration_checks.values()),
        "scope": (
            "Self-contained 2D triangle implementation of the exact plane-strain "
            "restriction of Apple's StableNeoHookean and StableNeoHookeanActive "
            "energies. It matches nonlinear-equilibrium and implicit-adjoint inverse "
            "semantics, but is not executed through Apple's tetrahedron-only Warp FEM."
        ),
        "energy": "0.5*mu*(||G||_F^2-2)-mu*(det(G)-1)+0.5*lambda*(det(G)-1)^2",
        "active_kinematics": "G=F@Ainv; Ainv=I+[[a0,a3],[a3,a1]]",
        "loss": "mean squared residual after embedding each 2D top displacement as (ux,uy,0)",
        "inverse": {
            "forward": "damped Newton with Armijo line search and exact sparse tangent",
            "gradient": "exact implicit adjoint H p=-dL/du; dL/da=(dr/da)^T p",
            "optimizer": "Adam",
            "learning_rate": cfg.learning_rate,
            "activation_bounds": None,
            "baseline_activation_smoothing": None,
            "max_steps": cfg.max_steps,
            "patience": cfg.patience,
            "minimum_delta": cfg.minimum_delta,
            "physical_comparison_state": (
                "minimum objective among orientation-preserving, strictly "
                "equilibrated evaluations before the first orientation failure"
            ),
        },
        "geometry": {
            "domain": [1.0, 0.1],
            "SMAS_y": [0.04, 0.06],
            "muscle_x": [0.06, 0.22],
            "muscle_y": [0.04, 0.06],
            "fixed": "bottom, left, and right; both in-plane components",
            "target": [0.0, cfg.target_y, 0.0],
        },
        "materials": {
            name: {
                "young_modulus_MPa": float(young),
                "poisson_ratio": cfg.poisson_ratio,
            }
            for name, young in zip(MATERIAL_NAMES, YOUNG_MPA, strict=True)
        },
        "control_variants": {
            "free": "one unrestricted symmetric in-plane Ainv tensor (3 DoFs) per muscle triangle",
            "tied": "fine muscle triangles tied to a fixed physical coarse-triangle partition",
            "regularized": (
                "free per-triangle controls plus 0.5*weight*sum_edges "
                "(|edge|/centroid_distance)*||a_left-a_right||^2"
            ),
            "regularization_weight": cfg.regularization_weight,
        },
        "derivative_validation": validation,
        "tangent_identifiability": certificates,
        "checks": execution_checks,
        "solver_diagnostics": solver_diagnostics,
        "configuration_checks": configuration_checks,
        "validity_checks": validity_checks,
        "cases": cases,
        "elapsed_seconds": float(time.perf_counter() - started),
        "runtime": {
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "pyvista": pv.__version__,
        },
    }
    write_json(cfg.output_dir / "summary.json", summary)
    logger.info("wrote %s (complete=%s)", cfg.output_dir, summary["complete"])


if __name__ == "__main__":
    cherries.main(main)
