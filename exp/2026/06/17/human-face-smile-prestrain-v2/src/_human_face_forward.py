from __future__ import annotations

from typing import Any, cast, override

import attrs
import numpy as np
import pyvista as pv
import torch
from _human_face_config import (
    ADJOINT_ATOL,
    ADJOINT_MAXITER,
    ADJOINT_RTOL,
    APONEUROSIS_E,
    APONEUROSIS_FRACTION,
    APONEUROSIS_NU,
    FAT_E,
    FAT_FRACTION,
    FAT_NU,
    FORWARD_ATOL,
    FORWARD_MAX_STEPS,
    FORWARD_RTOL,
    MUSCLE_E,
    MUSCLE_FRACTION,
    MUSCLE_NU,
    SKIN_THICKNESS,
    InverseCase,
    InverseConfig,
)
from _human_face_mesh import lame_parameters
from _human_face_skin import skin_for_case
from liblaf.peach.linalg import FallbackSolver
from liblaf.peach.linalg.base import BaseProblem, Problem, Result
from liblaf.peach.linalg.cupy import CupyCG, CupyMinRes


@attrs.define
class SuccessPreferredFallbackSolver(FallbackSolver):
    @override
    def compute(self, problem: BaseProblem, state: Any) -> Result:
        problem = cast("Problem", problem)
        absolute_residuals = []
        relative_residuals = []
        success_index: int | None = None
        for index, solver in enumerate(self.solvers):
            solution = solver.solve(problem, state.init_params)
            state.solutions.append(solution)
            absolute_residual = torch.linalg.vector_norm(
                problem.matvec(solution.state.params) - problem.b
            )
            relative_residual = absolute_residual / torch.linalg.vector_norm(problem.b)
            absolute_residuals.append(absolute_residual)
            relative_residuals.append(relative_residual)
            if solution.success:
                success_index = index
                break

        state.absolute_residuals = torch.as_tensor(absolute_residuals)
        state.relative_residuals = torch.as_tensor(relative_residuals)
        if success_index is None:
            state.best_index = torch.argmin(state.absolute_residuals)
        else:
            state.best_index = torch.as_tensor(success_index, dtype=torch.int32)
        return state.result


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


def build_forward(
    mesh: pv.UnstructuredGrid, case: InverseCase, *, area_ratio_floor: float
) -> tuple[Any, pv.PolyData | None, dict[str, Any]]:
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

    skin = None
    skin_metrics: dict[str, Any] = {
        "skin/enabled": bool(case.skin_enabled),
        "skin/prestrain_enabled": bool(case.skin_prestrain_enabled),
    }
    if case.skin_enabled:
        skin, skin_metrics = skin_for_case(
            mesh, case, area_ratio_floor=area_ratio_floor
        )
        skin_metrics = {
            **skin_metrics,
            "skin/enabled": True,
            "skin/prestrain_enabled": bool(case.skin_prestrain_enabled),
        }
        builder.add_potential(
            Koiter.from_pyvista(skin, name="skin", thickness=SKIN_THICKNESS)
        )

    forward = Forward(builder.finalize())
    forward.optimizer = forward.default_optimizer(
        max_steps=FORWARD_MAX_STEPS,
        atol=FORWARD_ATOL,
        rtol=FORWARD_RTOL,
    )
    return forward, skin, skin_metrics


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
    cfg: InverseConfig,
    active_ids: np.ndarray,
    n_cells: int,
    cell_source_ids: np.ndarray | None = None,
) -> np.ndarray:
    if cfg.initial_activation_mesh is None:
        return np.zeros((active_ids.size, 6), dtype=np.float64)

    from liblaf.apple.common import ACTIVATION_INV

    mesh = pv.read(cfg.initial_activation_mesh)
    activation_inv = np.asarray(mesh.cell_data[ACTIVATION_INV.vtk], dtype=np.float64)
    if activation_inv.shape == (n_cells, 6):
        return activation_inv[active_ids].copy()
    if (
        cell_source_ids is not None
        and activation_inv.ndim == 2
        and activation_inv.shape[1] == 6
        and int(cell_source_ids.max()) < activation_inv.shape[0]
    ):
        return activation_inv[cell_source_ids[active_ids]].copy()
    expected = f"({n_cells}, 6)"
    if cell_source_ids is not None:
        expected = f"{expected} or a source mesh with at least {int(cell_source_ids.max()) + 1} cells"
    if activation_inv.shape != (n_cells, 6):
        msg = (
            f"{cfg.initial_activation_mesh} {ACTIVATION_INV.vtk} must have shape "
            f"{expected}, got {activation_inv.shape}"
        )
        raise ValueError(msg)
    msg = "unreachable"
    raise AssertionError(msg)


def initial_forward_displacement(
    cfg: InverseConfig, n_points: int, point_source_ids: np.ndarray | None = None
) -> np.ndarray | None:
    if cfg.initial_activation_mesh is None:
        return None

    mesh = pv.read(cfg.initial_activation_mesh)
    if "Displacement" not in mesh.point_data:
        return None
    displacement = np.asarray(mesh.point_data["Displacement"], dtype=np.float64)
    if displacement.shape == (n_points, 3):
        return displacement.copy()
    if (
        point_source_ids is not None
        and displacement.ndim == 2
        and displacement.shape[1] == 3
        and int(point_source_ids.max()) < displacement.shape[0]
    ):
        return displacement[point_source_ids].copy()
    expected = f"({n_points}, 3)"
    if point_source_ids is not None:
        expected = f"{expected} or a source mesh with at least {int(point_source_ids.max()) + 1} points"
    if displacement.shape != (n_points, 3):
        msg = (
            f"{cfg.initial_activation_mesh} Displacement must have shape "
            f"{expected}, got {displacement.shape}"
        )
        raise ValueError(msg)
    msg = "unreachable"
    raise AssertionError(msg)


def make_adjoint_solver() -> Any:
    return SuccessPreferredFallbackSolver(
        solvers=[
            CupyCG(maxiter=ADJOINT_MAXITER, rtol=ADJOINT_RTOL, atol=ADJOINT_ATOL),
            CupyMinRes(maxiter=ADJOINT_MAXITER, tol=ADJOINT_RTOL),
        ]
    )
