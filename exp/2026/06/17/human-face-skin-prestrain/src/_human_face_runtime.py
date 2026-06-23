from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
import torch
from _human_face_config import (
    VTK_ORIGINAL_CELL_IDS,
    VTK_ORIGINAL_POINT_IDS,
    InverseCase,
    InverseConfig,
)
from _human_face_forward import (
    build_forward,
    initial_active_activation_inv,
    initial_forward_displacement,
    make_adjoint_solver,
)
from _human_face_output import surface_edges_for_mask


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


@dataclass(frozen=True)
class CasePaths:
    target: Path
    result: Path
    summary: Path
    history: Path

    @classmethod
    def from_case(cls, data_dir: Path, case: InverseCase) -> CasePaths:
        return cls(
            target=data_dir / f"{case.stem}-target.vtu",
            result=data_dir / f"{case.stem}.vtu",
            summary=data_dir / f"{case.stem}-summary.json",
            history=data_dir / f"{case.stem}-steps.vtkhdf",
        )

    def remove_stale(self) -> None:
        history_tmp = self.history.with_name(f"{self.history.name}.tmp")
        for path in (self.target, self.result, self.summary, self.history, history_tmp):
            if path.exists():
                path.unlink()


@dataclass(frozen=True)
class SourceIds:
    points: np.ndarray | None
    cells: np.ndarray | None


@dataclass
class CaseRuntime:
    mesh: pv.UnstructuredGrid
    skin: pv.PolyData | None
    differentiable_forward: RecordingDifferentiableForward
    base_materials: dict[str, dict[str, torch.Tensor]]
    optimizer: torch.optim.Optimizer
    activation_parameter: torch.nn.Parameter
    initial_activation: np.ndarray
    initial_displacement: np.ndarray | None
    global_ids: np.ndarray
    target_ids: np.ndarray
    active_ids: np.ndarray
    active_ids_t: torch.Tensor
    target_t: torch.Tensor
    target_ids_t: torch.Tensor
    target_global_ids_t: torch.Tensor
    bump_edges: np.ndarray


def source_ids(mesh: pv.UnstructuredGrid) -> SourceIds:
    point_ids = None
    if VTK_ORIGINAL_POINT_IDS in mesh.point_data:
        point_ids = np.asarray(mesh.point_data[VTK_ORIGINAL_POINT_IDS], dtype=np.int64)
    cell_ids = None
    if VTK_ORIGINAL_CELL_IDS in mesh.cell_data:
        cell_ids = np.asarray(mesh.cell_data[VTK_ORIGINAL_CELL_IDS], dtype=np.int64)
    return SourceIds(points=point_ids, cells=cell_ids)


def build_case_runtime(
    *,
    case: InverseCase,
    cfg: InverseConfig,
    mesh: pv.UnstructuredGrid,
    target: np.ndarray,
    loss_mask: np.ndarray,
) -> CaseRuntime:
    from liblaf.apple.common import GLOBAL_POINT_ID

    forward, skin = build_forward(mesh, case)
    ids = source_ids(mesh)
    initial_displacement = (
        initial_forward_displacement(cfg, mesh.n_points, ids.points)
        if cfg.use_initial_displacement
        else None
    )
    if initial_displacement is not None:
        initial_displacement_t = torch.as_tensor(
            initial_displacement,
            dtype=torch.get_default_dtype(),
            device=torch.get_default_device(),
        )
        forward.model.update(forward.state, initial_displacement_t)

    global_ids = np.asarray(mesh.point_data[GLOBAL_POINT_ID.vtk], dtype=np.int64)
    target_ids = np.flatnonzero(loss_mask).astype(np.int64)
    active_ids = np.flatnonzero(
        np.asarray(mesh.cell_data["ActivationMask"], dtype=bool)
    ).astype(np.int64)
    if active_ids.size == 0:
        msg = f"{case.stem} has no active muscle tetrahedra"
        raise ValueError(msg)

    initial_activation = initial_active_activation_inv(
        cfg, active_ids, mesh.n_cells, ids.cells
    )
    activation_parameter = torch.nn.Parameter(
        torch.as_tensor(
            initial_activation,
            dtype=torch.get_default_dtype(),
            device=torch.get_default_device(),
        )
    )
    return CaseRuntime(
        mesh=mesh,
        skin=skin,
        differentiable_forward=RecordingDifferentiableForward(
            forward, make_adjoint_solver()
        ),
        base_materials=forward.model.get_materials(),
        optimizer=torch.optim.Adam([activation_parameter], lr=cfg.inverse_lr),
        activation_parameter=activation_parameter,
        initial_activation=initial_activation,
        initial_displacement=initial_displacement,
        global_ids=global_ids,
        target_ids=target_ids,
        active_ids=active_ids,
        active_ids_t=torch.as_tensor(
            active_ids, dtype=torch.long, device=torch.get_default_device()
        ),
        target_t=torch.as_tensor(
            target, dtype=torch.get_default_dtype(), device=torch.get_default_device()
        ),
        target_ids_t=torch.as_tensor(
            target_ids, dtype=torch.long, device=torch.get_default_device()
        ),
        target_global_ids_t=torch.as_tensor(
            global_ids[target_ids], dtype=torch.long, device=torch.get_default_device()
        ),
        bump_edges=surface_edges_for_mask(mesh, loss_mask),
    )
