"""Matched 3-D folding study: geometry, muscle extent, sharing, and Poisson ratio.

The 16 cases form the Cartesian product of long/short domains, band/full
muscle, per-tetrahedron/shared activation, and nu=0.35/0.49.  They deliberately
retain unconstrained activation and determinant behavior as observable output.
"""

from __future__ import annotations

# ruff: noqa: C901, EM101, EM102, FBT001, FBT003, PLR0911, PLR0915, PT018, TRY003, TRY301
import contextlib
import csv
import hashlib
import io
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pydantic_settings as ps
import pyvista as pv
import scipy.ndimage as ndi
import torch
import warp as wp
from liblaf.peach.linalg import FallbackSolver
from liblaf.peach.linalg.cupy import CupyCG, CupyMinRes

from liblaf import cherries, melon
from liblaf.apple.common import (
    ACTIVATION_INV,
    FIXED_MASK,
    FIXED_VALUE,
    FRACTION,
    LAMBDA,
    MU,
)
from liblaf.apple.common import NU as POISSON
from liblaf.apple.common import E as YOUNG
from liblaf.apple.forward import Forward, ModelBuilder
from liblaf.apple.inverse import DifferentiableForward
from liblaf.apple.warp.fem import StableNeoHookean, StableNeoHookeanActive

FAT_E, MUSCLE_E = 0.003, 0.030
FORWARD_MAX_STEPS, FORWARD_RTOL, FORWARD_ATOL = 10_000, 5.0e-4, 1.0e-10
ADJOINT_MAXITER, ADJOINT_RTOL = 20_000, 5.0e-4


@dataclass(frozen=True)
class Case:
    domain_id: Literal["long", "short"]
    muscle_layout: Literal["band", "full"]
    activation_mode: Literal["per_tet", "shared"]
    poisson: float
    height: float = 0.05

    @property
    def name(self) -> str:
        return (
            f"{self.domain_id}-{self.muscle_layout}-{self.activation_mode}-"
            f"nu{round(self.poisson * 100):02d}-h050"
        )


DOMAINS = {
    "long": ((1.0, 0.1, 1.0), (50, 5, 50)),
    "short": ((0.1, 0.1, 0.1), (5, 5, 5)),
}
CASES = tuple(
    Case(domain_id, layout, sharing, poisson)
    for domain_id in ("long", "short")
    for layout in ("band", "full")
    for sharing in ("per_tet", "shared")
    for poisson in (0.35, 0.49)
)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)

    cases: str = "all"
    output_dir: Path = cherries.output("70-pork-folding-3d", mkdir=True)
    inverse_steps: int = 600
    learning_rate: float = 0.02
    lr_decay: float = 0.99
    tail_absolute_tolerance: float = 1e-10
    validate_derivatives: bool = True
    require_inverse_convergence: bool = True
    smoke: bool = False


def write_json(path: Path, data: Any) -> None:
    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        if isinstance(value, np.ndarray):
            return clean(value.tolist())
        if isinstance(value, np.generic):
            return clean(value.item())
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    path.write_text(json.dumps(clean(data), indent=2, sort_keys=True) + "\n")


def lame(young: float, poisson: float) -> tuple[float, float]:
    if not 0.0 <= poisson < 0.5:
        raise ValueError(f"Poisson ratio must be in [0, 0.5), got {poisson}")
    return (
        young * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson)),
        young / (2.0 * (1.0 + poisson)),
    )


def configure() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("3-D folding study requires CUDA")
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.float64)
    wp.config.mode = "release"
    wp.init()


def mesh_for(case: Case, *, smoke: bool) -> pv.UnstructuredGrid:
    (lx, ly, lz), resolution = DOMAINS[case.domain_id]
    nx, ny, nz = (3, 5, 3) if smoke else resolution
    ys = np.linspace(0.0, ly, ny + 1)
    if not (np.any(np.isclose(ys, 0.04)) and np.any(np.isclose(ys, 0.06))):
        raise ValueError("y resolution must include 0.04 and 0.06")
    xs, zs = np.linspace(0.0, lx, nx + 1), np.linspace(0.0, lz, nz + 1)
    points = np.asarray([(x, y, z) for y in ys for z in zs for x in xs])

    def vid(i: int, j: int, k: int) -> int:
        return (j * (nz + 1) + k) * (nx + 1) + i

    raw = (
        (0, 1, 3, 7),
        (0, 3, 2, 7),
        (0, 2, 6, 7),
        (0, 6, 4, 7),
        (0, 4, 5, 7),
        (0, 5, 1, 7),
    )
    tets: list[list[int]] = []
    for j in range(ny):
        for k in range(nz):
            for i in range(nx):
                corners = (
                    vid(i, j, k),
                    vid(i + 1, j, k),
                    vid(i, j, k + 1),
                    vid(i + 1, j, k + 1),
                    vid(i, j + 1, k),
                    vid(i + 1, j + 1, k),
                    vid(i, j + 1, k + 1),
                    vid(i + 1, j + 1, k + 1),
                )
                for local in raw:
                    tet = [corners[index] for index in local]
                    a, b, c, d = points[tet]
                    if np.linalg.det(np.stack((b - a, c - a, d - a))) < 0:
                        tet[0], tet[1] = tet[1], tet[0]
                    tets.append(tet)
    packed = np.column_stack((np.full(len(tets), 4, dtype=np.int64), tets)).ravel()
    mesh = pv.UnstructuredGrid(packed, np.full(len(tets), pv.CellType.TETRA), points)
    tol = 1.0e-12
    p = mesh.points
    fixed = (
        (np.abs(p[:, 1]) < tol)
        | (np.abs(p[:, 0]) < tol)
        | (np.abs(p[:, 0] - lx) < tol)
        | (np.abs(p[:, 2]) < tol)
        | (np.abs(p[:, 2] - lz) < tol)
    )
    mesh.point_data[FIXED_MASK.vtk] = np.repeat(fixed[:, None], 3, axis=1)
    mesh.point_data[FIXED_VALUE.vtk] = np.zeros((mesh.n_points, 3))
    mesh.point_data["FixedBoundary"] = fixed.astype(np.uint8)
    top = np.abs(p[:, 1] - ly) < tol
    mesh.point_data["TopSurface"] = top.astype(np.uint8)
    mesh.point_data["TargetSurface"] = (top & ~fixed).astype(np.uint8)
    if case.muscle_layout == "full":
        muscle = np.ones(mesh.n_cells, dtype=bool)
    else:
        cy = mesh.cell_centers().points[:, 1]
        muscle = (cy >= 0.04 - tol) & (cy <= 0.06 + tol)
    if not muscle.any():
        raise AssertionError("case must have active muscle cells")
    mesh.cell_data["Muscle"] = muscle.astype(np.uint8)
    return mesh


def set_material(
    mesh: pv.UnstructuredGrid, young: float, poisson: float, fraction: np.ndarray
) -> None:
    la, mu = lame(young, poisson)
    mesh.cell_data[YOUNG.vtk] = np.full(mesh.n_cells, young)
    mesh.cell_data[POISSON.vtk] = np.full(mesh.n_cells, poisson)
    mesh.cell_data[LAMBDA.vtk] = np.full(mesh.n_cells, la)
    mesh.cell_data[MU.vtk] = np.full(mesh.n_cells, mu)
    mesh.cell_data[FRACTION.vtk] = fraction.astype(float)


def output_materials(mesh: pv.UnstructuredGrid, poisson: float) -> None:
    muscle = np.asarray(mesh.cell_data["Muscle"], bool)
    young = np.where(muscle, MUSCLE_E, FAT_E)
    mesh.cell_data[YOUNG.vtk] = young
    mesh.cell_data[POISSON.vtk] = np.full(mesh.n_cells, poisson)
    mesh.cell_data[LAMBDA.vtk] = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
    mesh.cell_data[MU.vtk] = young / (2 * (1 + poisson))
    mesh.cell_data[FRACTION.vtk] = np.ones(mesh.n_cells)
    mesh.cell_data["FatFraction"] = (~muscle).astype(float)
    mesh.cell_data["MuscleFraction"] = muscle.astype(float)


def build_forward(mesh: pv.UnstructuredGrid, poisson: float) -> Forward:
    muscle = np.asarray(mesh.cell_data["Muscle"], bool)
    builder = ModelBuilder()
    builder.add_vertices(mesh)
    builder.add_fixed(mesh)
    # Full-muscle cases omit a zero-fraction fat potential rather than pretending
    # that an inactive fat material participates in the model.
    if (~muscle).any():
        set_material(mesh, FAT_E, poisson, ~muscle)
        builder.add_potential(StableNeoHookean.from_pyvista(mesh, name="fat"))
    set_material(mesh, MUSCLE_E, poisson, muscle)
    mesh.cell_data[ACTIVATION_INV.vtk] = np.zeros((mesh.n_cells, 6))
    builder.add_potential(StableNeoHookeanActive.from_pyvista(mesh, name="muscle"))
    forward = Forward(builder.finalize())
    forward.optimizer = forward.default_optimizer(
        max_steps=FORWARD_MAX_STEPS, rtol=FORWARD_RTOL, atol=FORWARD_ATOL
    )
    return forward


class RecordedForward:
    def __init__(self, forward: Forward) -> None:
        self.impl = DifferentiableForward(forward)
        self.impl.adjoint_solver = FallbackSolver(
            solvers=[
                CupyCG(maxiter=ADJOINT_MAXITER, rtol=ADJOINT_RTOL, atol=0.0),
                CupyMinRes(maxiter=ADJOINT_MAXITER, tol=ADJOINT_RTOL),
            ]
        )
        self.forward_solution: Any = None

    def forward(self, materials: Any) -> torch.Tensor:
        result = self.impl.forward(materials)
        self.forward_solution = self.impl.last_solution
        return result

    @property
    def adjoint_solution(self) -> Any:
        return self.impl.last_adjoint_solution


def solution_info(solution: Any, name: str) -> dict[str, Any]:
    if solution is None:
        return {f"{name}/success": False, f"{name}/result": "missing"}
    if name == "forward":
        state = solution.state.convergence_state
        return {
            f"{name}/success": bool(solution.success),
            f"{name}/result": str(solution.result),
            f"{name}/steps": int(state.step),
            f"{name}/grad_norm": float(state.grad_norm.detach().cpu()),
        }
    state = solution.state
    best = int(state.best_index.detach().cpu())
    return {
        f"{name}/success": bool(solution.success),
        f"{name}/result": str(solution.result),
        f"{name}/best_solver": best,
        f"{name}/absolute_residual": float(
            state.absolute_residuals[best].detach().cpu()
        ),
        f"{name}/relative_residual": float(
            state.relative_residuals[best].detach().cpu()
        ),
    }


def target(mesh: pv.UnstructuredGrid, case: Case) -> np.ndarray:
    lx, _, lz = DOMAINS[case.domain_id][0]
    p, d = mesh.points, np.zeros_like(mesh.points)
    top = np.asarray(mesh.point_data["TopSurface"], bool)
    d[top, 1] = (
        case.height
        * 16
        * (p[top, 0] / lx)
        * (1 - p[top, 0] / lx)
        * (p[top, 2] / lz)
        * (1 - p[top, 2] / lz)
    )
    return d


def group_map(
    mesh: pv.UnstructuredGrid, mode: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    active = np.flatnonzero(np.asarray(mesh.cell_data["Muscle"], bool))
    cells_to_groups = np.full(mesh.n_cells, -1, dtype=np.int64)
    active_groups = (
        np.arange(len(active), dtype=np.int64)
        if mode == "per_tet"
        else np.zeros(len(active), dtype=np.int64)
    )
    cells_to_groups[active] = active_groups
    counts = np.bincount(active_groups, minlength=int(active_groups.max()) + 1)
    return (
        active,
        active_groups,
        cells_to_groups,
        {
            "sharing_id": mode,
            "group_count": len(counts),
            "cells_per_group_min": int(counts.min()),
            "cells_per_group_max": int(counts.max()),
            "raw_dofs_per_group": 6,
            "raw_activation_dofs": int(6 * len(counts)),
            "cell_to_group_sha256": hashlib.sha256(
                cells_to_groups.astype("<i8", copy=False).tobytes()
            ).hexdigest(),
        },
    )


def expand_activation(
    parameter: torch.Tensor,
    active_ids: torch.Tensor,
    active_groups: torch.Tensor,
    ncells: int,
) -> torch.Tensor:
    values = parameter.index_select(0, active_groups)
    return torch.zeros(
        (ncells, 6), device=parameter.device, dtype=parameter.dtype
    ).index_copy(0, active_ids, values)


def cell_determinants(
    mesh: pv.UnstructuredGrid, u: np.ndarray, activation: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cells = np.asarray(mesh.cells).reshape(-1, 5)[:, 1:]
    x, xd = mesh.points[cells], (mesh.points + u)[cells]
    dm = np.transpose(x[:, 1:] - x[:, :1], (0, 2, 1))
    ds = np.transpose(xd[:, 1:] - xd[:, :1], (0, 2, 1))
    det_f = np.linalg.det(ds @ np.linalg.inv(dm))
    A = np.zeros((mesh.n_cells, 3, 3))
    A[:, (0, 1, 2), (0, 1, 2)] = 1.0 + activation[:, :3]
    A[:, 0, 1] = A[:, 1, 0] = activation[:, 3]
    A[:, 1, 2] = A[:, 2, 1] = activation[:, 4]
    A[:, 0, 2] = A[:, 2, 0] = activation[:, 5]
    return (
        det_f,
        det_f * np.linalg.det(A),
        np.linalg.det(A),
        np.abs(np.linalg.det(dm)) / 6,
    )


def neighbors(mesh: pv.UnstructuredGrid) -> np.ndarray:
    cells = np.asarray(mesh.cells).reshape(-1, 5)[:, 1:]
    active = np.flatnonzero(np.asarray(mesh.cell_data["Muscle"], bool))
    owner: dict[tuple[int, int, int], int] = {}
    pairs: list[tuple[int, int]] = []
    for cid in active:
        for omit in range(4):
            face = tuple(sorted(np.delete(cells[cid], omit)))
            if face in owner:
                pairs.append((owner.pop(face), int(cid)))
            else:
                owner[face] = int(cid)
    return np.asarray(pairs, dtype=np.int64).reshape((-1, 2))


def diagnostics(
    mesh: pv.UnstructuredGrid,
    u: np.ndarray,
    a: np.ndarray,
    tgt: np.ndarray,
    pairs: np.ndarray,
) -> dict[str, float]:
    p = mesh.points
    top = np.asarray(mesh.point_data["TopSurface"], bool)
    coords, values = p[top], u[top, 1]
    xs, zs = (
        sorted(set(np.round(coords[:, 0], 12))),
        sorted(set(np.round(coords[:, 2], 12))),
    )
    lookup = {
        (round(x, 12), round(z, 12)): y
        for (x, _, z), y in zip(coords, values, strict=True)
    }
    field = np.asarray([[lookup[x, z] for x in xs] for z in zs])
    dx, dz = float(np.mean(np.diff(xs))), float(np.mean(np.diff(zs)))
    smooth = ndi.gaussian_filter(
        field, sigma=(max(0.02 / dz, 0.5), max(0.02 / dx, 0.5)), mode="nearest"
    )
    grad_z, grad_x = np.gradient(field, dz, dx)
    dzz, dzx = np.gradient(grad_z, dz, dx)
    dxz, dxx = np.gradient(grad_x, dz, dx)
    det_f, det_g, det_a, rest = cell_determinants(mesh, u, a)
    negative = det_f < 0
    active = a[np.asarray(mesh.cell_data["Muscle"], bool)]
    jumps = a[pairs[:, 0]] - a[pairs[:, 1]] if pairs.size else np.empty((0, 6))
    error = np.linalg.norm(
        (u - tgt)[np.asarray(mesh.point_data["TargetSurface"], bool)], axis=1
    )
    return {
        "target/mae": float(error.mean()),
        "target/rms": float(np.sqrt(np.mean(error**2))),
        "target/max": float(error.max()),
        "top/highpass_rms": float(np.sqrt(np.mean((field - smooth) ** 2))),
        "top/slope_rms": float(np.sqrt(np.mean(grad_x**2 + grad_z**2))),
        "top/curvature_rms": float(np.sqrt(np.mean(dxx**2 + dzz**2 + dxz**2 + dzx**2))),
        "top/laplacian_rms": float(np.sqrt(np.mean((dxx + dzz) ** 2))),
        "activation/rms": float(np.sqrt(np.mean(active**2))),
        "activation/neighbor_jump_rms": float(np.sqrt(np.mean(jumps**2)))
        if jumps.size
        else 0.0,
        "detF/min": float(det_f.min()),
        "detG/min": float(det_g.min()),
        "detAinv/min": float(det_a.min()),
        "inverted_cell_fraction": float(negative.mean()),
        "inverted_rest_measure_fraction": float(rest[negative].sum() / rest.sum()),
        # Rest-volume-weighted mean of max(-detF, 0) over all cells.
        "negative_det_f_mean": float(
            (rest * np.maximum(-det_f, 0.0)).sum() / rest.sum()
        ),
    }


def frame(
    mesh: pv.UnstructuredGrid,
    tgt: np.ndarray,
    u: np.ndarray,
    a: np.ndarray,
    row: dict[str, Any],
) -> pv.UnstructuredGrid:
    out = mesh.copy(deep=True)
    out.point_data["Displacement"], out.point_data["TargetDisplacement"] = u, tgt
    out.point_data["DisplacementError"] = u - tgt
    out.point_data["DeformedPoint"], out.point_data["TargetPoint"] = (
        mesh.points + u,
        mesh.points + tgt,
    )
    out.cell_data[ACTIVATION_INV.vtk] = a
    det_f, det_g, det_a, _ = cell_determinants(mesh, u, a)
    out.cell_data["DetF"], out.cell_data["DetG"], out.cell_data["DetAinv"] = (
        det_f,
        det_g,
        det_a,
    )
    for key, value in row.items():
        if isinstance(value, (int, float, bool)) and math.isfinite(float(value)):
            out.field_data[key.replace("/", "_")] = np.asarray([value])
    return out


def loss(residual: torch.Tensor) -> torch.Tensor:
    return residual.square().sum(dim=1).mean()


def derivative_check(case: Case) -> dict[str, Any]:
    mesh = mesh_for(case, smoke=True)
    active, groups, _, group_info = group_map(mesh, case.activation_mode)
    active_t, groups_t = (
        torch.as_tensor(active, dtype=torch.long),
        torch.as_tensor(groups, dtype=torch.long),
    )
    tgt, top = (
        target(mesh, case),
        torch.as_tensor(
            np.flatnonzero(np.asarray(mesh.point_data["TargetSurface"], bool)),
            dtype=torch.long,
        ),
    )
    target_t = torch.as_tensor(tgt)
    shape = (group_info["group_count"], 6)

    def evaluate(
        values: np.ndarray, backward: bool
    ) -> tuple[float, np.ndarray | None, dict[str, Any], dict[str, Any]]:
        local_mesh = mesh_for(case, smoke=True)
        recorded = RecordedForward(build_forward(local_mesh, case.poisson))
        parameter = torch.tensor(values, requires_grad=backward)
        materials = {
            key: dict(value)
            for key, value in recorded.impl.model.get_materials().items()
        }
        materials["muscle"]["activation_inv"] = expand_activation(
            parameter, active_t, groups_t, local_mesh.n_cells
        )
        with contextlib.redirect_stdout(io.StringIO()):
            u = recorded.forward(materials).clone()
        value = loss(u[top] - target_t[top])
        gradient = None
        if backward:
            value.backward()
            if parameter.grad is None:
                raise RuntimeError("derivative check produced no gradient")
            gradient = parameter.grad.detach().cpu().numpy()
        return (
            float(value.detach().cpu()),
            gradient,
            solution_info(recorded.forward_solution, "forward"),
            solution_info(recorded.adjoint_solution, "adjoint"),
        )

    zero = np.zeros(shape)
    value, grad, fwd, adj = evaluate(zero, True)
    assert grad is not None
    index = tuple(int(v) for v in np.unravel_index(np.argmax(np.abs(grad)), grad.shape))
    # The forward PNCG tolerance makes a 1e-4 perturbation too close to solver
    # noise for the 270-control per-tet nu=0.35 representative.
    eps = 1.0e-3
    low, high = zero.copy(), zero.copy()
    low[index] -= eps
    high[index] += eps
    left, _, left_fwd, _ = evaluate(low, False)
    right, _, right_fwd, _ = evaluate(high, False)
    finite = (right - left) / (2 * eps)
    analytic = float(grad[index])
    relative = abs(finite - analytic) / max(abs(finite), abs(analytic), 1.0e-14)
    return {
        "representative": asdict(case),
        "sharing": group_info,
        "objective": value,
        "component": list(index),
        "epsilon": eps,
        "analytic": analytic,
        "finite_difference": finite,
        "relative_error": relative,
        "forward_converged": fwd["forward/success"],
        "adjoint_converged": adj["adjoint/success"],
        "minus_forward_converged": left_fwd["forward/success"],
        "plus_forward_converged": right_fwd["forward/success"],
        "passed": bool(
            fwd["forward/success"]
            and adj["adjoint/success"]
            and left_fwd["forward/success"]
            and right_fwd["forward/success"]
            and relative < 0.01
        ),
    }


def run_case(case: Case, cfg: Config) -> dict[str, Any]:
    mesh = mesh_for(case, smoke=cfg.smoke)
    outdir = cfg.output_dir / case.name
    outdir.mkdir(parents=True, exist_ok=False)
    tgt = target(mesh, case)
    recorded = RecordedForward(build_forward(mesh, case.poisson))
    output_materials(mesh, case.poisson)
    active, groups, cell_to_group, sharing = group_map(mesh, case.activation_mode)
    mesh.cell_data["ActivationGroup"] = cell_to_group
    active_t, groups_t = (
        torch.as_tensor(active, dtype=torch.long),
        torch.as_tensor(groups, dtype=torch.long),
    )
    top = torch.as_tensor(
        np.flatnonzero(np.asarray(mesh.point_data["TargetSurface"], bool)),
        dtype=torch.long,
    )
    target_t = torch.as_tensor(tgt)
    parameter = torch.nn.Parameter(torch.zeros((sharing["group_count"], 6)))
    optimizer = torch.optim.Adam([parameter], lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.lr_decay)
    pairs, trace, series = neighbors(mesh), [], []
    frames_dir = outdir / "frames"
    frames_dir.mkdir()
    melon.save(
        frame(mesh, tgt, np.zeros_like(tgt), np.zeros((mesh.n_cells, 6)), {}),
        outdir / "target.vtu",
    )
    failures = {"forward": 0, "adjoint": 0, "nonfinite": 0}
    best: tuple[float, np.ndarray, np.ndarray, int] | None = None
    best_converged: tuple[float, np.ndarray, np.ndarray, int] | None = None
    best_orientation: tuple[float, np.ndarray, np.ndarray, int] | None = None

    def persist_partial() -> None:
        if trace:
            with (outdir / "trace.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=sorted({key for row in trace for key in row})
                )
                writer.writeheader()
                writer.writerows(trace)
        write_json(
            outdir / "history.vtu.series",
            {"file-series-version": "1.0", "files": series},
        )

    for step in range(cfg.inverse_steps + 1):
        optimizer.zero_grad()
        materials = {
            key: dict(value)
            for key, value in recorded.impl.model.get_materials().items()
        }
        materials["muscle"]["activation_inv"] = expand_activation(
            parameter, active_t, groups_t, mesh.n_cells
        )
        started = time.perf_counter()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                u = recorded.forward(materials).clone()
            objective = loss(u[top] - target_t[top])
            objective.backward()
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise FloatingPointError("non-finite or absent inverse gradient")
        except Exception as error:
            failures["nonfinite"] += 1
            row = {
                "step": step,
                "evaluation_success": False,
                "error": repr(error),
                "elapsed_s": time.perf_counter() - started,
                **solution_info(recorded.forward_solution, "forward"),
                **solution_info(recorded.adjoint_solution, "adjoint"),
            }
            trace.append(row)
            write_json(outdir / "failure.json", row)
            persist_partial()
            raise RuntimeError(
                f"inverse evaluation {step} failed; partial trace retained"
            ) from error
        fwd, adj = (
            solution_info(recorded.forward_solution, "forward"),
            solution_info(recorded.adjoint_solution, "adjoint"),
        )
        success = bool(fwd["forward/success"] and adj["adjoint/success"])
        failures["forward"] += int(not fwd["forward/success"])
        failures["adjoint"] += int(not adj["adjoint/success"])
        u_np = u.detach().cpu().numpy()
        a_np = (
            expand_activation(parameter.detach(), active_t, groups_t, mesh.n_cells)
            .cpu()
            .numpy()
        )
        diagnostic = diagnostics(mesh, u_np, a_np, tgt, pairs)
        row = {
            "step": step,
            "evaluation_success": success,
            "loss": float(objective.detach().cpu()),
            "grad_norm": float(parameter.grad.norm().detach().cpu()),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "elapsed_s": time.perf_counter() - started,
            **diagnostic,
            **fwd,
            **adj,
        }
        trace.append(row)
        path = frames_dir / f"step-{step:05d}.vtu"
        melon.save(frame(mesh, tgt, u_np, a_np, row), path)
        series.append({"name": str(path.relative_to(outdir)), "time": float(step)})
        persist_partial()
        cherries.set_step(step)
        cherries.log_metrics(
            {
                f"{case.name}/loss": row["loss"],
                f"{case.name}/inverted_cell_fraction": row["inverted_cell_fraction"],
            }
        )
        if not success:
            write_json(outdir / "failure.json", row)
            persist_partial()
            raise RuntimeError(
                f"inverse evaluation {step} has nonconverged forward/adjoint"
            )
        if best is None or row["loss"] < best[0]:
            best = (row["loss"], u_np.copy(), a_np.copy(), step)
        if best_converged is None or row["loss"] < best_converged[0]:
            best_converged = (row["loss"], u_np.copy(), a_np.copy(), step)
        if (
            diagnostic["detF/min"] > 0
            and diagnostic["detG/min"] > 0
            and diagnostic["detAinv/min"] > 0
            and (best_orientation is None or row["loss"] < best_orientation[0])
        ):
            best_orientation = (row["loss"], u_np.copy(), a_np.copy(), step)
        if step < cfg.inverse_steps:
            optimizer.step()
            scheduler.step()

    assert (
        best is not None and best_converged is not None and best_orientation is not None
    )
    persist_partial()
    final_u, final_a = u_np, a_np
    melon.save(
        frame(
            mesh, tgt, best[1], best[2], {"best_step": best[3], "best_loss": best[0]}
        ),
        outdir / "best.vtu",
    )
    melon.save(
        frame(
            mesh,
            tgt,
            best_converged[1],
            best_converged[2],
            {
                "best_converged_step": best_converged[3],
                "best_converged_loss": best_converged[0],
            },
        ),
        outdir / "best-converged.vtu",
    )
    melon.save(
        frame(
            mesh,
            tgt,
            best_orientation[1],
            best_orientation[2],
            {
                "best_orientation_preserving_step": best_orientation[3],
                "best_orientation_preserving_loss": best_orientation[0],
            },
        ),
        outdir / "best-orientation-preserving.vtu",
    )
    melon.save(frame(mesh, tgt, final_u, final_a, trace[-1]), outdir / "final.vtu")
    tail = trace[-min(50, len(trace)) :]
    losses = np.asarray([row["loss"] for row in tail])
    tail_absolute_range = float(np.ptp(losses))
    tail_range = float(tail_absolute_range / max(abs(float(losses.min())), 1e-30))
    tail_valid = all(row["forward/success"] and row["adjoint/success"] for row in tail)
    final = trace[-1]
    report = {
        "case": {
            "name": case.name,
            "geometry_id": case.domain_id,
            "domain_id": case.domain_id,
            "muscle_layout": case.muscle_layout,
            "activation_mode": case.activation_mode,
            "poisson": case.poisson,
            "height": case.height,
        },
        "geometry": {
            "geometry_id": case.domain_id,
            "muscle_extent_id": case.muscle_layout,
            "domain": list(DOMAINS[case.domain_id][0]),
            "structured_resolution": [3, 5, 3]
            if cfg.smoke
            else list(DOMAINS[case.domain_id][1]),
            "active_band_y": [0.04, 0.06] if case.muscle_layout == "band" else None,
            "fixed": "bottom and all four lateral faces, xyz",
            "top_and_interior": "free",
        },
        "physics": {
            "elasticity": "stable_neo_hookean",
            "target_loss": "l2",
            "target_observation": "free top nodes, xyz components",
        },
        "materials": {
            "fat": None
            if case.muscle_layout == "full"
            else {"E_MPa": FAT_E, "nu": case.poisson},
            "muscle": {"E_MPa": MUSCLE_E, "nu": case.poisson},
            "skin_energy": None,
        },
        "activation": {
            **sharing,
            "bounds": None,
            "regularizer": None,
            "det_constraint": None,
        },
        "counts": {
            "points": mesh.n_points,
            "tets": mesh.n_cells,
            "muscle_tets": len(active),
            "fat_tets": int(mesh.n_cells - len(active)),
            "activation_dofs": int(parameter.numel()),
            "free_top_vertices": len(top),
            "observed_components": int(3 * len(top)),
            "muscle_neighbor_pairs": len(pairs),
        },
        "inverse": {
            "schedule": "fixed",
            "evaluations": len(trace),
            "updates": cfg.inverse_steps,
            "early_stopping": None,
            "completion_gate": "all forward/adjoint evaluations valid and (final 50-step loss relative range <= 1% or absolute L2 range <= tolerance)",
            "best_step": best[3],
            "best_loss": best[0],
            "best_converged_step": best_converged[3],
            "best_converged_loss": best_converged[0],
            "best_orientation_preserving_step": best_orientation[3],
            "best_orientation_preserving_loss": best_orientation[0],
            "first_inversion_step": next(
                (row["step"] for row in trace if row["inverted_cell_fraction"] > 0),
                None,
            ),
            "last_inversion_step": next(
                (
                    row["step"]
                    for row in reversed(trace)
                    if row["inverted_cell_fraction"] > 0
                ),
                None,
            ),
            "inverted_frame_fraction": float(
                np.mean([row["inverted_cell_fraction"] > 0 for row in trace])
            ),
            "minimum_det_f": min(row["detF/min"] for row in trace),
            "minimum_det_g": min(row["detG/min"] for row in trace),
            "minimum_det_ainv": min(row["detAinv/min"] for row in trace),
            "final_detF_negative_cell_fraction": final["inverted_cell_fraction"],
            "final_detF_negative_rest_measure_fraction": final[
                "inverted_rest_measure_fraction"
            ],
            "peak_detF_negative_cell_fraction": max(
                row["inverted_cell_fraction"] for row in trace
            ),
            "peak_detF_negative_rest_measure_fraction": max(
                row["inverted_rest_measure_fraction"] for row in trace
            ),
            "final_negative_det_f_mean": final["negative_det_f_mean"],
            "peak_negative_det_f_mean": max(
                row["negative_det_f_mean"] for row in trace
            ),
            "tail": {
                "window": len(tail),
                "all_forward_adjoint_converged": tail_valid,
                "relative_range": tail_range,
                "absolute_range": tail_absolute_range,
                "absolute_tolerance": cfg.tail_absolute_tolerance,
                "criterion": "all valid and (relative range <= 1% or absolute L2 range <= tolerance)",
                "inverse_converged_1pct_tail_gate": bool(
                    tail_valid
                    and (
                        tail_range <= 0.01
                        or tail_absolute_range <= cfg.tail_absolute_tolerance
                    )
                ),
            },
            "failures": {**failures, "inverse": failures["nonfinite"]},
        },
        "metrics": {
            "best": diagnostics(mesh, best[1], best[2], tgt, pairs),
            "best_converged": diagnostics(
                mesh, best_converged[1], best_converged[2], tgt, pairs
            ),
            "best_orientation_preserving": diagnostics(
                mesh, best_orientation[1], best_orientation[2], tgt, pairs
            ),
            "final": diagnostics(mesh, final_u, final_a, tgt, pairs),
        },
        "target": "u_y=h*16*(x/Lx)*(1-x/Lx)*(z/Lz)*(1-z/Lz), u_x=u_z=0",
        "paraview": {
            "vtu_series": str(outdir / "history.vtu.series"),
            "fps": 30,
            "frames": len(series),
            "source_time": "integer optimization step",
        },
        "artifacts": {
            "target": "target.vtu",
            "series": "history.vtu.series",
            "best": "best.vtu",
            "best_converged": "best-converged.vtu",
            "best_orientation_preserving": "best-orientation-preserving.vtu",
            "final": "final.vtu",
        },
        "trace_csv": str(outdir / "trace.csv"),
    }
    write_json(outdir / "summary.json", report)
    return report


def select(cfg: Config) -> tuple[Case, ...]:
    requested = [name.strip() for name in cfg.cases.split(",") if name.strip()]
    if requested == ["all"]:
        return CASES
    known = {case.name: case for case in CASES}
    unknown = [name for name in requested if name not in known]
    if unknown:
        raise ValueError(
            f"unknown cases {unknown!r}; expected names from {sorted(known)!r} or all"
        )
    return tuple(known[name] for name in requested)


def main(cfg: Config) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    if any(cfg.output_dir.iterdir()):
        raise FileExistsError(
            f"refusing nonempty output root: {cfg.output_dir}; choose a new root"
        )
    configure()
    representatives = (
        Case("long", "band", activation_mode, poisson)
        for activation_mode in ("per_tet", "shared")
        for poisson in (0.35, 0.49)
    )
    checks = (
        [derivative_check(case) for case in representatives]
        if cfg.validate_derivatives
        else [{"skipped": True}]
    )
    if cfg.validate_derivatives and not all(item["passed"] for item in checks):
        raise RuntimeError(f"derivative check failed: {checks}")
    reports = [run_case(case, cfg) for case in select(cfg)]
    write_json(
        cfg.output_dir / "summary.json",
        {
            "design": "3d-pork-folding-full-factorial",
            "derivative_checks": checks,
            "cases": reports,
        },
    )
    failed_tail = [
        report["case"]["name"]
        for report in reports
        if not report["inverse"]["tail"]["inverse_converged_1pct_tail_gate"]
    ]
    if cfg.require_inverse_convergence and not cfg.smoke and failed_tail:
        raise RuntimeError(
            "1% tail gate failed after writing all requested summaries: "
            + ", ".join(failed_tail)
        )


if __name__ == "__main__":
    cherries.main(main)
