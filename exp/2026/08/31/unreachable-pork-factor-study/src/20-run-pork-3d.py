"""Unreachable-target 3-D pork factor study.

This is deliberately an *unregularised* inverse problem: every muscle tetrahedron
owns six raw entries of ``Ainv = I + sym(a)``.  In particular, it does not clip
activations, repair inversions, or replace failed solves with a previous state.
Those states are useful evidence when diagnosing the observed bumpiness.
"""

from __future__ import annotations

# ruff: noqa: C901, E402, EM101, EM102, PERF401, PLR0911, PLR0915, TRY003, TRY301
import contextlib
import csv
import io
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

import attrs
import numpy as np
import pydantic_settings as ps
import pyvista as pv
import scipy.ndimage as ndi
import torch
import warp as wp

from liblaf import cherries, melon

LOG = logging.getLogger(__name__)
FAT_E, MUSCLE_E, NU = 0.003, 0.030, 0.49
FORWARD_MAX_STEPS, FORWARD_RTOL, FORWARD_ATOL = 10_000, 5.0e-4, 1.0e-10
ADJOINT_MAXITER, ADJOINT_RTOL = 20_000, 5.0e-4


# Self-contained small-strain volumetric potentials.  They intentionally use
# the same six-entry raw ``Ainv`` convention as StableNeoHookeanActive:
# G=F+(Ainv-I), epsilon=sym(G-I).  No spectral projection, determinant guard, or
# activation bound is applied.  The Hessian is constant, which makes this
# a useful rigorous linear-elastic control rather than a relabelled nonlinear
# experiment.
from liblaf.apple.common import ACTIVATION_INV, LAMBDA, MU
from liblaf.apple.warp.fem import func as fem_func
from liblaf.apple.warp.fem import utils as fem_utils
from liblaf.apple.warp.fem._base import WarpPotentialFem
from liblaf.apple.warp.model import MaterialField

_floating = Any
_mat33 = Any
_mat43 = Any
_materials = Any


@wp.func
def _linear_energy_g(G: _mat33, la: _floating, mu: _floating) -> _floating:
    e = G.dtype(0.5) * (G + wp.transpose(G)) - wp.identity(3, dtype=G.dtype)
    return G.dtype(0.5) * la * wp.trace(e) * wp.trace(e) + mu * wp.ddot(e, e)


@wp.func
def _linear_stress_g(G: _mat33, la: _floating, mu: _floating) -> _mat33:
    e = G.dtype(0.5) * (G + wp.transpose(G)) - wp.identity(3, dtype=G.dtype)
    return la * wp.trace(e) * wp.identity(3, dtype=G.dtype) + G.dtype(2.0) * mu * e


@wp.func
def _linear_hess_scalar(
    dh: Any, component: int, A: _mat33, la: _floating, mu: _floating
) -> _floating:
    e = wp.vector(dh.dtype(0.0), dh.dtype(0.0), dh.dtype(0.0))
    if component == 0:
        e = wp.vector(dh.dtype(1.0), dh.dtype(0.0), dh.dtype(0.0))
    if component == 1:
        e = wp.vector(dh.dtype(0.0), dh.dtype(1.0), dh.dtype(0.0))
    if component == 2:
        e = wp.vector(dh.dtype(0.0), dh.dtype(0.0), dh.dtype(1.0))
    dG = wp.outer(e, dh) @ A
    dE = dG.dtype(0.5) * (dG + wp.transpose(dG))
    return la * wp.trace(dE) * wp.trace(dE) + dG.dtype(2.0) * mu * wp.ddot(dE, dE)


@wp.func
def _linear_hess_diag(dhdX: _mat43, A: _mat33, la: _floating, mu: _floating) -> _mat43:
    return wp.matrix_from_rows(
        wp.vector(
            _linear_hess_scalar(dhdX[0], 0, A, la, mu),
            _linear_hess_scalar(dhdX[0], 1, A, la, mu),
            _linear_hess_scalar(dhdX[0], 2, A, la, mu),
        ),
        wp.vector(
            _linear_hess_scalar(dhdX[1], 0, A, la, mu),
            _linear_hess_scalar(dhdX[1], 1, A, la, mu),
            _linear_hess_scalar(dhdX[1], 2, A, la, mu),
        ),
        wp.vector(
            _linear_hess_scalar(dhdX[2], 0, A, la, mu),
            _linear_hess_scalar(dhdX[2], 1, A, la, mu),
            _linear_hess_scalar(dhdX[2], 2, A, la, mu),
        ),
        wp.vector(
            _linear_hess_scalar(dhdX[3], 0, A, la, mu),
            _linear_hess_scalar(dhdX[3], 1, A, la, mu),
            _linear_hess_scalar(dhdX[3], 2, A, la, mu),
        ),
    )


@wp.func
def _linear_hess_prod(
    p: _mat43, dhdX: _mat43, A: _mat33, la: _floating, mu: _floating
) -> _mat43:
    dG = fem_func.deformation_gradient_jvp(dhdX, p) @ A
    dE = dG.dtype(0.5) * (dG + wp.transpose(dG))
    dS = la * wp.trace(dE) * wp.identity(3, dtype=dG.dtype) + dG.dtype(2.0) * mu * dE
    return fem_func.deformation_gradient_vjp(dhdX, dS @ wp.transpose(A))


@wp.func
def _linear_hess_quad(
    p: _mat43, dhdX: _mat43, A: _mat33, la: _floating, mu: _floating
) -> _floating:
    dG = fem_func.deformation_gradient_jvp(dhdX, p) @ A
    dE = dG.dtype(0.5) * (dG + wp.transpose(dG))
    return la * wp.trace(dE) * wp.trace(dE) + dG.dtype(2.0) * mu * wp.ddot(dE, dE)


@wp.func
def _linear_passive_energy(F: _mat33, materials: _materials, cid: int) -> _floating:
    return _linear_energy_g(F, materials.lmbda[cid], materials.mu[cid])


@wp.func
def _linear_passive_stress(F: _mat33, materials: _materials, cid: int) -> _mat33:
    return _linear_stress_g(F, materials.lmbda[cid], materials.mu[cid])


@wp.func
def _linear_passive_diag(
    F: _mat33, dh: _mat43, materials: _materials, cid: int
) -> _mat43:
    return _linear_hess_diag(
        dh, wp.identity(3, dtype=F.dtype), materials.lmbda[cid], materials.mu[cid]
    )


@wp.func
def _linear_passive_prod(
    F: _mat33, p: _mat43, dh: _mat43, materials: _materials, cid: int
) -> _mat43:
    return _linear_hess_prod(
        p, dh, wp.identity(3, dtype=F.dtype), materials.lmbda[cid], materials.mu[cid]
    )


@wp.func
def _linear_passive_quad(
    F: _mat33, p: _mat43, dh: _mat43, materials: _materials, cid: int
) -> _floating:
    return _linear_hess_quad(
        p, dh, wp.identity(3, dtype=F.dtype), materials.lmbda[cid], materials.mu[cid]
    )


@wp.func
def _linear_active_energy(F: _mat33, materials: _materials, cid: int) -> _floating:
    A = fem_func.make_activation_mat33(materials.activation_inv[cid])
    G = F + A - wp.identity(3, dtype=F.dtype)
    return _linear_energy_g(G, materials.lmbda[cid], materials.mu[cid])


@wp.func
def _linear_active_stress(F: _mat33, materials: _materials, cid: int) -> _mat33:
    A = fem_func.make_activation_mat33(materials.activation_inv[cid])
    G = F + A - wp.identity(3, dtype=F.dtype)
    return _linear_stress_g(G, materials.lmbda[cid], materials.mu[cid])


@wp.func
def _linear_active_diag(
    F: _mat33, dh: _mat43, materials: _materials, cid: int
) -> _mat43:
    return _linear_hess_diag(
        dh, wp.identity(3, dtype=F.dtype), materials.lmbda[cid], materials.mu[cid]
    )


@wp.func
def _linear_active_prod(
    F: _mat33, p: _mat43, dh: _mat43, materials: _materials, cid: int
) -> _mat43:
    return _linear_hess_prod(
        p, dh, wp.identity(3, dtype=F.dtype), materials.lmbda[cid], materials.mu[cid]
    )


@wp.func
def _linear_active_quad(
    F: _mat33, p: _mat43, dh: _mat43, materials: _materials, cid: int
) -> _floating:
    return _linear_hess_quad(
        p, dh, wp.identity(3, dtype=F.dtype), materials.lmbda[cid], materials.mu[cid]
    )


class _LinearBase(WarpPotentialFem):
    """Kernel boilerplate shared by the two linear material field schemas."""

    energy_density_kernel: ClassVar[wp.Kernel]
    first_piola_kirchhoff_kernel: ClassVar[wp.Kernel]
    fun_kernel: ClassVar[wp.Kernel]
    grad_kernel: ClassVar[wp.Kernel]
    hess_prod_kernel: ClassVar[wp.Kernel]
    hess_diag_kernel: ClassVar[wp.Kernel]
    hess_quad_kernel: ClassVar[wp.Kernel]


@attrs.define
class LinearElastic(_LinearBase):
    class Materials(WarpPotentialFem.Materials):
        lmbda: wp.array
        mu: wp.array

    MATERIAL_FIELDS: ClassVar[dict[str, MaterialField]] = {
        **WarpPotentialFem.MATERIAL_FIELDS,
        LAMBDA.value: MaterialField.CELL.floating(LAMBDA.value),
        MU.value: MaterialField.CELL.floating(MU.value),
    }
    energy_density_func = cast("wp.Function", _linear_passive_energy)
    first_piola_kirchhoff_func = cast("wp.Function", _linear_passive_stress)
    hess_diag_func = cast("wp.Function", _linear_passive_diag)
    hess_prod_func = cast("wp.Function", _linear_passive_prod)
    hess_quad_func = cast("wp.Function", _linear_passive_quad)
    energy_density_kernel = WarpPotentialFem.make_energy_density_kernel(
        energy_density_func
    )
    first_piola_kirchhoff_kernel = WarpPotentialFem.make_first_piola_kirchhoff_kernel(
        first_piola_kirchhoff_func
    )
    fun_kernel = WarpPotentialFem.make_fun_kernel(energy_density_func)
    grad_kernel = WarpPotentialFem.make_grad_kernel(first_piola_kirchhoff_func)
    hess_prod_kernel = WarpPotentialFem.make_hess_prod_kernel(hess_prod_func)
    hess_diag_kernel = WarpPotentialFem.make_hess_diag_kernel(hess_diag_func)
    hess_quad_kernel = WarpPotentialFem.make_hess_quad_kernel(hess_quad_func)


@attrs.define
class LinearElasticActive(_LinearBase):
    class Materials(WarpPotentialFem.Materials):
        activation_inv: wp.array
        lmbda: wp.array
        mu: wp.array

    MATERIAL_FIELDS: ClassVar[dict[str, MaterialField]] = {
        **WarpPotentialFem.MATERIAL_FIELDS,
        ACTIVATION_INV.value: MaterialField(
            ACTIVATION_INV.value,
            lambda dtype: wp.array1d(dtype=wp.types.vector(6, dtype)),
            fem_utils.get_activation_inv,
        ),
        LAMBDA.value: MaterialField.CELL.floating(LAMBDA.value),
        MU.value: MaterialField.CELL.floating(MU.value),
    }
    energy_density_func = cast("wp.Function", _linear_active_energy)
    first_piola_kirchhoff_func = cast("wp.Function", _linear_active_stress)
    hess_diag_func = cast("wp.Function", _linear_active_diag)
    hess_prod_func = cast("wp.Function", _linear_active_prod)
    hess_quad_func = cast("wp.Function", _linear_active_quad)
    energy_density_kernel = WarpPotentialFem.make_energy_density_kernel(
        energy_density_func
    )
    first_piola_kirchhoff_kernel = WarpPotentialFem.make_first_piola_kirchhoff_kernel(
        first_piola_kirchhoff_func
    )
    fun_kernel = WarpPotentialFem.make_fun_kernel(energy_density_func)
    grad_kernel = WarpPotentialFem.make_grad_kernel(first_piola_kirchhoff_func)
    hess_prod_kernel = WarpPotentialFem.make_hess_prod_kernel(hess_prod_func)
    hess_diag_kernel = WarpPotentialFem.make_hess_diag_kernel(hess_diag_func)
    hess_quad_kernel = WarpPotentialFem.make_hess_quad_kernel(hess_quad_func)


@dataclass(frozen=True)
class Case:
    name: str
    energy: Literal["stable", "linear"]
    loss: Literal["l1", "l2", "linf"]
    resolution: Literal["low", "medium", "super_dense"]
    height: float


# One controlled factor changes in each non-baseline case.  ``case=all`` is
# intentional for a full experiment; the normal CLI default runs the baseline.
CASES = (
    Case("stable-l2-medium-h050", "stable", "l2", "medium", 0.050),
    Case("stable-l2-medium-h025", "stable", "l2", "medium", 0.025),
    Case("stable-l2-medium-h100", "stable", "l2", "medium", 0.100),
    Case("stable-l1-medium-h050", "stable", "l1", "medium", 0.050),
    Case("stable-linf-medium-h050", "stable", "linf", "medium", 0.050),
    Case("stable-l2-low-h050", "stable", "l2", "low", 0.050),
    Case("stable-l2-super-dense-h050", "stable", "l2", "super_dense", 0.050),
    Case("linear-l2-medium-h050", "linear", "l2", "medium", 0.050),
)
RESOLUTION = {"low": (32, 5, 32), "medium": (50, 5, 50), "super_dense": (100, 10, 100)}


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)

    case: str = "stable-l2-medium-h050"
    output_dir: Path = cherries.output("20-pork-3d", mkdir=True)
    inverse_steps: int = 600
    learning_rate: float = 0.02
    lr_decay: float = 0.99
    validate_derivatives: bool = True
    require_inverse_convergence: bool = True
    smoke: bool = False


def lame(E: float) -> tuple[float, float]:
    return E * NU / ((1 + NU) * (1 - 2 * NU)), E / (2 * (1 + NU))


def configure() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("3-D Warp inverse physics requires CUDA")
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.float64)
    wp.config.mode = "release"
    wp.init()
    logging.getLogger("liblaf.apple.forward._forward").setLevel(logging.WARNING)
    logging.getLogger("liblaf.apple.inverse._diff_forward").setLevel(logging.WARNING)


def write_json(path: Path, data: Any) -> None:
    def sanitize(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: sanitize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [sanitize(item) for item in value]
        if isinstance(value, np.ndarray):
            return sanitize(value.tolist())
        if isinstance(value, np.generic):
            return sanitize(value.item())
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, Path):
            return str(value)
        return value

    path.write_text(
        json.dumps(sanitize(data), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def structured_tets(
    resolution: tuple[int, int, int], *, smoke: bool
) -> pv.UnstructuredGrid:
    """Six positively-oriented tetrahedra per structured hexahedral cell."""
    nx, ny, nz = (6, 5, 6) if smoke else resolution
    # These planes make the active band exact, rather than centroid-sampled.
    if ny == 5:
        ys = np.array([0.0, 0.02, 0.04, 0.06, 0.08, 0.10])
    else:
        # The explicit concatenation keeps both band interfaces as grid planes.
        ys = np.linspace(0.0, 0.1, ny + 1)
        if not (np.any(np.isclose(ys, 0.04)) and np.any(np.isclose(ys, 0.06))):
            raise ValueError("ny must align the y=0.04 and y=0.06 muscle interfaces")
    xs, zs = np.linspace(0.0, 1.0, nx + 1), np.linspace(0.0, 1.0, nz + 1)
    points = np.array([(x, y, z) for y in ys for z in zs for x in xs], float)

    def vid(i: int, j: int, k: int) -> int:
        return (j * (nz + 1) + k) * (nx + 1) + i

    cells: list[list[int]] = []
    # diagonal v000--v111; orientation is corrected below from signed volume.
    raw = (
        (0, 1, 3, 7),
        (0, 3, 2, 7),
        (0, 2, 6, 7),
        (0, 6, 4, 7),
        (0, 4, 5, 7),
        (0, 5, 1, 7),
    )
    for j in range(len(ys) - 1):
        for k in range(nz):
            for i in range(nx):
                q = (
                    vid(i, j, k),
                    vid(i + 1, j, k),
                    vid(i, j, k + 1),
                    vid(i + 1, j, k + 1),
                    vid(i, j + 1, k),
                    vid(i + 1, j + 1, k),
                    vid(i, j + 1, k + 1),
                    vid(i + 1, j + 1, k + 1),
                )
                for tet0 in raw:
                    tet = [q[a] for a in tet0]
                    a, b, c, d = points[tet]
                    if np.linalg.det(np.stack((b - a, c - a, d - a))) < 0:
                        tet[0], tet[1] = tet[1], tet[0]
                    cells.append(tet)
    packed = np.column_stack(
        (np.full(len(cells), 4, np.int64), np.asarray(cells))
    ).ravel()
    mesh = pv.UnstructuredGrid(packed, np.full(len(cells), pv.CellType.TETRA), points)
    from liblaf.apple.common import FIXED_MASK, FIXED_VALUE

    tol = 1.0e-12
    p = mesh.points
    bottom = np.abs(p[:, 1]) < tol
    side = (
        (np.abs(p[:, 0]) < tol)
        | (np.abs(p[:, 0] - 1) < tol)
        | (np.abs(p[:, 2]) < tol)
        | (np.abs(p[:, 2] - 1) < tol)
    )
    fixed = bottom | side
    mesh.point_data[FIXED_MASK.vtk] = np.repeat(fixed[:, None], 3, axis=1)
    mesh.point_data[FIXED_VALUE.vtk] = np.zeros((mesh.n_points, 3))
    mesh.point_data["FixedBoundary"] = fixed.astype(np.uint8)
    mesh.point_data["TopSurface"] = (np.abs(p[:, 1] - 0.1) < tol).astype(np.uint8)
    mesh.point_data["TargetSurface"] = ((np.abs(p[:, 1] - 0.1) < tol) & ~fixed).astype(
        np.uint8
    )
    centers = mesh.cell_centers().points
    muscle = (centers[:, 1] >= 0.04 - tol) & (centers[:, 1] <= 0.06 + tol)
    if not muscle.any() or muscle.all():
        raise AssertionError("structured band must contain both fat and muscle")
    mesh.cell_data["Muscle"] = muscle.astype(np.uint8)
    return mesh


def set_material(mesh: pv.UnstructuredGrid, E: float, fraction: np.ndarray) -> None:
    from liblaf.apple.common import FRACTION, LAMBDA, MU
    from liblaf.apple.common import NU as NUA
    from liblaf.apple.common import E as YOUNG

    la, mu = lame(E)
    mesh.cell_data[YOUNG.vtk] = np.full(mesh.n_cells, E)
    mesh.cell_data[NUA.vtk] = np.full(mesh.n_cells, NU)
    mesh.cell_data[LAMBDA.vtk] = np.full(mesh.n_cells, la)
    mesh.cell_data[MU.vtk] = np.full(mesh.n_cells, mu)
    mesh.cell_data[FRACTION.vtk] = fraction.astype(float)


def set_output_material_metadata(mesh: pv.UnstructuredGrid) -> None:
    """Restore the effective two-material table after model construction."""
    from liblaf.apple.common import FRACTION
    from liblaf.apple.common import NU as POISSON
    from liblaf.apple.common import E as YOUNG

    muscle = np.asarray(mesh.cell_data["Muscle"], bool)
    young = np.where(muscle, MUSCLE_E, FAT_E)
    mu = young / (2 * (1 + NU))
    la = young * NU / ((1 + NU) * (1 - 2 * NU))
    mesh.cell_data[YOUNG.vtk] = young
    mesh.cell_data[POISSON.vtk] = np.full(mesh.n_cells, NU)
    mesh.cell_data[LAMBDA.vtk] = la
    mesh.cell_data[MU.vtk] = mu
    mesh.cell_data[FRACTION.vtk] = np.ones(mesh.n_cells)
    mesh.cell_data["FatFraction"] = (~muscle).astype(float)
    mesh.cell_data["MuscleFraction"] = muscle.astype(float)


def build_forward(mesh: pv.UnstructuredGrid, energy: str) -> Any:
    from liblaf.apple.forward import Forward, ModelBuilder
    from liblaf.apple.warp.fem import StableNeoHookean, StableNeoHookeanActive

    muscle = np.asarray(mesh.cell_data["Muscle"], bool)
    builder = ModelBuilder()
    builder.add_vertices(mesh)
    builder.add_fixed(mesh)
    set_material(mesh, FAT_E, ~muscle)
    Passive = StableNeoHookean if energy == "stable" else LinearElastic
    Active = StableNeoHookeanActive if energy == "stable" else LinearElasticActive
    builder.add_potential(Passive.from_pyvista(mesh, name="fat"))
    set_material(mesh, MUSCLE_E, muscle)
    from liblaf.apple.common import ACTIVATION_INV

    mesh.cell_data[ACTIVATION_INV.vtk] = np.zeros((mesh.n_cells, 6))
    builder.add_potential(Active.from_pyvista(mesh, name="muscle"))
    forward = Forward(builder.finalize())
    forward.optimizer = forward.default_optimizer(
        max_steps=FORWARD_MAX_STEPS, rtol=FORWARD_RTOL, atol=FORWARD_ATOL
    )
    return forward


class RecordedForward:
    def __init__(self, forward: Any) -> None:
        from liblaf.peach.linalg import FallbackSolver
        from liblaf.peach.linalg.cupy import CupyCG, CupyMinRes

        from liblaf.apple.inverse import DifferentiableForward

        self.impl = DifferentiableForward(forward)
        self.impl.adjoint_solver = FallbackSolver(
            solvers=[
                CupyCG(maxiter=ADJOINT_MAXITER, rtol=ADJOINT_RTOL, atol=0.0),
                CupyMinRes(maxiter=ADJOINT_MAXITER, tol=ADJOINT_RTOL),
            ]
        )
        self.forward_solution = None

    def forward(self, materials: Any) -> torch.Tensor:
        out = self.impl.forward(materials)
        self.forward_solution = self.impl.last_solution
        return out

    @property
    def adjoint_solution(self) -> Any:
        return self.impl.last_adjoint_solution


def full_activation(a: torch.Tensor, ids: torch.Tensor, ncells: int) -> torch.Tensor:
    return torch.zeros((ncells, 6), device=a.device, dtype=a.dtype).index_copy(
        0, ids, a
    )


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
    absolute = float(state.absolute_residuals[best].detach().cpu())
    relative = float(state.relative_residuals[best].detach().cpu())
    return {
        f"{name}/success": bool(solution.success),
        f"{name}/result": str(solution.result),
        f"{name}/best_solver": best,
        f"{name}/absolute_residual": absolute,
        f"{name}/relative_residual": relative,
    }


def target(mesh: pv.UnstructuredGrid, h: float) -> np.ndarray:
    p = mesh.points
    d = np.zeros_like(p)
    top = np.asarray(mesh.point_data["TopSurface"], bool)
    d[top, 1] = h * 16 * p[top, 0] * (1 - p[top, 0]) * p[top, 2] * (1 - p[top, 2])
    return d


def loss_of(residual: torch.Tensor, kind: str) -> torch.Tensor:
    # Exact vector norms, normalized per observed top vertex.
    norms = torch.linalg.vector_norm(residual, dim=1)
    if kind == "l1":
        return norms.mean()
    if kind == "l2":
        return residual.square().sum(dim=1).mean()
    if kind == "linf":
        return norms.max()
    raise ValueError(kind)


def cell_determinants(
    mesh: pv.UnstructuredGrid,
    displacement: np.ndarray,
    activation: np.ndarray,
    energy: str = "stable",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cells = np.asarray(mesh.cells).reshape(-1, 5)[:, 1:]
    x = mesh.points[cells]
    xd = x + displacement[cells]
    dm = np.transpose(x[:, 1:] - x[:, :1], (0, 2, 1))
    ds = np.transpose(xd[:, 1:] - xd[:, :1], (0, 2, 1))
    det_f = np.linalg.det(ds @ np.linalg.inv(dm))
    A = np.zeros((mesh.n_cells, 3, 3))
    A[:, 0, 0] = A[:, 1, 1] = A[:, 2, 2] = 1.0
    A[:, 0, 0] += activation[:, 0]
    A[:, 1, 1] += activation[:, 1]
    A[:, 2, 2] += activation[:, 2]
    A[:, 0, 1] = A[:, 1, 0] = activation[:, 3]
    A[:, 1, 2] = A[:, 2, 1] = activation[:, 4]
    A[:, 0, 2] = A[:, 2, 0] = activation[:, 5]
    det_a = np.linalg.det(A)
    if energy == "stable":
        det_g = det_f * det_a
    else:
        F = ds @ np.linalg.inv(dm)
        det_g = np.linalg.det(F + A - np.eye(3)[None])
    return det_f, det_g, det_a


def activation_neighbor_pairs(mesh: pv.UnstructuredGrid) -> np.ndarray:
    cells = np.asarray(mesh.cells).reshape(-1, 5)[:, 1:]
    active = np.flatnonzero(np.asarray(mesh.cell_data["Muscle"], bool))
    owner: dict[tuple[int, int, int], int] = {}
    pairs = []
    for cid in active:
        tet = cells[cid]
        for omit in range(4):
            face = tuple(sorted(np.delete(tet, omit)))
            if face in owner:
                pairs.append((owner.pop(face), int(cid)))
            else:
                owner[face] = cid
    return np.asarray(pairs, dtype=np.int64).reshape((-1, 2))


def metrics(
    mesh: pv.UnstructuredGrid,
    displacement: np.ndarray,
    activation: np.ndarray,
    target_displacement: np.ndarray | None = None,
    *,
    energy: str = "stable",
    neighbor_pairs: np.ndarray | None = None,
) -> dict[str, float]:
    p, top = mesh.points, np.asarray(mesh.point_data["TopSurface"], bool)
    # Structured x-z diagnostics use physical derivatives and a fixed 0.02
    # Gaussian width, matching the 2-D high-pass definition.
    coords = p[top]
    values = displacement[top, 1]
    xs, zs = (
        sorted({round(x, 12) for x in coords[:, 0]}),
        sorted({round(z, 12) for z in coords[:, 2]}),
    )
    lookup = {
        (round(x, 12), round(z, 12)): y
        for (x, _, z), y in zip(coords, values, strict=True)
    }
    field = np.asarray([[lookup[x, z] for x in xs] for z in zs])
    dx = float(np.mean(np.diff(xs)))
    dz = float(np.mean(np.diff(zs)))
    smooth = ndi.gaussian_filter(
        field, sigma=(max(0.02 / dz, 0.5), max(0.02 / dx, 0.5)), mode="nearest"
    )
    grad_z, grad_x = np.gradient(field, dz, dx)
    dzz, dzx = np.gradient(grad_z, dz, dx)
    dxz, dxx = np.gradient(grad_x, dz, dx)
    lap = dxx + dzz
    active = activation[np.asarray(mesh.cell_data["Muscle"], bool)]
    pairs = (
        activation_neighbor_pairs(mesh) if neighbor_pairs is None else neighbor_pairs
    )
    jumps = (
        activation[pairs[:, 0]] - activation[pairs[:, 1]]
        if pairs.size
        else np.empty((0, 6))
    )
    det_f, det_g, det_a = cell_determinants(mesh, displacement, activation, energy)
    result = {
        "top/highpass_rms": float(np.sqrt(np.mean((field - smooth) ** 2))),
        "top/slope_rms": float(np.sqrt(np.mean(grad_x**2 + grad_z**2))),
        "top/curvature_rms": float(np.sqrt(np.mean(dxx**2 + dzz**2 + dxz**2 + dzx**2))),
        "top/laplacian_rms": float(np.sqrt(np.mean(lap**2))),
        "activation/rms": float(np.sqrt(np.mean(active**2))),
        "activation/max_abs": float(np.max(np.abs(active))),
        "activation/neighbor_jump_rms": float(np.sqrt(np.mean(jumps**2)))
        if jumps.size
        else math.nan,
        "activation/neighbor_jump_max": float(np.max(np.abs(jumps)))
        if jumps.size
        else math.nan,
        "detF/min": float(det_f.min()),
        "detF/max": float(det_f.max()),
        "detG/min": float(det_g.min()),
        "detG/max": float(det_g.max()),
        "detAinv/min": float(det_a.min()),
        "detAinv/max": float(det_a.max()),
    }
    if target_displacement is not None:
        err = np.linalg.norm(
            (displacement - target_displacement)[
                np.asarray(mesh.point_data["TargetSurface"], bool)
            ],
            axis=1,
        )
        result.update(
            {
                "target/mae": float(err.mean()),
                "target/rms": float(np.sqrt(np.mean(err**2))),
                "target/max": float(err.max()),
            }
        )
    return result


def vtk_frame(
    mesh: pv.UnstructuredGrid,
    tgt: np.ndarray,
    u: np.ndarray,
    a: np.ndarray,
    row: dict[str, Any],
    energy: str = "stable",
) -> pv.UnstructuredGrid:
    from liblaf.apple.common import ACTIVATION_INV

    out = mesh.copy(deep=True)
    out.point_data["Displacement"] = u
    out.point_data["TargetDisplacement"] = tgt
    out.point_data["DisplacementError"] = u - tgt
    out.point_data["DeformedPoint"] = mesh.points + u
    out.point_data["TargetPoint"] = mesh.points + tgt
    out.cell_data[ACTIVATION_INV.vtk] = a
    det_f, det_g, det_a = cell_determinants(mesh, u, a, energy)
    out.cell_data["DetF"] = det_f
    out.cell_data["DetG"] = det_g
    out.cell_data["DetAinv"] = det_a
    for key, value in row.items():
        if isinstance(value, (int, float, bool)) and math.isfinite(float(value)):
            out.field_data[key.replace("/", "_")] = np.asarray([value])
    return out


def select(cfg: Config) -> tuple[Case, ...]:
    if cfg.case == "all":
        return CASES
    for c in CASES:
        if c.name == cfg.case:
            return (c,)
    raise ValueError(
        f"unknown case {cfg.case!r}; expected one of {[c.name for c in CASES]} or all"
    )


def derivative_check(energy: str) -> dict[str, Any]:
    """Compare one implicit-adjoint entry with a central finite difference."""
    reference = structured_tets(RESOLUTION["low"], smoke=True)
    n_active = int(np.asarray(reference.cell_data["Muscle"], bool).sum())
    zero = np.zeros((n_active, 6))

    def evaluate(
        values: np.ndarray, *, backward: bool
    ) -> tuple[float, np.ndarray | None, dict[str, Any], dict[str, Any]]:
        # Fresh models are intentional: a differentiable forward owns mutable
        # solver state, so reusing it would contaminate the finite difference.
        mesh = structured_tets(RESOLUTION["low"], smoke=True)
        recorded = RecordedForward(build_forward(mesh, energy))
        base = recorded.impl.model.get_materials()
        ids_np = np.flatnonzero(np.asarray(mesh.cell_data["Muscle"], bool))
        top_np = np.flatnonzero(np.asarray(mesh.point_data["TargetSurface"], bool))
        ids = torch.as_tensor(ids_np, dtype=torch.long)
        top = torch.as_tensor(top_np, dtype=torch.long)
        parameter = torch.tensor(values, requires_grad=backward)
        materials = {key: dict(value) for key, value in base.items()}
        materials["muscle"]["activation_inv"] = full_activation(
            parameter, ids, mesh.n_cells
        )
        with contextlib.redirect_stdout(io.StringIO()):
            displacement = recorded.forward(materials).clone()
        objective = loss_of(
            displacement[top] - torch.as_tensor(target(mesh, 0.05))[top], "l2"
        )
        gradient = None
        if backward:
            objective.backward()
            if parameter.grad is None:
                raise RuntimeError("3-D derivative check produced no gradient")
            gradient = parameter.grad.detach().cpu().numpy()
        return (
            float(objective.detach().cpu()),
            gradient,
            solution_info(recorded.forward_solution, "forward"),
            solution_info(recorded.adjoint_solution, "adjoint"),
        )

    value, gradient, forward_info, adjoint_info = evaluate(zero, backward=True)
    assert gradient is not None
    index = tuple(
        int(item)
        for item in np.unravel_index(np.argmax(np.abs(gradient)), gradient.shape)
    )
    epsilon = 1.0e-4
    minus, plus = zero.copy(), zero.copy()
    minus[index] -= epsilon
    plus[index] += epsilon
    value_minus, _, _, _ = evaluate(minus, backward=False)
    value_plus, _, _, _ = evaluate(plus, backward=False)
    finite_difference = (value_plus - value_minus) / (2 * epsilon)
    analytic = float(gradient[index])
    relative_error = abs(finite_difference - analytic) / max(
        abs(finite_difference), abs(analytic), 1.0e-14
    )
    passed = bool(
        forward_info["forward/success"]
        and adjoint_info["adjoint/success"]
        and relative_error < 0.01
    )
    return {
        "energy": energy,
        "objective": value,
        "component": list(index),
        "epsilon": epsilon,
        "analytic": analytic,
        "finite_difference": finite_difference,
        "relative_error": relative_error,
        "forward_converged": forward_info["forward/success"],
        "adjoint_converged": adjoint_info["adjoint/success"],
        "passed": passed,
    }


def run_case(case: Case, cfg: Config) -> dict[str, Any]:
    mesh = structured_tets(RESOLUTION[case.resolution], smoke=cfg.smoke)
    outdir = cfg.output_dir / case.name
    outdir.mkdir(parents=True, exist_ok=False)
    tgt = target(mesh, case.height)
    forward = RecordedForward(build_forward(mesh, case.energy))
    base = forward.impl.model.get_materials()
    set_output_material_metadata(mesh)
    melon.save(
        vtk_frame(
            mesh, tgt, np.zeros_like(tgt), np.zeros((mesh.n_cells, 6)), {}, case.energy
        ),
        outdir / "target.vtu",
    )
    ids_np = np.flatnonzero(np.asarray(mesh.cell_data["Muscle"], bool))
    top_np = np.flatnonzero(np.asarray(mesh.point_data["TargetSurface"], bool))
    # Global point ids equal mesh ids because only this one mesh is added.
    ids = torch.as_tensor(ids_np, dtype=torch.long)
    top = torch.as_tensor(top_np, dtype=torch.long)
    parameter = torch.nn.Parameter(torch.zeros((len(ids_np), 6)))
    optim = torch.optim.Adam([parameter], lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=cfg.lr_decay)
    target_t = torch.as_tensor(tgt)
    trace: list[dict[str, Any]] = []
    best: tuple[float, np.ndarray, np.ndarray, int] | None = None
    best_converged: tuple[float, np.ndarray, np.ndarray, int] | None = None
    best_orientation_preserving: tuple[float, np.ndarray, np.ndarray, int] | None = None
    frames = 0
    failures = {"forward": 0, "adjoint": 0, "nonfinite": 0}
    series = []
    frames_dir = outdir / "frames"
    frames_dir.mkdir(exist_ok=True)
    neighbor_pairs = activation_neighbor_pairs(mesh)

    def write_trace() -> None:
        if not trace:
            return
        with (outdir / "trace.csv").open("w", newline="") as handle:
            keys = sorted({key for item in trace for key in item})
            writer_csv = csv.DictWriter(handle, fieldnames=keys)
            writer_csv.writeheader()
            writer_csv.writerows(trace)

    for step in range(cfg.inverse_steps + 1):
        optim.zero_grad()
        a = parameter
        materials = {k: dict(v) for k, v in base.items()}
        materials["muscle"]["activation_inv"] = full_activation(a, ids, mesh.n_cells)
        started = time.perf_counter()
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                u = forward.forward(materials).clone()
            residual = u[top] - target_t[top]
            objective = loss_of(residual, case.loss)
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
                **solution_info(forward.forward_solution, "forward"),
                **solution_info(forward.adjoint_solution, "adjoint"),
            }
            trace.append(row)
            write_trace()
            write_json(outdir / "failure.json", row)
            write_json(
                outdir / "history.vtu.series",
                {"file-series-version": "1.0", "files": series},
            )
            # A failed state is recorded and terminates visibly: continuing from
            # undefined autograd state would be a silent fallback.
            raise RuntimeError(
                f"inverse evaluation {step} failed; trace is retained at {outdir}"
            ) from error
        fwd = solution_info(forward.forward_solution, "forward")
        adj = solution_info(forward.adjoint_solution, "adjoint")
        solver_success = bool(fwd["forward/success"] and adj["adjoint/success"])
        failures["forward"] += int(not fwd["forward/success"])
        failures["adjoint"] += int(not adj["adjoint/success"])
        u_np = u.detach().cpu().numpy()
        a_np = full_activation(a.detach(), ids, mesh.n_cells).cpu().numpy()
        value = float(objective.detach().cpu())
        diagnostic = metrics(
            mesh, u_np, a_np, tgt, energy=case.energy, neighbor_pairs=neighbor_pairs
        )
        row = {
            "step": step,
            "evaluation_success": solver_success,
            "loss": value,
            "error_mae": diagnostic["target/mae"],
            "error_rms": diagnostic["target/rms"],
            "error_max": diagnostic["target/max"],
            "top_highpass_rms": diagnostic["top/highpass_rms"],
            "top_slope_rms": diagnostic["top/slope_rms"],
            "top_curvature_rms": diagnostic["top/curvature_rms"],
            "top_laplacian_rms": diagnostic["top/laplacian_rms"],
            "activation_rms": diagnostic["activation/rms"],
            "activation_neighbor_jump_rms": diagnostic["activation/neighbor_jump_rms"],
            "min_det_f": diagnostic["detF/min"],
            "min_det_g": diagnostic["detG/min"],
            "min_det_ainv": diagnostic["detAinv/min"],
            "grad_norm": float(parameter.grad.norm().detach().cpu()),
            "learning_rate": float(optim.param_groups[0]["lr"]),
            "elapsed_s": time.perf_counter() - started,
            **fwd,
            **adj,
        }
        trace.append(row)
        write_trace()
        frame = frames_dir / f"step-{step:05d}.vtu"
        melon.save(vtk_frame(mesh, tgt, u_np, a_np, row, case.energy), frame)
        series.append({"name": str(frame.relative_to(outdir)), "time": float(step)})
        frames += 1
        cherries.set_step(step)
        cherries.log_metrics(
            {f"{case.name}/loss": value, f"{case.name}/error_rms": row["error_rms"]}
        )
        if not solver_success:
            write_json(outdir / "failure.json", row)
            write_json(
                outdir / "history.vtu.series",
                {"file-series-version": "1.0", "files": series},
            )
            raise RuntimeError(
                f"inverse evaluation {step} has a nonconverged forward/adjoint; "
                f"partial trace retained at {outdir}"
            )
        if best is None or value < best[0]:
            best = (value, u_np.copy(), a_np.copy(), step)
        if (
            fwd["forward/success"]
            and adj["adjoint/success"]
            and (best_converged is None or value < best_converged[0])
        ):
            best_converged = (value, u_np.copy(), a_np.copy(), step)
        orientation_preserving = (
            fwd["forward/success"]
            and adj["adjoint/success"]
            and diagnostic["detF/min"] > 0
            and diagnostic["detG/min"] > 0
            and diagnostic["detAinv/min"] > 0
        )
        if orientation_preserving and (
            best_orientation_preserving is None
            or value < best_orientation_preserving[0]
        ):
            best_orientation_preserving = (value, u_np.copy(), a_np.copy(), step)
        if step < cfg.inverse_steps:
            optim.step()
            scheduler.step()  # fixed schedule: no plateau stop.
    assert best is not None
    if best_converged is None:
        raise RuntimeError(
            f"{case.name} produced no converged forward/adjoint evaluation"
        )
    if best_orientation_preserving is None:
        raise RuntimeError(
            f"{case.name} produced no orientation-preserving converged evaluation"
        )
    final_u, final_a = u_np, a_np
    best_mesh = vtk_frame(
        mesh,
        tgt,
        best[1],
        best[2],
        {"best_step": best[3], "best_loss": best[0]},
        case.energy,
    )
    best_converged_mesh = vtk_frame(
        mesh,
        tgt,
        best_converged[1],
        best_converged[2],
        {
            "best_converged_step": best_converged[3],
            "best_converged_loss": best_converged[0],
        },
        case.energy,
    )
    best_orientation_mesh = vtk_frame(
        mesh,
        tgt,
        best_orientation_preserving[1],
        best_orientation_preserving[2],
        {
            "best_orientation_preserving_step": best_orientation_preserving[3],
            "best_orientation_preserving_loss": best_orientation_preserving[0],
        },
        case.energy,
    )
    final_mesh = vtk_frame(mesh, tgt, final_u, final_a, trace[-1], case.energy)
    melon.save(best_mesh, outdir / "best.vtu")
    melon.save(best_converged_mesh, outdir / "best-converged.vtu")
    melon.save(best_orientation_mesh, outdir / "best-orientation-preserving.vtu")
    melon.save(final_mesh, outdir / "final.vtu")
    write_json(
        outdir / "history.vtu.series", {"file-series-version": "1.0", "files": series}
    )
    tail = trace[-min(50, len(trace)) :]
    first_tail = float(tail[0]["loss"])
    last_tail = float(tail[-1]["loss"])
    tail_losses = np.asarray([item["loss"] for item in tail])
    tail_range = float(np.ptp(tail_losses) / max(abs(float(tail_losses.min())), 1e-30))
    tail_valid = all(
        item["forward/success"] and item["adjoint/success"] for item in tail
    )
    tail_gate = bool(tail_valid and tail_range <= 0.01)
    first_inversion = next(
        (
            item["step"]
            for item in trace
            if item["min_det_f"] <= 0
            or item["min_det_g"] <= 0
            or item["min_det_ainv"] <= 0
        ),
        None,
    )
    report = {
        "case": case.__dict__,
        "geometry": {
            "domain": [1, 0.1, 1],
            "structured_resolution": (6, 5, 6)
            if cfg.smoke
            else RESOLUTION[case.resolution],
            "active_band_y": [0.04, 0.06],
            "fixed": "bottom and all four sides, xyz",
            "top_and_interior": "free",
        },
        "materials": {
            "fat": {"E_MPa": FAT_E, "nu": NU},
            "muscle": {"E_MPa": MUSCLE_E, "nu": NU},
            "skin_energy": None,
        },
        "activation": {
            "raw_unbounded_dofs_per_muscle_tet": 6,
            "bounds": None,
            "regularizer": None,
            "det_constraint": None,
        },
        "loss": case.loss,
        "target": "u_y=h*16*x*(1-x)*z*(1-z), u_x=u_z=0",
        "counts": {
            "points": mesh.n_points,
            "tets": mesh.n_cells,
            "muscle_tets": len(ids_np),
            "activation_dofs": int(parameter.numel()),
            "muscle_neighbor_pairs": len(neighbor_pairs),
        },
        "inverse": {
            "schedule": "fixed",
            "evaluations": len(trace),
            "updates": cfg.inverse_steps,
            "early_stopping": None,
            "completion_gate": "all forward/adjoint evaluations valid and final 50-step loss relative range <= 1%",
            "checkpoint_selection_is_posthoc_only": True,
            "best_step": best[3],
            "best_loss": best[0],
            "best_converged_step": best_converged[3],
            "best_converged_loss": best_converged[0],
            "best_orientation_preserving_step": best_orientation_preserving[3],
            "best_orientation_preserving_loss": best_orientation_preserving[0],
            "first_inversion_step": first_inversion,
            "minimum_det_f": min(item["min_det_f"] for item in trace),
            "minimum_det_g": min(item["min_det_g"] for item in trace),
            "minimum_det_ainv": min(item["min_det_ainv"] for item in trace),
            "tail": {
                "window": len(tail),
                "all_forward_adjoint_converged": tail_valid,
                "loss_first": first_tail,
                "loss_last": last_tail,
                "relative_improvement": (first_tail - last_tail)
                / max(abs(first_tail), 1e-30),
                "relative_range": tail_range,
                "grad_norm_last": float(tail[-1]["grad_norm"]),
                "inverse_converged_1pct_tail_gate": tail_gate,
            },
            "forward_pncg": {
                "max_steps": FORWARD_MAX_STEPS,
                "rtol": FORWARD_RTOL,
                "atol": FORWARD_ATOL,
            },
            "failures": failures,
        },
        "paraview": {
            "vtu_series": str(outdir / "history.vtu.series"),
            "fps": 30,
            "frames": frames,
            "source_time": "integer optimization step",
        },
        "artifacts": {
            "series": "history.vtu.series",
            "target": "target.vtu",
            "best": "best.vtu",
            "best_converged": "best-converged.vtu",
            "best_orientation_preserving": "best-orientation-preserving.vtu",
            "final": "final.vtu",
        },
        "metrics": {
            "best": metrics(
                mesh,
                best[1],
                best[2],
                tgt,
                energy=case.energy,
                neighbor_pairs=neighbor_pairs,
            ),
            "best_converged": metrics(
                mesh,
                best_converged[1],
                best_converged[2],
                tgt,
                energy=case.energy,
                neighbor_pairs=neighbor_pairs,
            ),
            "best_orientation_preserving": metrics(
                mesh,
                best_orientation_preserving[1],
                best_orientation_preserving[2],
                tgt,
                energy=case.energy,
                neighbor_pairs=neighbor_pairs,
            ),
            "final": metrics(
                mesh,
                final_u,
                final_a,
                tgt,
                energy=case.energy,
                neighbor_pairs=neighbor_pairs,
            ),
        },
        "trace_csv": str(outdir / "trace.csv"),
        "trace": trace,
    }
    write_json(outdir / "summary.json", report)
    if cfg.require_inverse_convergence and not cfg.smoke and not tail_gate:
        raise RuntimeError(
            f"{case.name} completed {cfg.inverse_steps} updates but failed the 1% "
            "tail gate; increase --inverse-steps and rerun"
        )
    return report


def main(cfg: Config) -> None:
    configure()
    checks = (
        {energy: derivative_check(energy) for energy in ("linear", "stable")}
        if cfg.validate_derivatives
        else {"skipped": True}
    )
    if cfg.validate_derivatives and not all(item["passed"] for item in checks.values()):
        raise RuntimeError(f"derivative check failed: {checks}")
    reports = []
    for case in select(cfg):
        reports.append(run_case(case, cfg))
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        cfg.output_dir / "summary.json",
        {
            "design": "3d-unreachable-pork-controlled-OFAT",
            "derivative_check": checks,
            "cases": reports,
        },
    )


if __name__ == "__main__":
    cherries.main(main)
