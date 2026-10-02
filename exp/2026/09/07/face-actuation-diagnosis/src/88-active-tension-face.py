"""Run the target-independent smile-elevator active-tension diagnostic."""

# ruff: noqa: EM101, EM102, PLR0915, TRY003

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, cast, no_type_check

import attrs
import face_physics as fp
import numpy as np
import pydantic_settings as ps
import pyvista as pv
import torch
import warp as wp
from experiment_profile import ProfileCometNoCommit
from face_physics import ForwardConvergenceError

from liblaf import cherries
from liblaf.apple.common import ACTIVATION_INV, FRACTION, GLOBAL_POINT_ID, LAMBDA, MU
from liblaf.apple.torch.fem import Region
from liblaf.apple.warp import math as warp_math
from liblaf.apple.warp.fem import StableNeoHookean, WarpPotentialFem, func, utils
from liblaf.apple.warp.model import MaterialField

LOG = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
REPOSITORY = ROOT.parents[4]
MANUAL_SOURCE = ROOT / "src/10-manual-activation.py"
SMILE_IDS = (57, 58, 63, 64, 142, 143, 218, 219, 283, 284)
GAINS = (0.0, 1.0, 3.0, 10.0)
FIBER_VTK = "ActivationFiber"
TENSION = "active_tension"

floating = Any
mat33 = Any
mat43 = Any
Materials = Any


def load_manual_helpers() -> Any:
    spec = importlib.util.spec_from_file_location("manual_activation_10", MANUAL_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {MANUAL_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MANUAL = load_manual_helpers()


class Config(cherries.BaseConfig):
    """Configuration for the Warp audit and full-face forward diagnostic."""

    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    fixture: Path = (
        Path(__file__).resolve().parents[2]
        / "face-activation-materials/data/10-fixture"
    )
    output_dir: Path = cherries.output("88-active-tension-face", mkdir=True)
    audit_only: bool = False
    gains: tuple[float, ...] = GAINS
    fat_factor: float = 1.0
    muscle_factor: float = 0.8
    soft_nu: float = 0.46
    fat_model: str = "stable"
    fat_nu: float = 0.49
    forward_rtol: float = 1.0e-5
    forward_atol: float = 1.0e-12
    target_name: str = "Smile"
    finite_difference_step: float = 1.0e-5


def get_fiber(region: Region, annotation: Any) -> wp.array:
    values = np.asarray(region.cell_data[FIBER_VTK], dtype=np.float64)
    return wp.from_numpy(values, dtype=annotation.dtype)


def get_tension(region: Region, annotation: Any) -> wp.array:
    values = np.zeros(region.mesh.n_cells, dtype=np.float64)
    return wp.from_numpy(values, dtype=annotation.dtype)


@wp.func
def energy_density(F: mat33, materials: Materials, cid: int) -> floating:
    """Stable passive energy plus bounded explicit reference-fiber tension."""
    la = materials.lmbda[cid]  # float
    mu = materials.mu[cid]  # float
    fiber = materials.fiber[cid]  # vec3
    tension = materials.active_tension[cid]  # float
    I2 = func.I2(F)  # float
    J = func.I3(F)  # float
    Ff = F @ fiber  # vec3
    passive = (
        F.dtype(0.5) * mu * (I2 - F.dtype(3.0))
        - mu * (J - F.dtype(1.0))
        + F.dtype(0.5) * la * warp_math.square(J - F.dtype(1.0))
    )
    active = F.dtype(0.5) * tension * (wp.dot(Ff, Ff) - F.dtype(1.0))
    return passive + active


@wp.func
def first_piola_kirchhoff(F: mat33, materials: Materials, cid: int) -> mat33:
    """Return ``P_passive + T (F f) outer f``."""
    la = materials.lmbda[cid]  # float
    mu = materials.mu[cid]  # float
    fiber = materials.fiber[cid]  # vec3
    tension = materials.active_tension[cid]  # float
    J = func.I3(F)  # float
    dPsi_dI2 = F.dtype(0.5) * mu  # float
    dPsi_dI3 = -mu + la * (J - F.dtype(1.0))  # float
    passive = dPsi_dI2 * func.g2(F) + dPsi_dI3 * func.g3(F)  # mat33
    active = tension * wp.outer(F @ fiber, fiber)  # mat33
    return passive + active


@wp.func
@no_type_check
def hess_diag(
    F: mat33,
    dhdX: mat43,
    materials: Materials,
    cid: int,
    *,
    clamp_lambda: bool = True,  # noqa: ARG001
) -> mat43:
    """Return the exact nodal Hessian diagonal before kernel clamping."""
    la = materials.lmbda[cid]  # float
    mu = materials.mu[cid]  # float
    fiber = materials.fiber[cid]  # vec3
    tension = materials.active_tension[cid]  # float
    J = func.I3(F)  # float
    g3 = func.g3(F)  # mat33
    dPsi_dI2 = F.dtype(0.5) * mu  # float
    dPsi_dI3 = -mu + la * (J - F.dtype(1.0))  # float
    passive = (
        la * func.h3_diag(dhdX, g3)
        + dPsi_dI2 * func.h5_diag(dhdX)
        + dPsi_dI3 * func.h6_diag(dhdX, F)
    )
    q0 = wp.dot(dhdX[0], fiber)
    q1 = wp.dot(dhdX[1], fiber)
    q2 = wp.dot(dhdX[2], fiber)
    q3 = wp.dot(dhdX[3], fiber)
    active = tension * wp.matrix_from_rows(
        wp.vector(q0 * q0, q0 * q0, q0 * q0),
        wp.vector(q1 * q1, q1 * q1, q1 * q1),
        wp.vector(q2 * q2, q2 * q2, q2 * q2),
        wp.vector(q3 * q3, q3 * q3, q3 * q3),
    )
    return passive + active


@wp.func
def hess_prod(
    F: mat33,
    p: mat43,
    dhdX: mat43,
    materials: Materials,
    cid: int,
) -> mat43:
    """Apply the exact assembled tangent to one nodal direction."""
    la = materials.lmbda[cid]  # float
    mu = materials.mu[cid]  # float
    fiber = materials.fiber[cid]  # vec3
    tension = materials.active_tension[cid]  # float
    J = func.I3(F)  # float
    g3 = func.g3(F)  # mat33
    dPsi_dI2 = F.dtype(0.5) * mu  # float
    dPsi_dI3 = -mu + la * (J - F.dtype(1.0))  # float
    passive = (
        la * func.h3_prod(p, dhdX, g3)
        + dPsi_dI2 * func.h5_prod(p, dhdX)
        + dPsi_dI3 * func.h6_prod(p, dhdX, F)
    )
    dF = func.deformation_gradient_jvp(dhdX, p)  # mat33
    active_dP = tension * wp.outer(dF @ fiber, fiber)  # mat33
    active = func.deformation_gradient_vjp(dhdX, active_dP)  # mat43
    return passive + active


@wp.func
def hess_quad(
    F: mat33,
    p: mat43,
    dhdX: mat43,
    materials: Materials,
    cid: int,
) -> floating:
    """Return the exact assembled tangent quadratic for one direction."""
    la = materials.lmbda[cid]  # float
    mu = materials.mu[cid]  # float
    fiber = materials.fiber[cid]  # vec3
    tension = materials.active_tension[cid]  # float
    J = func.I3(F)  # float
    g3 = func.g3(F)  # mat33
    dPsi_dI2 = F.dtype(0.5) * mu  # float
    dPsi_dI3 = -mu + la * (J - F.dtype(1.0))  # float
    passive = (
        la * func.h3_quad(p, dhdX, g3)
        + dPsi_dI2 * func.h5_quad(p, dhdX)
        + dPsi_dI3 * func.h6_quad(p, dhdX, F)
    )
    dF = func.deformation_gradient_jvp(dhdX, p)  # mat33
    dFf = dF @ fiber  # vec3
    return passive + tension * wp.dot(dFf, dFf)


@attrs.define
class StableNeoHookeanActiveTension(WarpPotentialFem):
    """Stable Neo-Hookean material with one additive active-tension term."""

    class Materials(WarpPotentialFem.Materials):
        activation_inv: wp.array
        active_tension: wp.array
        fiber: wp.array
        lmbda: wp.array
        mu: wp.array

    MATERIAL_FIELDS: ClassVar[Mapping[str, MaterialField]] = {
        **WarpPotentialFem.MATERIAL_FIELDS,
        ACTIVATION_INV.value: MaterialField(
            name=ACTIVATION_INV.value,
            annotation=lambda dtype: wp.array1d(dtype=wp.types.vector(6, dtype)),
            factory=utils.get_activation_inv,
        ),
        TENSION: MaterialField(
            name=TENSION,
            annotation=lambda dtype: wp.array1d(dtype=dtype),
            factory=get_tension,
        ),
        "fiber": MaterialField(
            name="fiber",
            annotation=lambda dtype: wp.array1d(dtype=wp.types.vector(3, dtype)),
            factory=get_fiber,
        ),
        LAMBDA.value: MaterialField.CELL.floating(LAMBDA.value),
        MU.value: MaterialField.CELL.floating(MU.value),
    }

    energy_density_func: ClassVar[wp.Function] = cast("wp.Function", energy_density)
    first_piola_kirchhoff_func: ClassVar[wp.Function] = cast(
        "wp.Function", first_piola_kirchhoff
    )
    hess_diag_func: ClassVar[wp.Function] = cast("wp.Function", hess_diag)
    hess_prod_func: ClassVar[wp.Function] = cast("wp.Function", hess_prod)
    hess_quad_func: ClassVar[wp.Function] = cast("wp.Function", hess_quad)

    energy_density_kernel: ClassVar[wp.Kernel] = (
        WarpPotentialFem.make_energy_density_kernel(energy_density_func)
    )
    first_piola_kirchhoff_kernel: ClassVar[wp.Kernel] = (
        WarpPotentialFem.make_first_piola_kirchhoff_kernel(first_piola_kirchhoff_func)
    )
    fun_kernel: ClassVar[wp.Kernel] = WarpPotentialFem.make_fun_kernel(
        energy_density_func
    )
    grad_kernel: ClassVar[wp.Kernel] = WarpPotentialFem.make_grad_kernel(
        first_piola_kirchhoff_func
    )
    hess_prod_kernel: ClassVar[wp.Kernel] = WarpPotentialFem.make_hess_prod_kernel(
        hess_prod_func
    )
    hess_diag_kernel: ClassVar[wp.Kernel] = WarpPotentialFem.make_hess_diag_kernel(
        hess_diag_func
    )
    hess_quad_kernel: ClassVar[wp.Kernel] = WarpPotentialFem.make_hess_quad_kernel(
        hess_quad_func
    )


def write_json(path: Path, data: object) -> None:
    path.write_text(
        json.dumps(
            data,
            indent=2,
            allow_nan=False,
            default=lambda value: (
                value.item() if isinstance(value, np.generic) else str(value)
            ),
        )
        + "\n"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, str | int]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def from_torch_vec3(value: torch.Tensor) -> wp.array:
    dtype = wp.dtype_from_torch(value.dtype)
    return wp.from_torch(value, dtype=wp.types.vector(3, dtype))


def from_torch_float(value: torch.Tensor) -> wp.array:
    return wp.from_torch(value, dtype=wp.dtype_from_torch(value.dtype))


def potential_fun(potential: Any, u: torch.Tensor) -> torch.Tensor:
    output = torch.zeros((1,), dtype=u.dtype, device=u.device)
    potential.fun(from_torch_vec3(u), from_torch_float(output))
    wp.synchronize()
    return output[0]


def potential_grad(potential: Any, u: torch.Tensor) -> torch.Tensor:
    output = torch.zeros_like(u)
    potential.grad(from_torch_vec3(u), from_torch_vec3(output))
    wp.synchronize()
    return output


def potential_hess_diag(potential: Any, u: torch.Tensor) -> torch.Tensor:
    output = torch.zeros_like(u)
    potential.hess_diag(from_torch_vec3(u), from_torch_vec3(output))
    wp.synchronize()
    return output


def potential_hess_prod(
    potential: Any, u: torch.Tensor, direction: torch.Tensor
) -> torch.Tensor:
    output = torch.zeros_like(u)
    potential.hess_prod(
        from_torch_vec3(u),
        from_torch_vec3(direction),
        from_torch_vec3(output),
    )
    wp.synchronize()
    return output


def potential_hess_quad(
    potential: Any, u: torch.Tensor, direction: torch.Tensor
) -> torch.Tensor:
    output = torch.zeros((1,), dtype=u.dtype, device=u.device)
    potential.hess_quad(
        from_torch_vec3(u),
        from_torch_vec3(direction),
        from_torch_float(output),
    )
    wp.synchronize()
    return output[0]


def make_audit_mesh(*, mu: float, lambda_code: float) -> pv.UnstructuredGrid:
    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    mesh = pv.UnstructuredGrid(
        np.asarray((4, 0, 1, 2, 3)),
        np.asarray((pv.CellType.TETRA,), dtype=np.uint8),
        points,
    )
    mesh.cell_data[LAMBDA.vtk] = np.asarray((lambda_code,))
    mesh.cell_data[MU.vtk] = np.asarray((mu,))
    mesh.cell_data[FRACTION.vtk] = np.ones(1)
    mesh.cell_data[ACTIVATION_INV.vtk] = np.zeros((1, 6))
    mesh.cell_data[FIBER_VTK] = np.asarray(((1.0, 0.0, 0.0),))
    mesh.point_data[GLOBAL_POINT_ID.vtk] = np.arange(4, dtype=np.int32)
    return mesh


def warp_derivative_audit(step: float) -> dict[str, object]:
    """Compile and finite-difference all Warp assembly paths on CPU."""
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float64)
    wp.init()
    wp.set_device("cpu")
    young = 0.024
    nu = 0.46
    mu = young / (2.0 * (1.0 + nu))
    lambda_classical = young * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    lambda_code = lambda_classical + mu
    tension = 1.3 * 3.0 * mu
    active = StableNeoHookeanActiveTension.from_pyvista(
        make_audit_mesh(mu=mu, lambda_code=lambda_code),
        name="muscle",
    )
    active.set_materials({TENSION: torch.as_tensor((tension,))})
    zero = StableNeoHookeanActiveTension.from_pyvista(
        make_audit_mesh(mu=mu, lambda_code=lambda_code),
        name="muscle",
    )
    passive = StableNeoHookean.from_pyvista(
        make_audit_mesh(mu=mu, lambda_code=lambda_code),
        name="muscle",
    )
    reference = torch.as_tensor(
        np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    )
    F = torch.as_tensor(((1.08, 0.04, -0.02), (0.01, 0.93, 0.03), (-0.02, 0.02, 1.04)))
    dF = torch.as_tensor(
        ((0.03, -0.04, 0.02), (0.01, 0.05, -0.03), (-0.02, 0.01, 0.04))
    )
    u = reference @ torch.transpose(F - torch.eye(3), 0, 1)
    direction = reference @ torch.transpose(dF, 0, 1)
    energy = potential_fun(active, u)
    gradient = potential_grad(active, u)
    hess_diag_value = potential_hess_diag(active, u)
    hess_prod_value = potential_hess_prod(active, u, direction)
    hess_quad_value = potential_hess_quad(active, u, direction)
    fd_grad_dot = (
        potential_fun(active, u + step * direction)
        - potential_fun(active, u - step * direction)
    ) / (2.0 * step)
    fd_hess_prod = (
        potential_grad(active, u + step * direction)
        - potential_grad(active, u - step * direction)
    ) / (2.0 * step)
    fd_hess_quad = (
        potential_fun(active, u + step * direction)
        - 2.0 * energy
        + potential_fun(active, u - step * direction)
    ) / step**2
    fd_hess_diag = torch.empty_like(u)
    for index in range(u.numel()):
        basis = torch.zeros_like(u).reshape(-1)
        basis[index] = 1.0
        basis = basis.reshape_as(u)
        fd = (
            potential_grad(active, u + step * basis)
            - potential_grad(active, u - step * basis)
        ) / (2.0 * step)
        fd_hess_diag.reshape(-1)[index] = fd.reshape(-1)[index]

    zero_checks = {
        "energy_abs_error": abs(
            float(potential_fun(zero, u) - potential_fun(passive, u))
        ),
        "gradient_max_abs_error": float(
            torch.max(torch.abs(potential_grad(zero, u) - potential_grad(passive, u)))
        ),
        "hess_diag_max_abs_error": float(
            torch.max(
                torch.abs(
                    potential_hess_diag(zero, u) - potential_hess_diag(passive, u)
                )
            )
        ),
        "hess_prod_max_abs_error": float(
            torch.max(
                torch.abs(
                    potential_hess_prod(zero, u, direction)
                    - potential_hess_prod(passive, u, direction)
                )
            )
        ),
        "hess_quad_abs_error": abs(
            float(
                potential_hess_quad(zero, u, direction)
                - potential_hess_quad(passive, u, direction)
            )
        ),
    }
    audit = {
        "device": "cpu",
        "finite_difference_step": step,
        "material": {
            "young_MPa": young,
            "poisson_ratio": nu,
            "mu_MPa": mu,
            "lambda_code_MPa": lambda_code,
            "gain": 1.3,
            "tension_MPa": tension,
        },
        "energy_raw_MPa_m3": float(energy),
        "gradient_directional_abs_error": abs(
            float(torch.sum(gradient * direction) - fd_grad_dot)
        ),
        "hess_prod_max_abs_error": float(
            torch.max(torch.abs(hess_prod_value - fd_hess_prod))
        ),
        "hess_diag_max_abs_error": float(
            torch.max(torch.abs(hess_diag_value - fd_hess_diag))
        ),
        "hess_quad_vs_prod_abs_error": abs(
            float(hess_quad_value - torch.sum(hess_prod_value * direction))
        ),
        "hess_quad_finite_difference_abs_error": abs(
            float(hess_quad_value - fd_hess_quad)
        ),
        "identity_activation_inv_max_abs": float(
            torch.max(
                torch.abs(wp.to_torch(active.get_materials()[ACTIVATION_INV.value]))
            )
        ),
        "zero_tension_passive_restoration": zero_checks,
    }
    if audit["gradient_directional_abs_error"] >= 1.0e-10:
        raise AssertionError("Warp energy gradient directional check failed")
    if audit["hess_prod_max_abs_error"] >= 1.0e-10:
        raise AssertionError("Warp hess_prod check failed")
    if audit["hess_diag_max_abs_error"] >= 1.0e-10:
        raise AssertionError("Warp hess_diag check failed")
    if audit["hess_quad_vs_prod_abs_error"] >= 1.0e-12:
        raise AssertionError("Warp hess_quad disagrees with hess_prod")
    if audit["hess_quad_finite_difference_abs_error"] >= 1.0e-8:
        raise AssertionError("Warp hess_quad finite-difference check failed")
    if any(value != 0.0 for value in zero_checks.values()):
        raise AssertionError("zero tension does not exactly restore passive Warp paths")
    if audit["identity_activation_inv_max_abs"] != 0.0:
        raise AssertionError("activation_inv is not identity encoded")
    return audit


def static_face_preflight(fixture: Path) -> dict[str, object]:
    mesh = pv.read(fixture / "volume.vtu")
    tets = np.asarray(mesh.cells).reshape(-1, 5)[:, 1:]
    active = np.asarray(mesh.cell_data["ActivationMask"], dtype=bool)
    muscle_id = np.asarray(mesh.cell_data["MuscleId"], dtype=int)
    selected = active & np.isin(muscle_id, SMILE_IDS)
    fibers = np.asarray(mesh.cell_data[FIBER_VTK])
    selected_norms = np.linalg.norm(fibers[selected], axis=1)
    fixed = np.asarray(mesh.point_data["FixedMask"], dtype=bool)
    activation_inv = np.asarray(mesh.cell_data[ACTIVATION_INV.vtk])
    assert selected.any()
    assert np.allclose(selected_norms, 1.0, rtol=1.0e-12, atol=1.0e-12)
    assert np.all(activation_inv == 0.0)
    return {
        "n_tets": mesh.n_cells,
        "n_vertices": mesh.n_points,
        "active_tets": int(active.sum()),
        "selected_smile_elevator_tets": int(selected.sum()),
        "selected_smile_elevator_vertices": int(np.unique(tets[selected]).size),
        "fixed_vertices": int(np.any(fixed, axis=1).sum()),
        "smile_ids": SMILE_IDS,
        "selected_fiber_norm": MANUAL.quantiles(selected_norms),
        "fixture_activation_inv_max_abs": float(np.abs(activation_inv).max()),
    }


def batch_cofactor(F: np.ndarray) -> np.ndarray:
    return np.linalg.det(F)[:, None, None] * np.linalg.inv(F).transpose(0, 2, 1)


def stress_diagnostics(
    p: Any,
    u: np.ndarray,
    selected_local: np.ndarray,
    tension_mpa: float,
) -> dict[str, object]:
    selected_cells = p.ids[selected_local]
    F = MANUAL.deformation_gradients(p, u)[selected_cells]
    J = np.linalg.det(F)
    fibers = np.asarray(p.mesh.cell_data[FIBER_VTK])[selected_cells]
    mu = float(p.material_spec["muscle_mu_code_MPa"])
    la = float(p.material_spec["muscle_lambda_code_MPa"])
    cof = batch_cofactor(F)
    passive_P = mu * F + (-mu + la * (J - 1.0))[:, None, None] * cof
    Ff = np.einsum("nij,nj->ni", F, fibers)
    active_P = tension_mpa * np.einsum("ni,nj->nij", Ff, fibers)
    active_cauchy = np.einsum("nij,nkj->nik", active_P, F) / J[:, None, None]
    total_P = passive_P + active_P
    weights = (
        p.volumes_all[selected_cells]
        * np.asarray(p.mesh.cell_data["MuscleFraction"])[selected_cells]
    )
    weights /= weights.sum()

    def norm_stats(value: np.ndarray) -> dict[str, float]:
        norms = np.linalg.norm(value.reshape(len(value), -1), axis=1)
        result = MANUAL.quantiles(norms)
        result["fraction_volume_weighted_mean"] = float(np.sum(weights * norms))
        return result

    current_fiber = Ff / np.linalg.norm(Ff, axis=1, keepdims=True)
    active_fiber_cauchy = np.einsum(
        "ni,nij,nj->n", current_fiber, active_cauchy, current_fiber
    )
    return {
        "units": "MPa",
        "tension_parameter_MPa": tension_mpa,
        "passive_first_piola_frobenius": norm_stats(passive_P),
        "active_first_piola_frobenius": norm_stats(active_P),
        "total_first_piola_frobenius": norm_stats(total_P),
        "active_current_fiber_cauchy": MANUAL.quantiles(active_fiber_cauchy),
    }


def save_case(
    p: Any,
    path: Path,
    u: np.ndarray,
    gain: float,
    tension_mpa: float,
    selected_local: np.ndarray,
    diagnostics: dict[str, object],
    arrays: dict[str, np.ndarray],
) -> None:
    path.mkdir(parents=True, exist_ok=False)
    identity = np.broadcast_to(np.eye(3), (len(p.ids), 3, 3)).copy()
    p.save_mesh(path / "state.vtu", u, identity)
    mesh = pv.read(path / "state.vtu")
    selected_cells = p.ids[selected_local]
    pattern = np.zeros(mesh.n_cells, dtype=np.int8)
    pattern[selected_cells] = 1
    gain_field = np.zeros(mesh.n_cells)
    gain_field[selected_cells] = gain
    tension_field = np.zeros(mesh.n_cells)
    tension_field[selected_cells] = tension_mpa
    mesh.cell_data["ActiveTensionPattern"] = pattern
    mesh.cell_data["ActiveTensionGain"] = gain_field
    mesh.cell_data["ActiveTensionMPa"] = tension_field
    mesh.save(path / "state.vtu")
    np.savez_compressed(
        path / "state.npz",
        u=u,
        gain=np.asarray(gain),
        tension_MPa=np.asarray(tension_mpa),
        **arrays,
    )
    write_json(path / "diagnostics.json", diagnostics)


def snapshot_provenance(out: Path, fixture: Path) -> dict[str, object]:
    source_dir = out / "sources"
    source_dir.mkdir()
    paths = {
        "executed": Path(__file__),
        "face_physics": ROOT / "src/face_physics.py",
        "manual_helpers": MANUAL_SOURCE,
        "passive_warp": REPOSITORY / "src/liblaf/apple/warp/fem/_stable_neo_hookean.py",
        "active_strain_warp_for_comparison": REPOSITORY
        / "src/liblaf/apple/warp/fem/_stable_neo_hookean_active.py",
    }
    sources = {}
    for name, path in paths.items():
        destination = source_dir / path.name
        shutil.copy2(path, destination)
        sources[name] = record(destination)
    inputs = {
        name: record(fixture / filename)
        for name, filename in {
            "fixture_volume": "volume.vtu",
            "fixture_skin": "skin.vtp",
            "fixture_summary": "summary.json",
        }.items()
    }
    return {"sources": sources, "inputs": inputs}


def write_manifest(out: Path, provenance: dict[str, object]) -> None:
    files = {}
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files[str(path.relative_to(out))] = record(path)
    write_json(
        out / "manifest.json",
        {
            "schema_version": 1,
            "scope": "Warp active-tension audit and target-independent face diagnostic",
            "provenance": provenance,
            "files": files,
        },
    )


def run_face(cfg: Config, out: Path, preflight: dict[str, object]) -> dict[str, object]:
    if tuple(cfg.gains) != GAINS:
        raise ValueError(f"diagnostic gains must remain exactly {GAINS}")
    if cfg.forward_rtol != 1.0e-5 or cfg.forward_atol != 1.0e-12:
        raise ValueError("diagnostic must retain the manual forward tolerances")
    if cfg.muscle_factor != 0.8 or cfg.soft_nu != 0.46:
        raise ValueError("diagnostic must retain E=0.024 MPa and nu=0.46")
    fp.StableNeoHookeanActive = StableNeoHookeanActiveTension
    fp.configure()
    wp.set_device("cuda:0")
    p = fp.FacePhysics(
        cfg.fixture,
        skin_factor=0.0,
        fat_factor=cfg.fat_factor,
        muscle_factor=cfg.muscle_factor,
        rtol=cfg.forward_rtol,
        atol=cfg.forward_atol,
        soft_nu=cfg.soft_nu,
        target_name=cfg.target_name,
        fat_model=cfg.fat_model,
        fat_nu=cfg.fat_nu,
    )
    p.material_spec["muscle_model"] = "stable-active-tension"
    p.material_spec["active_tension_energy"] = "T/2*(f^T F^T F f-1)"
    p.material_spec["skin_enabled"] = False
    active_muscle_id = np.asarray(p.mesh.cell_data["MuscleId"], dtype=int)[p.ids]
    selected_local = np.flatnonzero(np.isin(active_muscle_id, SMILE_IDS))
    if len(selected_local) != preflight["selected_smile_elevator_tets"]:
        raise AssertionError("runtime smile-elevator selection differs from preflight")
    reference_tension = 3.0 * float(p.material_spec["muscle_mu_code_MPa"])
    activation_identity = torch.zeros((len(p.ids), 6))
    zero_seed = np.zeros_like(p.points)
    rows: list[dict[str, object]] = []
    start = time.perf_counter()
    for case_index, gain in enumerate(cfg.gains):
        tension_mpa = gain * reference_tension
        full_tension = torch.zeros(p.mesh.n_cells)
        full_tension[p.id_t[selected_local]] = tension_mpa
        p.materials["muscle"][TENSION] = full_tension
        case_name = f"gain-{round(100 * gain):04d}"
        LOG.info(
            "Solving %s from zero displacement: %d selected cells, T=%.9g MPa",
            case_name,
            len(selected_local),
            tension_mpa,
        )
        try:
            u = p.solve(activation_identity, zero_seed).detach().cpu().numpy()
        except ForwardConvergenceError as error:
            failure = {
                "case": case_name,
                "gain": gain,
                "tension_MPa": tension_mpa,
                "seed": "zero displacement",
                "error": str(error),
                "forward": p.last_forward,
            }
            path = out / "cases" / case_name
            path.mkdir(parents=True, exist_ok=False)
            write_json(path / "failure.json", failure)
            write_json(out / "failure.json", failure)
            raise
        identity = np.broadcast_to(np.eye(3), (len(p.ids), 3, 3)).copy()
        diagnostics, arrays = MANUAL.case_diagnostics(p, u, identity, selected_local)
        diagnostics.update(
            {
                "case": case_name,
                "gain": gain,
                "tension_MPa": tension_mpa,
                "reference_tension_MPa": reference_tension,
                "muscle_ids": SMILE_IDS,
                "seed": "zero displacement; no continuation from another gain",
                "target_role": "post-hoc motion projection only; absent from energy, control, and equilibrium solve",
                "activation_inverse": {
                    "encoding_max_abs": 0.0,
                    "matrix": "identity for every cell",
                },
                "stress": stress_diagnostics(p, u, selected_local, tension_mpa),
                "forward": p.last_forward,
            }
        )
        save_case(
            p,
            out / "cases" / case_name,
            u,
            gain,
            tension_mpa,
            selected_local,
            diagnostics,
            arrays,
        )
        physical = diagnostics["physical_deformation"]
        surface = diagnostics["surface_motion"]
        muscle = diagnostics["selected_muscle_motion"]
        lip = diagnostics["lip_motion"]
        row = {
            "case": case_name,
            "gain": gain,
            "tension_MPa": tension_mpa,
            "selected_cells": len(selected_local),
            "forward_steps": p.last_forward["steps"],
            "forward_grad_norm": p.last_forward["grad_norm"],
            "detF_min_all": physical["detF_all"]["min"],
            "detF_q01_all": physical["detF_all"]["q01"],
            "inverted_tets_all": physical["inverted_tets_all"],
            "fiber_stretch_median": physical["fiber_stretch_F_selected"]["median"],
            "fiber_stretch_weighted_mean": physical[
                "fiber_stretch_F_selected_fraction_volume_weighted_mean"
            ],
            "muscle_centroid_rms_mm": muscle["centroid_displacement_rms_mm"],
            "surface_rms_mm": surface["weighted_rms_mm"],
            "surface_max_mm": surface["weighted_max_mm"],
            "smile_projection": surface["smile_target_projection_amplitude"],
            "lip_rms_mm": lip["rms_mm"],
            "lip_radial_outward_mean_mm": lip["mean_outward_radial_xy_mm"],
            "wall_s": time.perf_counter() - start,
        }
        rows.append(row)
        with (out / "trace.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(rows)
        cherries.log_metrics(
            {
                "active_tension_face/surface_rms_mm": row["surface_rms_mm"],
                "active_tension_face/smile_projection": row["smile_projection"],
                "active_tension_face/detF_min": row["detF_min_all"],
                "active_tension_face/fiber_stretch_median": row["fiber_stretch_median"],
            },
            step=case_index,
        )
        LOG.info("Completed %s: %s", case_name, row)
    return {
        "status": "completed",
        "scope": "target-independent no-skin forward equilibrium; no inverse fit",
        "case_count": len(rows),
        "preflight": preflight,
        "gains": cfg.gains,
        "smile_ids": SMILE_IDS,
        "reference_tension_MPa": reference_tension,
        "materials": p.material_spec,
        "solver": {
            "max_steps": 10000,
            "rtol": cfg.forward_rtol,
            "atol": cfg.forward_atol,
            "initialization": "Every gain starts from the same zero displacement seed.",
            "continuation": False,
            "inverse_optimization": False,
            "target_in_equilibrium": False,
        },
        "activation_inverse": "identity for every cell and every gain",
        "rows": rows,
        "wall_s": time.perf_counter() - start,
    }


def main(cfg: Config) -> None:
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("choose an empty output directory")
    write_json(out / "config.json", cfg.model_dump(mode="json"))
    provenance = snapshot_provenance(out, cfg.fixture)
    write_json(out / "provenance.json", provenance)
    preflight = static_face_preflight(cfg.fixture)
    audit = warp_derivative_audit(cfg.finite_difference_step)
    write_json(out / "warp-audit.json", audit)
    if cfg.audit_only:
        summary = {
            "status": "audit-completed",
            "scope": "CPU Warp constitutive and assembly audit; no face solve",
            "preflight": preflight,
            "warp_audit": audit,
        }
    else:
        summary = run_face(cfg, out, preflight)
        summary["warp_audit"] = audit
    summary["git_sha"] = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
    ).strip()
    write_json(out / "summary.json", summary)
    write_manifest(out, provenance)
    LOG.info("Wrote %s", out / "summary.json")


if __name__ == "__main__":
    cherries.main(
        main, profile=None if os.getenv("DEBUG") == "1" else ProfileCometNoCommit
    )
