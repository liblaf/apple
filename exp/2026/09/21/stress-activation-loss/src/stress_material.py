"""Stable Neo-Hookean bulk material with unconstrained additive stress."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, cast, no_type_check

import attrs
import warp as wp

from liblaf.apple.common import LAMBDA, MU
from liblaf.apple.torch.fem import Region
from liblaf.apple.warp import math as warp_math
from liblaf.apple.warp.fem import WarpPotentialFem, func
from liblaf.apple.warp.model import MaterialField

ACTIVE_STRESS = "active_stress"
floating = Any
mat33 = Any
mat43 = Any
Materials = Any


def get_active_stress(region: Region, annotation: Any) -> wp.array:
    """Use zero additive stress unless the caller supplies a material tensor."""
    return wp.zeros((region.mesh.n_cells,), dtype=annotation.dtype)


@wp.func
def energy_density(F: mat33, materials: Materials, cid: int) -> floating:
    """Return W_SNH + 1/2 Q:(F^T F-I) for any finite symmetric Q."""
    la, mu, Q = (
        materials.lmbda[cid],
        materials.mu[cid],
        materials.active_stress[cid],
    )
    J = func.I3(F)
    passive = (
        F.dtype(0.5) * mu * (func.I2(F) - F.dtype(3.0))
        - mu * (J - F.dtype(1.0))
        + F.dtype(0.5) * la * warp_math.square(J - F.dtype(1.0))
    )
    return passive + F.dtype(0.5) * wp.ddot(
        Q, wp.transpose(F) @ F - wp.identity(3, dtype=F.dtype)
    )


@wp.func
def first_piola_kirchhoff(F: mat33, materials: Materials, cid: int) -> mat33:
    """Return P_SNH + FQ under the caller-owned symmetric-Q contract."""
    la, mu, Q = (
        materials.lmbda[cid],
        materials.mu[cid],
        materials.active_stress[cid],
    )
    J = func.I3(F)
    passive = F.dtype(0.5) * mu * func.g2(F) + (
        -mu + la * (J - F.dtype(1.0))
    ) * func.g3(F)
    return passive + F @ Q


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
    la, mu, Q = (
        materials.lmbda[cid],
        materials.mu[cid],
        materials.active_stress[cid],
    )
    J = func.I3(F)
    passive = (
        la * func.h3_diag(dhdX, func.g3(F))
        + F.dtype(0.5) * mu * func.h5_diag(dhdX)
        + (-mu + la * (J - F.dtype(1.0))) * func.h6_diag(dhdX, F)
    )
    q0, q1, q2, q3 = (
        wp.dot(dhdX[0], Q @ dhdX[0]),
        wp.dot(dhdX[1], Q @ dhdX[1]),
        wp.dot(dhdX[2], Q @ dhdX[2]),
        wp.dot(dhdX[3], Q @ dhdX[3]),
    )
    active = wp.matrix_from_rows(
        wp.vector(q0, q0, q0),
        wp.vector(q1, q1, q1),
        wp.vector(q2, q2, q2),
        wp.vector(q3, q3, q3),
    )
    return passive + active


@wp.func
def hess_prod(F: mat33, p: mat43, dhdX: mat43, materials: Materials, cid: int) -> mat43:
    la, mu, Q = (
        materials.lmbda[cid],
        materials.mu[cid],
        materials.active_stress[cid],
    )
    J = func.I3(F)
    passive = (
        la * func.h3_prod(p, dhdX, func.g3(F))
        + F.dtype(0.5) * mu * func.h5_prod(p, dhdX)
        + (-mu + la * (J - F.dtype(1.0))) * func.h6_prod(p, dhdX, F)
    )
    dF = func.deformation_gradient_jvp(dhdX, p)
    return passive + func.deformation_gradient_vjp(dhdX, dF @ Q)


@wp.func
def hess_quad(
    F: mat33, p: mat43, dhdX: mat43, materials: Materials, cid: int
) -> floating:
    la, mu, Q = (
        materials.lmbda[cid],
        materials.mu[cid],
        materials.active_stress[cid],
    )
    J = func.I3(F)
    passive = (
        la * func.h3_quad(p, dhdX, func.g3(F))
        + F.dtype(0.5) * mu * func.h5_quad(p, dhdX)
        + (-mu + la * (J - F.dtype(1.0))) * func.h6_quad(p, dhdX, F)
    )
    dF = func.deformation_gradient_jvp(dhdX, p)
    return passive + wp.ddot(dF, dF @ Q)


@attrs.define
class StableNeoHookeanStress(WarpPotentialFem):
    """Polynomial Stable Neo-Hookean material with signed symmetric cell Q."""

    class Materials(WarpPotentialFem.Materials):
        active_stress: wp.array
        lmbda: wp.array
        mu: wp.array

    MATERIAL_FIELDS: ClassVar[Mapping[str, MaterialField]] = {
        **WarpPotentialFem.MATERIAL_FIELDS,
        ACTIVE_STRESS: MaterialField(
            name=ACTIVE_STRESS,
            annotation=lambda dtype: wp.array1d(dtype=wp.types.matrix((3, 3), dtype)),
            factory=get_active_stress,
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


__all__ = ["ACTIVE_STRESS", "StableNeoHookeanStress"]
