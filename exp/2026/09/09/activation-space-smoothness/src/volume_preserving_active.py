"""Historical Raw6 norm term with a penalty on physical det(F).

The finite determinant penalty is evaluated on F in both its linear and quadratic
terms. This experimental material does not impose exact incompressibility or an
inversion barrier. The historical shared material is intentionally unchanged.
"""

from collections.abc import Mapping
from typing import Any, ClassVar, cast

import attrs
import warp as wp

from liblaf.apple.common import ACTIVATION_INV, LAMBDA, MU
from liblaf.apple.warp import math
from liblaf.apple.warp.fem import WarpPotentialFem, func, utils
from liblaf.apple.warp.model import MaterialField

floating = Any
mat33 = Any
mat43 = Any
Materials = Any


@wp.func
def energy_density(F: mat33, materials: Materials, cid: int) -> floating:
    A_inv = func.make_activation_mat33(materials.activation_inv[cid])
    la = materials.lmbda[cid]
    mu = materials.mu[cid]
    G = F @ A_inv
    J = func.I3(F)
    return (
        F.dtype(0.5) * mu * (func.I2(G) - F.dtype(3.0))
        - mu * (J - F.dtype(1.0))
        + F.dtype(0.5) * la * math.square(J - F.dtype(1.0))
    )


@wp.func
def first_piola_kirchhoff(F: mat33, materials: Materials, cid: int) -> mat33:
    A_inv = func.make_activation_mat33(materials.activation_inv[cid])
    la = materials.lmbda[cid]
    mu = materials.mu[cid]
    G = F @ A_inv
    J = func.I3(F)
    determinant_derivative = -mu + la * (J - F.dtype(1.0))
    return mu * G @ wp.transpose(A_inv) + determinant_derivative * func.g3(F)


@wp.func
def hess_diag(
    F: mat33,
    dhdX: mat43,
    materials: Materials,
    cid: int,
    *,
    clamp_lambda: bool = True,  # noqa: ARG001
) -> mat43:
    A_inv = func.make_activation_mat33(materials.activation_inv[cid])
    la = materials.lmbda[cid]
    mu = materials.mu[cid]
    J = func.I3(F)
    determinant_derivative = -mu + la * (J - F.dtype(1.0))
    return (
        la * func.h3_diag(dhdX, func.g3(F))
        + F.dtype(0.5) * mu * func.h5_diag(dhdX @ A_inv)
        + determinant_derivative * func.h6_diag(dhdX, F)
    )


@wp.func
def hess_prod(F: mat33, p: mat43, dhdX: mat43, materials: Materials, cid: int) -> mat43:
    A_inv = func.make_activation_mat33(materials.activation_inv[cid])
    la = materials.lmbda[cid]
    mu = materials.mu[cid]
    J = func.I3(F)
    determinant_derivative = -mu + la * (J - F.dtype(1.0))
    return (
        la * func.h3_prod(p, dhdX, func.g3(F))
        + F.dtype(0.5) * mu * func.h5_prod(p, dhdX @ A_inv)
        + determinant_derivative * func.h6_prod(p, dhdX, F)
    )


@wp.func
def hess_quad(
    F: mat33, p: mat43, dhdX: mat43, materials: Materials, cid: int
) -> floating:
    A_inv = func.make_activation_mat33(materials.activation_inv[cid])
    la = materials.lmbda[cid]
    mu = materials.mu[cid]
    J = func.I3(F)
    determinant_derivative = -mu + la * (J - F.dtype(1.0))
    return (
        la * func.h3_quad(p, dhdX, func.g3(F))
        + F.dtype(0.5) * mu * func.h5_quad(p, dhdX @ A_inv)
        + determinant_derivative * func.h6_quad(p, dhdX, F)
    )


@attrs.define
class StableNeoHookeanActivePhysicalVolume(WarpPotentialFem):
    """Raw6 activation with W = mu/2 |F Ainv|² + g(det F), up to constants."""

    class Materials(WarpPotentialFem.Materials):
        activation_inv: wp.array
        lmbda: wp.array
        mu: wp.array

    MATERIAL_FIELDS: ClassVar[Mapping[str, MaterialField]] = {
        **WarpPotentialFem.MATERIAL_FIELDS,
        ACTIVATION_INV.value: MaterialField(
            name=ACTIVATION_INV.value,
            annotation=lambda dtype: wp.array1d(dtype=wp.types.vector(6, dtype)),
            factory=utils.get_activation_inv,
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
