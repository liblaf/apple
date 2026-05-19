from collections.abc import Mapping
from typing import Any, ClassVar, cast

import attrs
import warp as wp

from liblaf.apple.common import LAMBDA, MU
from liblaf.apple.warp import math
from liblaf.apple.warp.model import MaterialField

from . import func
from ._base import WarpPotentialFem

floating = Any
mat33 = Any
mat43 = Any
Materials = Any


@wp.func
def energy_density(F: mat33, materials: Materials, cid: int) -> floating:
    la = materials.lmbda[cid]  # float
    mu = materials.mu[cid]  # float
    I2 = func.I2(F)  # float
    J = func.I3(F)  # float
    return (
        F.dtype(0.5) * mu * (I2 - F.dtype(3.0))
        - mu * (J - F.dtype(1.0))
        + F.dtype(0.5) * la * math.square(J - F.dtype(1.0))
    )


@wp.func
def first_piola_kirchhoff(F: mat33, materials: Materials, cid: int) -> mat33:
    la = materials.lmbda[cid]  # float
    mu = materials.mu[cid]  # float
    J = func.I3(F)  # float
    dPsi_dI2 = F.dtype(0.5) * mu  # float
    dPsi_dI3 = -mu + la * (J - F.dtype(1.0))  # float
    g2 = func.g2(F)  # mat33
    g3 = func.g3(F)  # mat33
    return dPsi_dI2 * g2 + dPsi_dI3 * g3


@wp.func
def hess_diag(
    F: mat33,
    dhdX: mat43,
    materials: Materials,
    cid: int,
    *,
    clamp_lambda: bool = True,  # noqa: ARG001
) -> mat33:
    la = materials.lmbda[cid]  # float
    mu = materials.mu[cid]  # float
    J = func.I3(F)  # float
    # g2 = func.g2(F)  # mat33
    g3 = func.g3(F)  # mat33
    dPsi_dI2 = F.dtype(0.5) * mu  # float
    dPsi_dI3 = -mu + la * (J - F.dtype(1.0))  # float
    # d2Psi_dI22 = F.dtype(0.0)  # float
    d2Psi_dI32 = la  # float
    # h2_diag = func.h2_diag(dhdX, g2)  # mat43
    h3_diag = func.h3_diag(dhdX, g3)  # mat43
    h5_diag = func.h5_diag(dhdX)  # mat43
    h6_diag = func.h6_diag(dhdX, F)  # mat43
    return (
        # d2Psi_dI22 * h2_diag
        d2Psi_dI32 * h3_diag + dPsi_dI2 * h5_diag + dPsi_dI3 * h6_diag
    )


@wp.func
def hess_prod(F: mat33, p: mat43, dhdX: mat43, materials: Materials, cid: int) -> mat33:
    la = materials.lmbda[cid]  # float
    mu = materials.mu[cid]  # float
    J = func.I3(F)  # float
    # g2 = func.g2(F)  # mat33
    g3 = func.g3(F)  # mat33
    dPsi_dI2 = F.dtype(0.5) * mu  # float
    dPsi_dI3 = -mu + la * (J - F.dtype(1.0))  # float
    # d2Psi_dI22 = F.dtype(0.0)  # float
    d2Psi_dI32 = la  # float
    h3_prod = func.h3_prod(p, dhdX, g3)  # mat43
    h5_prod = func.h5_prod(p, dhdX)  # mat43
    h6_prod = func.h6_prod(p, dhdX, F)  # mat43
    return d2Psi_dI32 * h3_prod + dPsi_dI2 * h5_prod + dPsi_dI3 * h6_prod


@wp.func
def hess_quad(
    F: mat33, p: mat43, dhdX: mat43, materials: Materials, cid: int
) -> floating:
    la = materials.lmbda[cid]  # float
    mu = materials.mu[cid]  # float
    J = func.I3(F)  # float
    # g2 = func.g2(F)  # mat33
    g3 = func.g3(F)  # mat33
    dPsi_dI2 = F.dtype(0.5) * mu  # float
    dPsi_dI3 = -mu + la * (J - F.dtype(1.0))  # float
    # d2Psi_dI22 = F.dtype(0.0)  # float
    d2Psi_dI32 = la  # float
    h3_quad = func.h3_quad(p, dhdX, g3)  # float
    h5_quad = func.h5_quad(p, dhdX)  # float
    h6_quad = func.h6_quad(p, dhdX, F)  # float
    return d2Psi_dI32 * h3_quad + dPsi_dI2 * h5_quad + dPsi_dI3 * h6_quad


@attrs.define
class StableNeoHookean(WarpPotentialFem):
    class Materials(WarpPotentialFem.Materials):
        lmbda: wp.array
        mu: wp.array

    MATERIAL_FIELDS: ClassVar[Mapping[str, MaterialField]] = {
        **WarpPotentialFem.MATERIAL_FIELDS,
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
