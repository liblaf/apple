from typing import Any, ClassVar, cast, override

import warp as wp

from liblaf import jarp
from liblaf.apple.warp import math

from . import func, utils
from ._base import WarpPotentialFem

floating = Any
mat33 = Any
mat43 = Any
Materials = Any


@wp.func
def energy_density(F: mat33, materials: Materials, cid: int) -> floating:
    mu = materials.mu[cid]  # float
    R, _S = math.polar_rv(F)
    return F.dtype(0.5) * mu * math.fro_norm_square(F - R)


@wp.func
def first_piola_kirchhoff(F: mat33, materials: Materials, cid: int) -> mat33:
    mu = materials.mu[cid]  # float
    R, _S = math.polar_rv(F)
    return mu * (F - R)


@wp.func
def hess_diag(
    F: mat33, dhdX: mat43, materials: Materials, cid: int, *, clamp_lambda: bool = True
) -> mat33:
    mu = materials.mu[cid]  # float
    U, sigma, V = math.svd_rv(F)  # mat33, vec3, mat33
    h4_diag = func.h4_diag(dhdX, U, sigma, V, clamp_lambda=clamp_lambda)  # mat43
    h5_diag = func.h5_diag(dhdX)  # mat43
    h_diag = -F.dtype(2.0) * h4_diag + h5_diag  # mat43
    return F.dtype(0.5) * mu * h_diag  # mat33


@wp.func
def hess_prod(
    F: mat33,
    p: mat33,
    dhdX: mat43,
    materials: Materials,
    cid: int,
    *,
    clamp_lambda: bool = True,
) -> mat33:
    mu = materials.mu[cid]  # float
    U, sigma, V = math.svd_rv(F)  # mat33, vec3, mat33
    h4_prod = func.h4_prod(dhdX, p, U, sigma, V, clamp_lambda=clamp_lambda)  # mat43
    h5_prod = func.h5_prod(dhdX, p)  # mat43
    h_prod = -F.dtype(2.0) * h4_prod + h5_prod  # mat43
    return F.dtype(0.5) * mu * h_prod  # mat33


@wp.func
def hess_quad(
    F: mat33,
    v: mat43,
    dhdX: mat43,
    materials: Materials,
    cid: int,
    *,
    clamp_lambda: bool = True,
) -> floating:
    mu = materials.mu[cid]  # float
    U, sigma, V = math.svd_rv(F)  # mat33, vec3, mat33
    h4_quad = func.h4_quad(v, dhdX, U, sigma, V, clamp_lambda=clamp_lambda)  # float
    h5_quad = func.h5_quad(v, dhdX)  # float
    h_quad = -F.dtype(2.0) * h4_quad + h5_quad  # float
    return F.dtype(0.5) * mu * h_quad  # float


@jarp.frozen_static
class Arap(WarpPotentialFem):
    @jarp.struct
    class Materials:
        mu: wp.array[floating]

        @classmethod
        def __annotations_factory__(cls, dtype: Any) -> dict[str, Any]:
            return {"mu": wp.array1d(dtype=dtype)}

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

    @classmethod
    @override
    def materials_from_region(cls, region: Any, requires_grad: Any) -> Materials:
        materials: cls.Materials = cls.Materials()
        materials.mu = utils.get_mu(region)
        return materials
