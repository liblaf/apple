from collections.abc import Sequence
from typing import Any, ClassVar, cast, no_type_check, override

import jarp
import jarp.warp.types as wpt
import warp as wp

from liblaf.apple.jax import Region
from liblaf.apple.warp.energies.elastic import utils

from . import func
from ._base import WarpElastic

int_ = Any
float_ = Any
mat33 = Any
mat43 = Any
Materials = Any


@wp.func
@no_type_check
def _stable_neo_hookean_energy_density_func(
    F: mat33, materials: Materials, cid: int_
) -> float_:
    fraction = materials.fraction[cid]
    mu = materials.mu[cid]
    lambda_ = materials.lambda_[cid]
    J = func.I3(F)
    J_minus_1 = J - F.dtype(1.0)
    return (
        F.dtype(0.5)
        * fraction
        * (
            mu * (func.I2(F) - F.dtype(3.0))
            - F.dtype(2.0) * mu * J_minus_1
            + lambda_ * J_minus_1 * J_minus_1
        )
    )


@wp.func
@no_type_check
def _stable_neo_hookean_first_piola_kirchhoff_func(
    F: mat33, materials: Materials, cid: int_
) -> mat33:
    fraction = materials.fraction[cid]
    mu = materials.mu[cid]
    lambda_ = materials.lambda_[cid]
    J = func.I3(F)
    c = -mu + lambda_ * (J - F.dtype(1.0))
    return fraction * (mu * F + c * func.g3(F))


@wp.func
@no_type_check
def _stable_neo_hookean_hess_diag_func(
    F: mat33,
    dhdX: mat43,
    materials: Materials,
    cid: int_,
    *,
    clamp: bool = False,  # noqa: ARG001
) -> mat33:
    fraction = materials.fraction[cid]
    mu = materials.mu[cid]
    lambda_ = materials.lambda_[cid]
    J = func.I3(F)
    c = -mu + lambda_ * (J - F.dtype(1.0))
    h3_diag = func.h3_diag(dhdX, func.g3(F))
    h5_diag = func.h5_diag(dhdX)
    h6_diag = func.h6_diag(dhdX, F)
    return fraction * (
        lambda_ * h3_diag + F.dtype(0.5) * mu * h5_diag + c * h6_diag
    )


@wp.func
@no_type_check
def _stable_neo_hookean_hess_prod_func(
    F: mat33,
    v: mat43,
    dhdX: mat43,
    materials: Materials,
    cid: int_,
    *,
    clamp: bool = False,  # noqa: ARG001
) -> mat43:
    fraction = materials.fraction[cid]
    mu = materials.mu[cid]
    lambda_ = materials.lambda_[cid]
    J = func.I3(F)
    c = -mu + lambda_ * (J - F.dtype(1.0))
    h3_prod = func.h3_prod(v, dhdX, func.g3(F))
    h5_prod = func.h5_prod(v, dhdX)
    h6_prod = func.h6_prod(v, dhdX, F)
    return fraction * (
        lambda_ * h3_prod + F.dtype(0.5) * mu * h5_prod + c * h6_prod
    )


@wp.func
@no_type_check
def _stable_neo_hookean_hess_quad_func(
    F: mat33,
    v: mat43,
    dhdX: mat43,
    materials: Materials,
    cid: int_,
    *,
    clamp: bool = False,  # noqa: ARG001
) -> float_:
    fraction = materials.fraction[cid]
    mu = materials.mu[cid]
    lambda_ = materials.lambda_[cid]
    J = func.I3(F)
    c = -mu + lambda_ * (J - F.dtype(1.0))
    h3_quad = func.h3_quad(v, dhdX, func.g3(F))
    h5_quad = func.h5_quad(v, dhdX)
    h6_quad = func.h6_quad(v, dhdX, F)
    return fraction * (
        lambda_ * h3_quad + F.dtype(0.5) * mu * h5_quad + c * h6_quad
    )


@jarp.frozen_static
class WarpStableNeoHookean(WarpElastic):
    energy_density_func: ClassVar[wp.Function] = cast(
        "wp.Function", _stable_neo_hookean_energy_density_func
    )
    first_piola_kirchhoff_func: ClassVar[wp.Function] = cast(
        "wp.Function", _stable_neo_hookean_first_piola_kirchhoff_func
    )
    hess_diag_func: ClassVar[wp.Function] = cast(
        "wp.Function", _stable_neo_hookean_hess_diag_func
    )
    hess_prod_func: ClassVar[wp.Function] = cast(
        "wp.Function", _stable_neo_hookean_hess_prod_func
    )
    hess_quad_func: ClassVar[wp.Function] = cast(
        "wp.Function", _stable_neo_hookean_hess_quad_func
    )

    energy_density_kernel: ClassVar[wp.Kernel] = WarpElastic.make_energy_density_kernel(
        energy_density_func, __module__
    )
    first_piola_kirchhoff_kernel: ClassVar[wp.Kernel] = (
        WarpElastic.make_first_piola_kirchhoff_kernel(
            first_piola_kirchhoff_func, __module__
        )
    )
    fun_kernel: ClassVar[wp.Kernel] = WarpElastic.make_fun_kernel(
        energy_density_func, __module__
    )
    grad_kernel: ClassVar[wp.Kernel] = WarpElastic.make_grad_kernel(
        first_piola_kirchhoff_func, __module__
    )
    hess_diag_kernel: ClassVar[wp.Kernel] = WarpElastic.make_hess_diag_kernel(
        hess_diag_func, __module__
    )
    hess_prod_kernel: ClassVar[wp.Kernel] = WarpElastic.make_hess_prod_kernel(
        hess_prod_func, __module__
    )
    hess_quad_kernel: ClassVar[wp.Kernel] = WarpElastic.make_hess_quad_kernel(
        hess_quad_func, __module__
    )

    @override
    @classmethod
    def make_materials(cls, region: Region, requires_grad: Sequence[str]) -> Any:
        @wp.struct
        class WarpStableNeoHookeanMaterials:
            fraction: wp.array1d(dtype=wpt.floating)
            lambda_: wp.array1d(dtype=wpt.floating)
            mu: wp.array1d(dtype=wpt.floating)

        materials = WarpStableNeoHookeanMaterials()
        materials.fraction = utils.get_fraction(region)
        materials.lambda_ = utils.get_lambda(region)
        materials.mu = utils.get_mu(region)
        utils.require_grads(materials, requires_grad)
        return materials
