from collections.abc import Sequence
from typing import Any, ClassVar, cast, no_type_check

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
def _neo_hookean_muscle_energy_density_func(
    F: mat33, materials: Materials, cid: int_
) -> float_:
    A = func.make_activation_mat33(materials.activation[cid])
    fraction = materials.fraction[cid]
    mu = materials.mu[cid]
    lambda_ = materials.lambda_[cid]
    G = F @ A
    J = func.I3(G)
    log_J = wp.log(J)
    return (
        F.dtype(0.5)
        * fraction
        * (
            mu * (func.I2(G) - F.dtype(3.0))
            - F.dtype(2.0) * mu * log_J
            + lambda_ * log_J * log_J
        )
    )


@wp.func
@no_type_check
def _neo_hookean_muscle_first_piola_kirchhoff_func(
    F: mat33, materials: Materials, cid: int_
) -> mat33:
    A = func.make_activation_mat33(materials.activation[cid])
    fraction = materials.fraction[cid]
    mu = materials.mu[cid]
    lambda_ = materials.lambda_[cid]
    G = F @ A
    J = func.I3(G)
    log_J = wp.log(J)
    g3 = func.g3(G)
    return fraction * (mu * G + ((-mu + lambda_ * log_J) / J) * g3) @ wp.transpose(A)


@wp.func
@no_type_check
def _neo_hookean_muscle_hess_diag_func(
    F: mat33,
    dhdX: mat43,
    materials: Materials,
    cid: int_,
    *,
    clamp: bool = False,  # noqa: ARG001
) -> mat33:
    A = func.make_activation_mat33(materials.activation[cid])
    fraction = materials.fraction[cid]
    mu = materials.mu[cid]
    lambda_ = materials.lambda_[cid]
    G = F @ A
    dhdX_A = dhdX @ A
    J = func.I3(G)
    log_J = wp.log(J)
    c1 = (lambda_ * (F.dtype(1.0) - log_J) + mu) / (J * J)
    c2 = (lambda_ * log_J - mu) / J
    h3_diag = func.h3_diag(dhdX_A, func.g3(G))
    h5_diag = func.h5_diag(dhdX_A)
    h6_diag = func.h6_diag(dhdX_A, G)
    return fraction * (c1 * h3_diag + F.dtype(0.5) * mu * h5_diag + c2 * h6_diag)


@wp.func
@no_type_check
def _neo_hookean_muscle_hess_prod_func(
    F: mat33,
    v: mat43,
    dhdX: mat43,
    materials: Materials,
    cid: int_,
    *,
    clamp: bool = False,  # noqa: ARG001
) -> mat43:
    A = func.make_activation_mat33(materials.activation[cid])
    fraction = materials.fraction[cid]
    mu = materials.mu[cid]
    lambda_ = materials.lambda_[cid]
    G = F @ A
    dhdX_A = dhdX @ A
    J = func.I3(G)
    log_J = wp.log(J)
    c1 = (lambda_ * (F.dtype(1.0) - log_J) + mu) / (J * J)
    c2 = (lambda_ * log_J - mu) / J
    h3_prod = func.h3_prod(v, dhdX_A, func.g3(G))
    h5_prod = func.h5_prod(v, dhdX_A)
    h6_prod = func.h6_prod(v, dhdX_A, G)
    return fraction * (c1 * h3_prod + F.dtype(0.5) * mu * h5_prod + c2 * h6_prod)


@wp.func
@no_type_check
def _neo_hookean_muscle_hess_quad_func(
    F: mat33,
    v: mat43,
    dhdX: mat43,
    materials: Materials,
    cid: int_,
    *,
    clamp: bool = False,  # noqa: ARG001
) -> float_:
    A = func.make_activation_mat33(materials.activation[cid])
    fraction = materials.fraction[cid]
    mu = materials.mu[cid]
    lambda_ = materials.lambda_[cid]
    G = F @ A
    dhdX_A = dhdX @ A
    J = func.I3(G)
    log_J = wp.log(J)
    c1 = (lambda_ * (F.dtype(1.0) - log_J) + mu) / (J * J)
    c2 = (lambda_ * log_J - mu) / J
    h3_quad = func.h3_quad(v, dhdX_A, func.g3(G))
    h5_quad = func.h5_quad(v, dhdX_A)
    h6_quad = func.h6_quad(v, dhdX_A, G)
    return fraction * (c1 * h3_quad + F.dtype(0.5) * mu * h5_quad + c2 * h6_quad)


@jarp.frozen_static
class WarpNeoHookeanMuscle(WarpElastic):
    energy_density_func: ClassVar[wp.Function] = cast(
        "wp.Function", _neo_hookean_muscle_energy_density_func
    )
    first_piola_kirchhoff_func: ClassVar[wp.Function] = cast(
        "wp.Function", _neo_hookean_muscle_first_piola_kirchhoff_func
    )
    hess_diag_func: ClassVar[wp.Function] = cast(
        "wp.Function", _neo_hookean_muscle_hess_diag_func
    )
    hess_prod_func: ClassVar[wp.Function] = cast(
        "wp.Function", _neo_hookean_muscle_hess_prod_func
    )
    hess_quad_func: ClassVar[wp.Function] = cast(
        "wp.Function", _neo_hookean_muscle_hess_quad_func
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

    @classmethod
    def make_materials(cls, region: Region, requires_grad: Sequence[str]) -> Any:
        @wp.struct
        class WarpNeoHookeanMuscleMaterials:
            activation: wp.array1d(dtype=wpt.vector(6))
            fraction: wp.array1d(dtype=wpt.floating)
            lambda_: wp.array1d(dtype=wpt.floating)
            mu: wp.array1d(dtype=wpt.floating)

        materials = WarpNeoHookeanMuscleMaterials()
        materials.activation = utils.get_activation(region)
        materials.fraction = utils.get_fraction(region)
        materials.lambda_ = utils.get_lambda(region)
        materials.mu = utils.get_mu(region)
        utils.require_grads(materials, requires_grad)
        return materials
