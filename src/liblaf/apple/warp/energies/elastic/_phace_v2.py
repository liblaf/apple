import functools
from collections.abc import Sequence
from typing import Any, ClassVar, cast, no_type_check, override

import jarp
import jarp.warp.types as wpt
import warp as wp

from liblaf.apple.consts import ACTIVATION, LAMBDA, MU, MUSCLE_FRACTION
from liblaf.apple.jax import Region
from liblaf.apple.warp import math
from liblaf.apple.warp.energies.elastic import func

from ._arap_func import (
    arap_energy_density_func,
    arap_first_piola_kirchhoff_func,
    arap_hess_prod_func,
)
from ._arap_muscle_func import (
    arap_muscle_energy_density_func,
    arap_muscle_first_piola_kirchhoff_func,
    arap_muscle_hess_prod_func,
)
from ._base import WarpElastic
from ._vol_preserve_det_func import (
    volume_preservation_determinant_energy_density_func,
    volume_preservation_determinant_first_piola_kirchhoff_func,
    volume_preservation_determinant_hess_diag_func,
    volume_preservation_determinant_hess_prod_func,
)

int_ = Any
float_ = Any
mat33 = Any
mat43 = Any
Materials = Any


@wp.func
@no_type_check
def _phace_v2_energy_density_func(F: mat33, materials: Materials, cid: int_) -> float_:
    muscle_fraction = materials.muscle_fraction[cid]  # float
    Psi_arap = arap_energy_density_func(F, materials, cid)  # float
    Psi_arap_muscle = arap_muscle_energy_density_func(F, materials, cid)  # float
    Psi_vol = volume_preservation_determinant_energy_density_func(
        F, materials, cid
    )  # float
    # return Psi_arap + Psi_vol
    return (
        (F.dtype(1.0) - muscle_fraction) * Psi_arap
        + muscle_fraction * Psi_arap_muscle
        + Psi_vol
    )  # float


@wp.func
@no_type_check
def _phace_v2_first_piola_kirchhoff_func(
    F: mat33, materials: Materials, cid: int_
) -> mat33:
    # return wp.matrix(shape=(3, 3), dtype=F.dtype)
    # if cid < 3:
    #     wp.printf("PK1: %d\n", cid)
    muscle_fraction = materials.muscle_fraction[cid]  # float
    P_arap = arap_first_piola_kirchhoff_func(F, materials, cid)  # mat33
    # mu = materials.mu[cid]  # float
    # R, _ = math.polar_rv(F)  # mat33, mat33
    # P_arap = mu * (F - R)  # mat33
    P_arap_muscle = arap_muscle_first_piola_kirchhoff_func(F, materials, cid)  # mat33
    # A = func.make_activation_mat33(materials.activation[cid])  # mat33
    # mu = materials.mu[cid]  # float
    # G = F @ A  # mat33
    # R, _ = math.polar_rv(G)  # mat33, mat33
    # P_arap_muscle = mu * (G - R) @ wp.transpose(A)  # mat33
    P_vol = volume_preservation_determinant_first_piola_kirchhoff_func(
        F, materials, cid
    )  # mat33
    # lambda_ = materials.lambda_[cid]  # float
    # J = func.I3(F)  # float
    # g3 = func.g3(F)  # mat33
    # P_vol = lambda_ * (J - F.dtype(1.0)) * g3  # mat33
    # wp.printf("muscle_fraction: %f\n", muscle_fraction)
    return (
        (F.dtype(1.0) - muscle_fraction) * P_arap
        + muscle_fraction * P_arap_muscle
        + P_vol
    )  # mat33


@wp.func
@no_type_check
def hess_diag_func(
    F: mat33, dhdX: mat43, materials: Materials, cid: int_, *, clamp: bool = False
) -> mat43:
    # return wp.matrix(shape=(4, 3), dtype=F.dtype)
    # if cid < 3:
    wp.printf("hess_diag: %d\n", cid)
    muscle_fraction = materials.muscle_fraction[cid]  # float
    # H_diag_arap = arap_hess_diag_func(F, dhdX, materials, cid, clamp=clamp)  # mat43
    mu = materials.mu[cid]  # float
    U, sigma, V = wp.svd3(F)  # mat33, vec3, mat33
    h4_diag = func.h4_diag(dhdX, U, sigma, V, clamp=clamp)  # mat43
    h5_diag = func.h5_diag(dhdX)  # mat43
    h_diag = -F.dtype(2.0) * h4_diag + h5_diag  # mat43
    H_diag_arap = F.dtype(0.5) * mu * h_diag  # mat43
    # H_diag_arap = wp.matrix(shape=(4, 3), dtype=F.dtype)
    # H_diag_arap_muscle = arap_muscle_hess_diag_func(
    #     F, dhdX, materials, cid, clamp=clamp
    # )  # mat43
    wp.printf("hess_diag H_diag_arap: %d\n", cid)
    A = func.make_activation_mat33(materials.activation[cid])  # mat33
    # A = wp.identity(3, dtype=F.dtype)
    # mu = materials.mu[cid]  # float
    # G = F @ A  # mat33
    G = F
    dhdX_A = dhdX @ A  # mat43
    U, sigma, V = wp.svd3(G)  # mat33, vec3, mat33
    h4_diag = func.h4_diag(dhdX_A, U, sigma, V)  # mat43 # !!!
    # h4_diag = wp.matrix(shape=(4, 3), dtype=F.dtype)
    h5_diag = func.h5_diag(dhdX_A)  # mat43 # !!!
    h_diag = -F.dtype(2.0) * h4_diag + h5_diag  # mat43
    # return h_diag
    H_diag_arap_muscle = F.dtype(0.5) * mu * h_diag  # mat33
    # H_diag_arap_muscle = wp.matrix(shape=(4, 3), dtype=F.dtype)
    # H_diag_arap_muscle = wp.matrix(shape=(4, 3), dtype=F.dtype)
    H_diag_vol = volume_preservation_determinant_hess_diag_func(
        F, dhdX, materials, cid, clamp=clamp
    )  # mat43
    # lambda_ = materials.lambda_[cid]  # float
    # J = func.I3(F)  # float
    # g3 = func.g3(F)  # mat33
    # h3_diag = func.h3_diag(dhdX, g3)  # mat33
    # h6_diag = func.h6_diag(dhdX, F)  # mat33
    # dPsi_dI3 = lambda_ * (J - F.dtype(1.0))  # float
    # dPsi_dI3_2 = lambda_  # float
    # H_diag_vol = dPsi_dI3_2 * h3_diag + dPsi_dI3 * h6_diag  # mat33
    return (
        (F.dtype(1.0) - muscle_fraction) * H_diag_arap
        + muscle_fraction * H_diag_arap_muscle
        + H_diag_vol
    )  # mat43


@wp.func
@no_type_check
def _phace_v2_hess_prod_func(
    F: mat33,
    v: mat43,
    dhdX: mat43,
    materials: Materials,
    cid: int_,
    *,
    clamp: bool = False,
) -> mat43:
    muscle_fraction = materials.muscle_fraction[cid]  # float
    H_prod_arap = arap_hess_prod_func(F, v, dhdX, materials, cid, clamp=clamp)  # mat43
    H_prod_arap_muscle = arap_muscle_hess_prod_func(
        F, v, dhdX, materials, cid, clamp=clamp
    )  # mat43
    H_prod_vol = volume_preservation_determinant_hess_prod_func(
        F, v, dhdX, materials, cid, clamp=clamp
    )  # mat43
    return (
        (F.dtype(1.0) - muscle_fraction) * H_prod_arap
        + muscle_fraction * H_prod_arap_muscle
        + H_prod_vol
    )  # mat43


@wp.func
@no_type_check
def _phace_v2_hess_quad_func(
    F: mat33,
    v: mat43,
    dhdX: mat43,
    materials: Materials,
    cid: int_,
    *,
    clamp: bool = False,
) -> float_:
    # return F.dtype(0.0)
    muscle_fraction = materials.muscle_fraction[cid]  # float
    # H_quad_arap = arap_hess_quad_func(F, v, dhdX, materials, cid, clamp=clamp)  # float
    mu = materials.mu[cid]  # float
    U, sigma, V = math.svd_rv(F)  # mat33, vec3, mat33
    h4_quad = func.h4_quad(v, dhdX, U, sigma, V, clamp=clamp)  # float
    h5_quad = func.h5_quad(v, dhdX)  # float
    h_quad = -F.dtype(2.0) * h4_quad + h5_quad  # float
    H_quad_arap = F.dtype(0.5) * mu * h_quad  # float

    # H_quad_arap_muscle = arap_muscle_hess_quad_func(
    #     F, v, dhdX, materials, cid, clamp=clamp
    # )  # float
    A = func.make_activation_mat33(materials.activation[cid])  # mat33
    mu = materials.mu[cid]  # float
    G = F @ A  # mat33
    dhdX_A = dhdX @ A  # mat43
    U, sigma, V = math.svd_rv(G)  # mat33, vec3, mat33
    h4_quad = func.h4_quad(v, dhdX_A, U, sigma, V, clamp=clamp)  # float
    h5_quad = func.h5_quad(v, dhdX_A)  # float
    h_quad = -F.dtype(2.0) * h4_quad + h5_quad  # float
    H_quad_arap_muscle = F.dtype(0.5) * mu * h_quad  # float

    # H_quad_vol = volume_preservation_determinant_hess_quad_func(
    #     F, v, dhdX, materials, cid, clamp=clamp
    # )  # float
    lambda_ = materials.lambda_[cid]  # float
    J = func.I3(F)  # float
    g3 = func.g3(F)  # mat33
    h3_quad = func.h3_quad(v, dhdX, g3)  # float
    h6_quad = func.h6_quad(v, dhdX, F)  # float
    dPsi_dI3 = lambda_ * (J - F.dtype(1.0))  # float
    dPsi_dI3_2 = lambda_  # float
    H_quad_vol = dPsi_dI3_2 * h3_quad + dPsi_dI3 * h6_quad  # float

    return (
        (F.dtype(1.0) - muscle_fraction) * H_quad_arap
        + muscle_fraction * H_quad_arap_muscle
        + H_quad_vol
    )  # float


@wp.struct
class WarpPhaceV2Materials:
    activation: wp.array1d(dtype=wpt.vector(6))
    lambda_: wp.array1d(dtype=wpt.float_)
    mu: wp.array1d(dtype=wpt.float_)
    muscle_fraction: wp.array1d(dtype=wpt.float_)


@jarp.frozen_static
class WarpPhaceV2(WarpElastic):
    energy_density_func: ClassVar[wp.Function] = cast(
        "wp.Function", _phace_v2_energy_density_func
    )
    first_piola_kirchhoff_func: ClassVar[wp.Function] = cast(
        "wp.Function", _phace_v2_first_piola_kirchhoff_func
    )
    hess_diag_func: ClassVar[wp.Function] = cast("wp.Function", hess_diag_func)
    hess_prod_func: ClassVar[wp.Function] = cast(
        "wp.Function", _phace_v2_hess_prod_func
    )
    hess_quad_func: ClassVar[wp.Function] = cast(
        "wp.Function", _phace_v2_hess_quad_func
    )

    # energy_density_kernel: ClassVar[wp.Kernel] = WarpElastic.make_energy_density_kernel(
    #     energy_density_func, None
    # )
    # first_piola_kirchhoff_kernel: ClassVar[wp.Kernel] = (
    #     WarpElastic.make_first_piola_kirchhoff_kernel(first_piola_kirchhoff_func, None)
    # )
    fun_kernel: ClassVar[wp.Kernel] = wp.overload(
        WarpElastic.make_fun_kernel(energy_density_func, None),
        {
            "u": wp.array1d(dtype=wpt.vec3),
            "cells": wp.array1d(dtype=wp.vec4i),
            "dhdX": wp.array2d(dtype=wpt.matrix((4, 3))),
            "dV": wp.array2d(dtype=wpt.float_),
            "materials": WarpPhaceV2Materials,
            "output": wp.array1d(dtype=wpt.float_),
        },
    )
    grad_kernel: ClassVar[wp.Kernel] = wp.overload(
        WarpElastic.make_grad_kernel(first_piola_kirchhoff_func, None),
        {
            "u": wp.array1d(dtype=wpt.vec3),
            "cells": wp.array1d(dtype=wp.vec4i),
            "dhdX": wp.array2d(dtype=wpt.matrix((4, 3))),
            "dV": wp.array2d(dtype=wpt.float_),
            "materials": WarpPhaceV2Materials,
            "output": wp.array1d(dtype=wpt.vec3),
        },
    )
    hess_diag_kernel: ClassVar[wp.Kernel] = wp.overload(
        WarpElastic.make_hess_diag_kernel(hess_diag_func, None),
        {
            "u": wp.array1d(dtype=wpt.vec3),
            "cells": wp.array1d(dtype=wp.vec4i),
            "dhdX": wp.array2d(dtype=wpt.matrix((4, 3))),
            "dV": wp.array2d(dtype=wpt.float_),
            "materials": WarpPhaceV2Materials,
            "output": wp.array1d(dtype=wpt.vec3),
        },
    )
    # hess_prod_kernel: ClassVar[wp.Kernel] = WarpElastic.make_hess_prod_kernel(
    #     hess_prod_func, None
    # )
    hess_quad_kernel: ClassVar[wp.Kernel] = wp.overload(
        WarpElastic.make_hess_quad_kernel(hess_quad_func, None),
        {
            "u": wp.array1d(dtype=wpt.vec3),
            "v": wp.array1d(dtype=wpt.vec3),
            "cells": wp.array1d(dtype=wp.vec4i),
            "dhdX": wp.array2d(dtype=wpt.matrix((4, 3))),
            "dV": wp.array2d(dtype=wpt.float_),
            "materials": WarpPhaceV2Materials,
            "output": wp.array1d(dtype=wpt.float_),
        },
    )

    # def __attrs_post_init__(self) -> None:
    #     print("Overloading PhaceV2 kernels")
    #     wp.overload(
    #         self.grad_kernel,
    #         {
    #             "u": wp.array1d(dtype=wpt.vec3),
    #             "cells": wp.array1d(dtype=wp.vec4i),
    #             "dhdX": wp.array2d(dtype=wpt.matrix((4, 3))),
    #             "dV": wp.array2d(dtype=wpt.float_),
    #             "materials": self.materials_struct(),
    #             "output": wp.array1d(dtype=wpt.vec3),
    #         },
    #     )
    #     wp.overload(
    #         self.hess_diag_kernel,
    #         {
    #             "u": wp.array1d(dtype=wpt.vec3),
    #             "cells": wp.array1d(dtype=wp.vec4i),
    #             "dhdX": wp.array2d(dtype=wpt.matrix((4, 3))),
    #             "dV": wp.array2d(dtype=wpt.float_),
    #             "materials": self.materials_struct(),
    #             "output": wp.array1d(dtype=wpt.vec3),
    #         },
    #     )

    @override
    @classmethod
    @functools.lru_cache
    def materials_struct(cls) -> Any:  # pyright: ignore[reportIncompatibleMethodOverride]
        @wp.struct
        class WarpPhaceV2Materials:
            activation: wp.array1d(dtype=wpt.vector(6))
            lambda_: wp.array1d(dtype=wpt.float_)
            mu: wp.array1d(dtype=wpt.float_)
            muscle_fraction: wp.array1d(dtype=wpt.float_)

        return WarpPhaceV2Materials

    @override
    @classmethod
    def make_materials(cls, region: Region, requires_grad: Sequence[str]) -> Any:
        # WarpPhaceV2Materials = WarpPhaceV2Materials
        activation: wp.array = jarp.to_warp(
            region.cell_data[ACTIVATION],
            wpt.vector(6),
            requires_grad=(ACTIVATION in requires_grad),
        )
        lambda_: wp.array = jarp.to_warp(
            region.cell_data[LAMBDA],
            wpt.float_,
            requires_grad=(LAMBDA in requires_grad),
        )
        mu: wp.array = jarp.to_warp(
            region.cell_data[MU], wpt.float_, requires_grad=(MU in requires_grad)
        )
        muscle_fraction: wp.array = jarp.to_warp(
            region.cell_data[MUSCLE_FRACTION],
            wpt.float_,
            requires_grad=(MUSCLE_FRACTION in requires_grad),
        )
        materials = WarpPhaceV2Materials()
        materials.activation = activation
        materials.lambda_ = lambda_
        materials.mu = mu
        materials.muscle_fraction = muscle_fraction
        return materials
