from collections.abc import Sequence
from typing import Any, ClassVar, cast, no_type_check, override

import jarp
import jarp.warp.types as wpt
import warp as wp

from liblaf.apple.consts import LAMBDA
from liblaf.apple.jax import Region
from liblaf.apple.warp import math

from . import func
from ._base import WarpElastic

int_ = Any
float_ = Any
mat33 = Any
mat43 = Any
Materials = Any


@wp.func
@no_type_check
def _volume_preservation_determinant_energy_density_func(
    F: mat33, materials: Materials, cid: int_
) -> float_:
    lambda_ = materials.lambda_[cid]  # float
    J = func.I3(F)  # float
    return F.dtype(0.5) * lambda_ * math.square(J - F.dtype(1.0))  # float


@wp.func
@no_type_check
def _volume_preservation_determinant_first_piola_kirchhoff_func(
    F: mat33, materials: Materials, cid: int_
) -> mat33:
    lambda_ = materials.lambda_[cid]  # float
    J = func.I3(F)  # float
    g3 = func.g3(F)  # mat33
    return lambda_ * (J - F.dtype(1.0)) * g3  # mat33


@wp.func
@no_type_check
def _volume_preservation_determinant_hess_diag_func(
    F: mat33,
    dhdX: mat43,
    materials: Materials,
    cid: int_,
    *,
    clamp: bool = False,  # noqa: ARG001
) -> mat33:
    lambda_ = materials.lambda_[cid]  # float
    J = func.I3(F)  # float
    g3 = func.g3(F)  # mat33
    h3_diag = func.h3_diag(dhdX, g3)  # mat33
    h6_diag = func.h6_diag(dhdX, F)  # mat33
    dPsi_dI3 = lambda_ * (J - F.dtype(1.0))  # float
    dPsi_dI3_2 = lambda_  # float
    return dPsi_dI3_2 * h3_diag + dPsi_dI3 * h6_diag  # mat33


@wp.func
@no_type_check
def _volume_preservation_determinant_hess_prod_func(
    F: mat33,
    v: mat43,
    dhdX: mat43,
    materials: Materials,
    cid: int_,
    *,
    clamp: bool = False,  # noqa: ARG001
) -> mat43:
    lambda_ = materials.lambda_[cid]  # float
    J = func.I3(F)  # float
    g3 = func.g3(F)  # mat33
    h3_prod = func.h3_prod(v, dhdX, g3)  # mat43
    h6_prod = func.h6_prod(v, dhdX, F)  # mat43
    dPsi_dI3 = lambda_ * (J - F.dtype(1.0))  # float
    dPsi_dI3_2 = lambda_  # float
    return dPsi_dI3_2 * h3_prod + dPsi_dI3 * h6_prod  # mat43


@wp.func
@no_type_check
def _volume_preservation_determinant_hess_quad_func(
    F: mat33,
    v: mat43,
    dhdX: mat43,
    materials: Materials,
    cid: int_,
    *,
    clamp: bool = False,  # noqa: ARG001
) -> float_:
    lambda_ = materials.lambda_[cid]  # float
    J = func.I3(F)  # float
    g3 = func.g3(F)  # mat33
    h3_quad = func.h3_quad(v, dhdX, g3)  # float
    h6_quad = func.h6_quad(v, dhdX, F)  # float
    dPsi_dI3 = lambda_ * (J - F.dtype(1.0))  # float
    dPsi_dI3_2 = lambda_  # float
    return dPsi_dI3_2 * h3_quad + dPsi_dI3 * h6_quad  # float


class WarpVolumePreservationDeterminant(WarpElastic):
    energy_density_func: ClassVar[wp.Function] = cast(
        "wp.Function", _volume_preservation_determinant_energy_density_func
    )
    first_piola_kirchhoff_func: ClassVar[wp.Function] = cast(
        "wp.Function", _volume_preservation_determinant_first_piola_kirchhoff_func
    )
    hess_diag_func: ClassVar[wp.Function] = cast(
        "wp.Function", _volume_preservation_determinant_hess_diag_func
    )
    hess_prod_func: ClassVar[wp.Function] = cast(
        "wp.Function", _volume_preservation_determinant_hess_prod_func
    )
    hess_quad_func: ClassVar[wp.Function] = cast(
        "wp.Function", _volume_preservation_determinant_hess_quad_func
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
        class WarpVolumePreservationDeterminantMaterials:
            lambda_: wp.array1d(dtype=wpt.float_)

        lambda_ = jarp.to_warp(
            region.cell_data[LAMBDA],
            wpt.float_,
            requires_grad=(LAMBDA in requires_grad),
        )
        materials = WarpVolumePreservationDeterminantMaterials()
        materials.lambda_ = lambda_
        return materials
