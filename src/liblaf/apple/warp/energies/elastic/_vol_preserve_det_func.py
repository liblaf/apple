from typing import Any, no_type_check

import warp as wp

from liblaf.apple.warp import math

from . import func

int_ = Any
float_ = Any
mat33 = Any
mat43 = Any
Materials = Any


@wp.func
@no_type_check
def volume_preservation_determinant_energy_density_func(
    F: mat33, materials: Materials, cid: int_
) -> float_:
    lambda_ = materials.lambda_[cid]  # float
    J = func.I3(F)  # float
    return F.dtype(0.5) * lambda_ * math.square(J - F.dtype(1.0))  # float


@wp.func
@no_type_check
def volume_preservation_determinant_first_piola_kirchhoff_func(
    F: mat33, materials: Materials, cid: int_
) -> mat33:
    lambda_ = materials.lambda_[cid]  # float
    J = func.I3(F)  # float
    g3 = func.g3(F)  # mat33
    return lambda_ * (J - F.dtype(1.0)) * g3  # mat33


@wp.func
@no_type_check
def volume_preservation_determinant_hess_diag_func(
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
def volume_preservation_determinant_hess_prod_func(
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
def volume_preservation_determinant_hess_quad_func(
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
