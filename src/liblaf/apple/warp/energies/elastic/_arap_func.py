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
def arap_energy_density_func(F: mat33, materials: Materials, cid: int_) -> float_:
    mu = materials.mu[cid]  # float
    R, _ = math.polar_rv(F)  # mat33, mat33
    return F.dtype(0.5) * mu * math.fro_norm_square(F - R)  # float


@wp.func
@no_type_check
def arap_first_piola_kirchhoff_func(F: mat33, materials: Materials, cid: int_) -> mat33:
    mu = materials.mu[cid]  # float
    R, _ = math.polar_rv(F)  # mat33, mat33
    return mu * (F - R)  # mat33


@wp.func
@no_type_check
def arap_hess_diag_func(
    F: mat33, dhdX: mat43, materials: Materials, cid: int_, *, clamp: bool = False
) -> mat43:
    mu = materials.mu[cid]  # float
    U, sigma, V = math.svd_rv(F)  # mat33, vec3, mat33
    h4_diag = func.h4_diag(dhdX, U, sigma, V, clamp=clamp)  # mat43
    h5_diag = func.h5_diag(dhdX)  # mat43
    h_diag = -F.dtype(2.0) * h4_diag + h5_diag  # mat43
    return F.dtype(0.5) * mu * h_diag  # mat43


@wp.func
@no_type_check
def arap_hess_prod_func(
    F: mat33,
    v: mat43,
    dhdX: mat43,
    materials: Materials,
    cid: int_,
    *,
    clamp: bool = False,
) -> mat43:
    mu = materials.mu[cid]  # float
    U, sigma, V = math.svd_rv(F)  # mat33, vec3, mat33
    h4_prod = func.h4_prod(v, dhdX, U, sigma, V, clamp=clamp)  # mat43
    h5_prod = func.h5_prod(v, dhdX)  # mat43
    h_prod = -F.dtype(2.0) * h4_prod + h5_prod  # mat43
    return F.dtype(0.5) * mu * h_prod  # mat43


@wp.func
@no_type_check
def arap_hess_quad_func(
    F: mat33,
    v: mat43,
    dhdX: mat43,
    materials: Materials,
    cid: int_,
    *,
    clamp: bool = False,
) -> float_:
    mu = materials.mu[cid]  # float
    U, sigma, V = math.svd_rv(F)  # mat33, vec3, mat33
    h4_quad = func.h4_quad(v, dhdX, U, sigma, V, clamp=clamp)  # float
    h5_quad = func.h5_quad(v, dhdX)  # float
    h_quad = -F.dtype(2.0) * h4_quad + h5_quad  # float
    return F.dtype(0.5) * mu * h_quad  # float
