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
def arap_muscle_energy_density_func(
    F: mat33, materials: Materials, cid: int_
) -> float_:
    A = func.make_activation_mat33(materials.activation[cid])  # mat33
    mu = materials.mu[cid]  # float
    G = F @ A  # mat33
    R, _ = math.polar_rv(G)  # mat33, mat33
    return F.dtype(0.5) * mu * math.fro_norm_square(G - R)  # float


@wp.func
@no_type_check
def arap_muscle_first_piola_kirchhoff_func(
    F: mat33, materials: Materials, cid: int_
) -> mat33:
    A = func.make_activation_mat33(materials.activation[cid])  # mat33
    mu = materials.mu[cid]  # float
    G = F @ A  # mat33
    R, _ = math.polar_rv(G)  # mat33, mat33
    return mu * (G - R) @ wp.transpose(A)  # mat33


@wp.func
@no_type_check
def arap_muscle_hess_diag_func(
    F: mat33, dhdX: mat43, materials: Materials, cid: int_, *, clamp: bool = False
) -> mat33:
    A = func.make_activation_mat33(materials.activation[cid])  # mat33
    mu = materials.mu[cid]  # float
    G = F @ A  # mat33
    dhdX_A = dhdX @ A  # mat43
    U, sigma, V = math.svd_rv(G)  # mat33, vec3, mat33
    h4_diag = func.h4_diag(dhdX_A, U, sigma, V, clamp=clamp)  # mat43
    h5_diag = func.h5_diag(dhdX_A)  # mat43
    h_diag = -F.dtype(2.0) * h4_diag + h5_diag  # mat43
    return F.dtype(0.5) * mu * h_diag  # mat33


@wp.func
@no_type_check
def arap_muscle_hess_prod_func(
    F: mat33,
    v: mat43,
    dhdX: mat43,
    materials: Materials,
    cid: int_,
    *,
    clamp: bool = False,
) -> mat43:
    A = func.make_activation_mat33(materials.activation[cid])  # mat33
    mu = materials.mu[cid]  # float
    G = F @ A  # mat33
    dhdX_A = dhdX @ A  # mat43
    U, sigma, V = math.svd_rv(G)  # mat33, vec3, mat33
    h4_prod = func.h4_prod(v, dhdX_A, U, sigma, V, clamp=clamp)  # mat43
    h5_prod = func.h5_prod(v, dhdX_A)  # mat43
    h_prod = -F.dtype(2.0) * h4_prod + h5_prod  # mat43
    return F.dtype(0.5) * mu * h_prod  # mat43


@wp.func
@no_type_check
def arap_muscle_hess_quad_func(
    F: mat33,
    v: mat43,
    dhdX: mat43,
    materials: Materials,
    cid: int_,
    *,
    clamp: bool = False,
) -> float_:
    A = func.make_activation_mat33(materials.activation[cid])  # mat33
    mu = materials.mu[cid]  # float
    G = F @ A  # mat33
    dhdX_A = dhdX @ A  # mat43
    U, sigma, V = math.svd_rv(G)  # mat33, vec3, mat33
    h4_quad = func.h4_quad(v, dhdX_A, U, sigma, V, clamp=clamp)  # float
    h5_quad = func.h5_quad(v, dhdX_A)  # float
    h_quad = -F.dtype(2.0) * h4_quad + h5_quad  # float
    return F.dtype(0.5) * mu * h_quad  # float
