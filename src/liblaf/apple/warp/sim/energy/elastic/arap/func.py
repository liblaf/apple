from typing import no_type_check

import warp as wp

from liblaf.apple.warp import math
from liblaf.apple.warp.sim.energy.elastic import func
from liblaf.apple.warp.typing import float_, mat33, mat43


@wp.struct
class Params:
    mu: wp.array(dtype=float_)


@wp.struct
class ParamsElem:
    mu: float_


@wp.func
@no_type_check
def energy_density(F: mat33, params: ParamsElem):  # -> float:
    R, _ = math.polar_rv(F)  # mat33
    Psi = type(F[0, 0])(0.5) * params.mu * math.frobenius_norm_square(F - R)  # float
    return Psi


@wp.func
@no_type_check
def first_piola_kirchhoff_stress_tensor(F: mat33, params: ParamsElem):  # -> mat33:
    R, _ = math.polar_rv(F)  # mat33
    PK1 = params.mu * (F - R)  # mat33
    return PK1


@wp.func
@no_type_check
def energy_density_hess_diag(F: mat33, dhdX: mat43, params: ParamsElem):  # -> mat43:
    U, s, V = math.svd_rv(F)  # mat33, vec3, mat33
    lambdas = func.lambdas(s)  # vec3
    Q0, Q1, Q2 = func.Qs(U, V)  # mat33, mat33, mat33
    h4_diag = func.h4_diag(dhdX, lambdas, Q0, Q1, Q2)  # mat43
    h5_diag = func.h5_diag(dhdX)  # mat43
    h_diag = -type(F[0, 0])(2.0) * h4_diag + h5_diag  # mat43
    return type(F[0, 0])(0.5) * params.mu * h_diag  # mat43


@wp.func
@no_type_check
def energy_density_hess_prod(
    F: mat33, p: mat43, dhdX: mat43, params: ParamsElem
):  # -> mat43:
    U, s, V = math.svd_rv(F)  # mat33, vec3, mat33
    lambdas = func.lambdas(s)  # vec3
    Q0, Q1, Q2 = func.Qs(U, V)  # mat33, mat33, mat33
    h4_prod = func.h4_prod(p, dhdX, lambdas, Q0, Q1, Q2)  # mat43
    h5_prod = func.h5_prod(p, dhdX)  # mat43
    h_prod = -type(F[0, 0])(2.0) * h4_prod + h5_prod  # mat43
    return type(F[0, 0])(0.5) * params.mu * h_prod  # mat43


@wp.func
@no_type_check
def energy_density_hess_quad(
    F: mat33, p: mat43, dhdX: mat43, params: ParamsElem
):  # -> float:
    U, s, V = math.svd_rv(F)  # mat33, vec3, mat33
    lambdas = func.lambdas(s)  # vec3
    Q0, Q1, Q2 = func.Qs(U, V)  # mat33, mat33, mat33
    h4_quad = func.h4_quad(p, dhdX, lambdas, Q0, Q1, Q2)
    h5_quad = func.h5_quad(p, dhdX)
    h_quad = -type(F[0, 0])(2.0) * h4_quad + h5_quad
    if h_quad < 0.0:
        wp.printf("h4_quad = %f, h5_quad = %f, h_quad = %f\n", h4_quad, h5_quad, h_quad)
    return type(F[0, 0])(0.5) * params.mu * h_quad


@wp.func
@no_type_check
def get_cell_params(params: Params, cid: int) -> ParamsElem:
    p = ParamsElem(mu=params.mu[cid])
    return p
