from typing import no_type_check

import warp as wp

from liblaf.apple.warp import math
from liblaf.apple.warp.sim.energy.elastic import utils
from liblaf.apple.warp.typing import float_, mat33, mat43


@wp.struct
class Params:
    mu: float_


@wp.func
@no_type_check
def energy_density(F: mat33, params: Params) -> None:
    R, _ = math.polar_rv(F)  # mat33
    Psi = type(params.mu)(0.5) * params.mu * math.frobenius_norm_square(F - R)  # float
    return Psi


@wp.func
@no_type_check
def first_piola_kirchhoff_stress_tensor(F: mat33, params: Params) -> mat33:
    R, _ = math.polar_rv(F)  # mat33
    PK1 = params.mu * (F - R)  # mat33
    return PK1


@wp.func
@no_type_check
def energy_density_hess_diag(F: mat33, dhdX: mat43, params: Params) -> mat33:
    U, s, V = math.svd_rv(F)  # mat33, vec3, mat33
    lambdas = utils.lambdas(s)  # vec3
    Q0, Q1, Q2 = utils.Qs(U, V)  # mat33, mat33, mat33
    h4_diag = utils.h4_diag(dhdX=dhdX, lambdas=lambdas, Q0=Q0, Q1=Q1, Q2=Q2)  # mat43
    h5_diag = utils.h5_diag(dh_dX=dhdX)  # mat43
    h_diag = -type(F[0, 0])(2.0) * h4_diag + h5_diag  # mat43
    return type(params.mu)(0.5) * params.mu * h_diag  # mat43


@wp.func
@no_type_check
def energy_density_hess_prod(F: mat33, p: mat43, dhdX: mat43, params: Params) -> mat43:
    U, s, V = math.svd_rv(F)  # mat33, vec3, mat33
    lambdas = utils.lambdas(s)  # vec3
    Q0, Q1, Q2 = utils.Qs(U, V)  # mat33, mat33, mat33
    h4_prod = utils.h4_prod(
        dhdX=dhdX, lambdas=lambdas, p=p, Q0=Q0, Q1=Q1, Q2=Q2
    )  # mat43
    h5_prod = utils.h5_prod(dhdX=dhdX, p=p)  # mat43
    h_prod = -type(F[0, 0])(2.0) * h4_prod + h5_prod  # mat43
    return type(params.mu)(0.5) * params.mu * h_prod  # mat43


@wp.func
@no_type_check
def energy_density_hess_quad(F: mat33, p: mat43, dhdX: mat43, params: Params) -> float_:
    U, s, V = math.svd_rv(F)  # mat33, vec3, mat33
    lambdas = utils.lambdas(s)  # vec3
    Q0, Q1, Q2 = utils.Qs(U, V)  # mat33, mat33, mat33
    h4_quad = utils.h4_quad(p=p, dhdX=dhdX, lambdas=lambdas, Q0=Q0, Q1=Q1, Q2=Q2)
    h5_quad = utils.h5_quad(p=p, dhdX=dhdX)
    h_quad = -type(F[0, 0])(2.0) * h4_quad + h5_quad
    return type(params.mu)(0.5) * params.mu * h_quad
