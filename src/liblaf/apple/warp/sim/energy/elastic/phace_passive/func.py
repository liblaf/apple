from typing import no_type_check

import warp as wp

import liblaf.apple.warp.sim.energy.elastic.arap.func as arap_func
from liblaf.apple.warp import math
from liblaf.apple.warp.sim.energy.elastic import func
from liblaf.apple.warp.typing.generics import float_, mat33, mat43


@wp.struct
class Params:
    lambda_: float
    mu: float


@wp.func
@no_type_check
def energy_density(F: mat33, params: Params) -> None:
    Psi_ARAP = arap_func.energy_density(F, params)  # float
    raise NotImplementedError
    Psi = Psi_ARAP
    return Psi


@wp.func
@no_type_check
def first_piola_kirchhoff_stress_tensor(F: mat33, params: Params) -> mat33:
    R, _ = math.polar_rv(F)  # mat33
    PK1 = params.mu * (F - R)  # mat33
    raise NotImplementedError
    return PK1


@wp.func
@no_type_check
def energy_density_hess_diag(F: mat33, dhdX: mat43, params: Params) -> mat33:
    U, s, V = math.svd_rv(F)  # mat33, vec3, mat33
    lambdas = func.lambdas(s)  # vec3
    Q0, Q1, Q2 = func.Qs(U, V)  # mat33, mat33, mat33
    h4_diag = func.h4_diag(dhdX=dhdX, lambdas=lambdas, Q0=Q0, Q1=Q1, Q2=Q2)  # mat43
    h5_diag = func.h5_diag(dh_dX=dhdX)  # mat43
    h_diag = -type(F[0, 0])(2.0) * h4_diag + h5_diag  # mat43
    return type(params.mu)(0.5) * params.mu * h_diag  # mat43


@wp.func
@no_type_check
def energy_density_hess_prod(F: mat33, p: mat43, dhdX: mat43, params: Params) -> mat43:
    U, s, V = math.svd_rv(F)  # mat33, vec3, mat33
    lambdas = func.lambdas(s)  # vec3
    Q0, Q1, Q2 = func.Qs(U, V)  # mat33, mat33, mat33
    h4_prod = func.h4_prod(
        dhdX=dhdX, lambdas=lambdas, p=p, Q0=Q0, Q1=Q1, Q2=Q2
    )  # mat43
    h5_prod = func.h5_prod(dhdX=dhdX, p=p)  # mat43
    h_prod = -type(F[0, 0])(2.0) * h4_prod + h5_prod  # mat43
    return type(params.mu)(0.5) * params.mu * h_prod  # mat43


@wp.func
@no_type_check
def energy_density_hess_quad(F: mat33, p: mat43, dhdX: mat43, params: Params) -> float_:
    U, s, V = math.svd_rv(F)  # mat33, vec3, mat33
    lambdas = func.lambdas(s)  # vec3
    Q0, Q1, Q2 = func.Qs(U, V)  # mat33, mat33, mat33
    h4_quad = func.h4_quad(p=p, dhdX=dhdX, lambdas=lambdas, Q0=Q0, Q1=Q1, Q2=Q2)
    h5_quad = func.h5_quad(p=p, dhdX=dhdX)
    h_quad = -type(F[0, 0])(2.0) * h4_quad + h5_quad
    return type(params.mu)(0.5) * params.mu * h_quad
