from typing import no_type_check

import warp as wp

from liblaf.apple.func import strain, utils
from liblaf.apple.typed.warp import mat33, mat43

from .arap import (
    arap_energy_density,
    arap_energy_density_hess_diag,
    arap_energy_density_hess_quad,
    arap_first_piola_kirchhoff_stress,
)


@wp.struct
class PhaceStaticParams:
    lambda_: float  # Lame's first parameter
    mu: float  # Lame's second parameter


@no_type_check
@wp.func
def phace_static_energy_density(F: mat33, params: PhaceStaticParams) -> float:
    mu = params.mu  # float
    lambda_ = params.lambda_  # float
    J = wp.determinant(F)  # scalar
    Psi_ARAP = arap_energy_density(F=F, mu=mu)  # scalar
    Psi_VP = lambda_ * utils.square(J - 1.0)  # scalar
    Psi = 2.0 * Psi_ARAP + Psi_VP  # scalar
    return Psi  # scalar


@no_type_check
@wp.func
def phace_static_first_piola_kirchhoff_stress(
    F: mat33, params: PhaceStaticParams
) -> mat33:
    mu = params.mu  # float
    lambda_ = params.lambda_  # float
    J = wp.determinant(F)  # scalar
    g3 = strain.g3(F)  # mat33
    PK1_ARAP = arap_first_piola_kirchhoff_stress(F=F, mu=mu)  # mat33
    PK1_VP = 2.0 * lambda_ * (J - 1.0) * g3  # mat33
    PK1 = 2.0 * PK1_ARAP + PK1_VP  # mat33
    return PK1  # mat33


@no_type_check
@wp.func
def phace_static_energy_density_hess_diag(
    F: mat33, params: PhaceStaticParams, dh_dX: mat43
) -> mat43:
    mu = params.mu  # float
    lambda_ = params.lambda_  # float
    g3 = strain.g3(F)  # mat33
    d2Psi_dI32 = 2.0 * lambda_  # scalar
    hess_diag_ARAP = arap_energy_density_hess_diag(F=F, mu=mu, dh_dX=dh_dX)  # mat43
    h3_diag = strain.h3_diag(dh_dX=dh_dX, g3=g3)  # mat43
    # h6_diag = 0
    hess_diag_VP = d2Psi_dI32 * h3_diag  # mat43
    return 2.0 * hess_diag_ARAP + hess_diag_VP  # mat43


@no_type_check
@wp.func
def phace_static_energy_density_hess_quad(
    F: mat33, p: mat43, params: PhaceStaticParams, dh_dX: mat43
) -> float:
    mu = params.mu  # float
    lambda_ = params.lambda_  # float
    J = wp.determinant(F)  # scalar
    g3 = strain.g3(F)  # mat33
    dPsi_dI3 = 2.0 * lambda_ * (J - 1.0)  # scalar
    d2Psi_dI32 = 2.0 * lambda_  # scalar
    hess_quad_ARAP = arap_energy_density_hess_quad(
        F=F, p=p, mu=mu, dh_dX=dh_dX
    )  # scalar
    h3_quad = strain.h3_quad(p=p, dh_dX=dh_dX, g3=g3)  # scalar
    h6_quad = strain.h6_quad(p=p, F=F, dh_dX=dh_dX)
    hess_quad_VP = d2Psi_dI32 * h3_quad + dPsi_dI3 * h6_quad
    return 2.0 * hess_quad_ARAP + hess_quad_VP  # scalar
