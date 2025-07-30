import warp as wp

from liblaf.apple.energy.elastic.arap import func as _arap
from liblaf.apple.func import strain, utils
from liblaf.apple.typed.warp import mat33, mat43


@wp.struct
class PhacePassiveParams:
    lambda_: float  # Lame's first parameter
    mu: float  # Lame's second parameter


# @no_type_check
@wp.func
def phace_passive_energy_density_func(F: mat33, params: PhacePassiveParams) -> float:
    mu = params.mu  # float
    lambda_ = params.lambda_  # float
    J = wp.determinant(F)  # scalar
    Psi_ARAP = _arap.arap_energy_density_func(F=F, mu=mu)  # scalar
    Psi_VP = lambda_ * utils.square(J - 1.0)  # scalar
    Psi = 2.0 * Psi_ARAP + Psi_VP  # scalar
    return Psi  # scalar


@wp.func
def phace_passive_first_piola_kirchhoff_stress_func(
    F: mat33, params: PhacePassiveParams
) -> mat33:
    mu = params.mu  # float
    lambda_ = params.lambda_  # float
    J = wp.determinant(F)  # scalar
    g3 = strain.g3(F)  # mat33
    PK1_ARAP = _arap.arap_first_piola_kirchhoff_stress_func(F=F, mu=mu)  # mat33
    PK1_VP = 2.0 * lambda_ * (J - 1.0) * g3  # mat33
    PK1 = 2.0 * PK1_ARAP + PK1_VP  # mat33
    return PK1  # mat33


@wp.func
def phace_passive_energy_density_hess_diag_func(
    F: mat33, dh_dX: mat43, params: PhacePassiveParams
) -> mat43:
    mu = params.mu  # float
    lambda_ = params.lambda_  # float
    g3 = strain.g3(F)  # mat33
    d2Psi_dI32 = 2.0 * lambda_  # scalar
    hess_diag_ARAP = _arap.arap_energy_density_hess_diag_func(
        F=F, mu=mu, dh_dX=dh_dX
    )  # mat43
    h3_diag = strain.h3_diag(dh_dX=dh_dX, g3=g3)  # mat43
    # h6_diag = 0
    hess_diag_VP = d2Psi_dI32 * h3_diag  # mat43
    return 2.0 * hess_diag_ARAP + hess_diag_VP  # pyright: ignore[reportReturnType]


@wp.func
def phace_passive_energy_density_hess_quad_func(
    F: mat33, p: mat43, dh_dX: mat43, params: PhacePassiveParams
) -> float:
    mu = params.mu  # float
    lambda_ = params.lambda_  # float
    J = wp.determinant(F)  # scalar
    g3 = strain.g3(F)  # mat33
    dPsi_dI3 = 2.0 * lambda_ * (J - 1.0)  # scalar
    d2Psi_dI32 = 2.0 * lambda_  # scalar
    hess_quad_ARAP = _arap.arap_energy_density_hess_quad_func(
        F=F, p=p, mu=mu, dh_dX=dh_dX
    )  # scalar
    h3_quad = strain.h3_quad(p=p, dh_dX=dh_dX, g3=g3)  # scalar
    h6_quad = strain.h6_quad(p=p, F=F, dh_dX=dh_dX)
    hess_quad_VP = d2Psi_dI32 * h3_quad + dPsi_dI3 * h6_quad
    return 2.0 * hess_quad_ARAP + hess_quad_VP  # scalar
