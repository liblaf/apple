r"""As-Rigid-As-Possible.

$$
\Psi = \frac{\mu}{2} \|F - R\|_F^2 = \frac{\mu}{2} (I_2 - 2 I_1 + 3)
$$
"""

from typing import no_type_check

import warp as wp

from liblaf.apple.energy.elastic.arap import func as _arap_passive
from liblaf.apple.func import strain, utils
from liblaf.apple.typed.warp import mat33, mat43


@wp.struct
class ArapActiveParams:
    activation: mat33
    mu: float
    muscle_fraction: float


@no_type_check
@wp.func
def arap_active_energy_density_func(F: mat33, params: ArapActiveParams) -> float:
    A = params.activation
    mu = params.mu
    muscle_fraction = params.muscle_fraction
    R, S = utils.polar_rv(F)
    Psi_active = 0.5 * mu * utils.frobenius_norm_square(F - R @ A)
    Psi_passive = _arap_passive.arap_energy_density_func(F=F, mu=mu)
    Psi = muscle_fraction * Psi_active + (1.0 - muscle_fraction) * Psi_passive
    return Psi


@no_type_check
@wp.func
def arap_active_first_piola_kirchhoff_stress_func(
    F: mat33, params: ArapActiveParams
) -> mat33:
    A = params.activation
    mu = params.mu
    muscle_fraction = params.muscle_fraction
    R, S = utils.polar_rv(F)
    PK1_active = mu * (F - R @ A)
    PK1_passive = _arap_passive.arap_first_piola_kirchhoff_stress_func(F=F, mu=mu)
    PK1 = muscle_fraction * PK1_active + (1.0 - muscle_fraction) * PK1_passive
    return PK1


@no_type_check
@wp.func
def arap_active_energy_density_hess_diag_func(
    F: mat33, dh_dX: mat43, params: ArapActiveParams
) -> mat43:
    # A = params.activation
    mu = params.mu
    muscle_fraction = params.muscle_fraction
    U, sigma, V = utils.svd_rv(F)
    lambdas = strain.lambdas(sigma=sigma)  # vec3
    Q0, Q1, Q2 = strain.Qs(U=U, V=V)  # mat33, mat33, mat33
    h4_diag = strain.h4_diag(dh_dX=dh_dX, lambdas=lambdas, Q0=Q0, Q1=Q1, Q2=Q2)  # mat43
    h5_diag = strain.h5_diag(dh_dX=dh_dX)  # mat43
    h = -2.0 * h4_diag + h5_diag  # mat43
    hess_diag_active = 0.5 * mu * h
    hess_diag_passive = _arap_passive.arap_energy_density_hess_diag_func(
        F=F, dh_dX=dh_dX, mu=mu
    )
    hess_diag = (
        muscle_fraction * hess_diag_active + (1.0 - muscle_fraction) * hess_diag_passive
    )
    return hess_diag


@no_type_check
@wp.func
def arap_active_energy_density_hess_quad_func(
    F: mat33, p: mat43, dh_dX: mat43, params: ArapActiveParams
) -> float:
    # A = params.activation
    mu = params.mu
    muscle_fraction = params.muscle_fraction
    U, sigma, V = utils.svd_rv(F)
    lambdas = strain.lambdas(sigma=sigma)  # vec3
    Q0, Q1, Q2 = strain.Qs(U=U, V=V)  # mat33, mat33, mat33
    h4_quad = strain.h4_quad(p=p, dh_dX=dh_dX, lambdas=lambdas, Q0=Q0, Q1=Q1, Q2=Q2)
    h5_quad = strain.h5_quad(p=p, dh_dX=dh_dX)  # float
    h = -2.0 * h4_quad + h5_quad  # float
    hess_quad_active = 0.5 * mu * h
    hess_quad_passive = _arap_passive.arap_energy_density_hess_quad_func(
        F=F, p=p, dh_dX=dh_dX, mu=mu
    )
    return (
        muscle_fraction * hess_quad_active + (1.0 - muscle_fraction) * hess_quad_passive
    )
