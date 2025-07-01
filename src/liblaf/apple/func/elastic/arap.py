r"""As-Rigid-As-Possible.

$$
\Psi = \frac{\mu}{2} \|F - R\|_F^2 = \frac{\mu}{2} (I_2 - 2 I_1 + 3)
$$
"""

from typing import no_type_check

import warp as wp

from liblaf.apple.func import strain, utils
from liblaf.apple.typed.warp import mat33, mat43


@no_type_check
@wp.func
def arap_energy_density(F: mat33, mu: float) -> float:
    R, S = utils.polar_rv(F)
    Psi = 0.5 * mu * utils.frobenius_norm_square(F - R)
    return Psi


@no_type_check
@wp.func
def arap_first_piola_kirchhoff_stress(F: mat33, mu: float) -> mat33:
    R, S = utils.polar_rv(F)
    PK1 = mu * (F - R)
    return PK1


@no_type_check
@wp.func
def arap_energy_density_hess_diag(F: mat33, mu: float, dh_dX: mat43) -> mat43:
    U, sigma, V = utils.svd_rv(F)
    lambdas = strain.lambdas(sigma=sigma)  # vec3
    Q0, Q1, Q2 = strain.Qs(U=U, V=V)  # mat33, mat33, mat33
    h4_diag = strain.h4_diag(dh_dX=dh_dX, lambdas=lambdas, Q0=Q0, Q1=Q1, Q2=Q2)  # mat43
    h5_diag = strain.h5_diag(dh_dX=dh_dX)  # mat43
    h = -2.0 * h4_diag + h5_diag  # mat43
    return 0.5 * mu * h


@no_type_check
@wp.func
def arap_energy_density_hess_quad(F: mat33, p: mat43, mu: float, dh_dX: mat43) -> float:
    U, sigma, V = utils.svd_rv(F)
    lambdas = strain.lambdas(sigma=sigma)  # vec3
    Q0, Q1, Q2 = strain.Qs(U=U, V=V)  # mat33, mat33, mat33
    h4_quad = strain.h4_quad(p=p, dh_dX=dh_dX, lambdas=lambdas, Q0=Q0, Q1=Q1, Q2=Q2)
    h5_quad = strain.h5_quad(p=p, dh_dX=dh_dX)  # float
    h = -2.0 * h4_quad + h5_quad  # float
    return 0.5 * mu * h
