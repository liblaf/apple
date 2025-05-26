from typing import no_type_check

import warp as wp

from liblaf.apple.func import matrix as _func


@no_type_check
@wp.func
def arap_energy_density(F: wp.mat33, mu: float) -> float:
    R, S = _func.polar_rv(F)
    Psi = 0.5 * mu * _func.frobenius_norm_square(F - R)
    return Psi


@no_type_check
@wp.func
def arap_first_piola_kirchhoff_stress(F: wp.mat33, mu: float) -> wp.mat33:
    R, S = _func.polar_rv(F)
    PK1 = mu * (F - R)
    return PK1
