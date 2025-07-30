from typing import no_type_check

import warp as wp

from liblaf.apple import utils
from liblaf.apple.typed.warp import mat33, mat43

from . import func


@no_type_check
@utils.jax_kernel
def arap_energy_density_kernel(
    F: wp.array(dtype=mat33), mu: wp.array(dtype=float), Psi: wp.array(dtype=float)
) -> None:
    tid = wp.tid()
    Psi[tid] = func.arap_energy_density_func(F=F[tid], mu=mu[tid])


@no_type_check
@utils.jax_kernel
def arap_first_piola_kirchhoff_stress_kernel(
    F: wp.array(dtype=mat33),
    mu: wp.array(dtype=float),
    PK1: wp.array(dtype=mat33),
) -> None:
    tid = wp.tid()
    PK1[tid] = func.arap_first_piola_kirchhoff_stress_func(F=F[tid], mu=mu[tid])


@no_type_check
@utils.jax_kernel
def arap_energy_density_hess_diag_kernel(
    F: wp.array(dtype=mat33),
    mu: wp.array(dtype=float),
    dh_dX: wp.array(dtype=mat43),
    hess_diag: wp.array(dtype=mat43),
) -> None:
    tid = wp.tid()
    hess_diag[tid] = func.arap_energy_density_hess_diag_func(
        F=F[tid], mu=mu[tid], dh_dX=dh_dX[tid]
    )


@no_type_check
@utils.jax_kernel
def arap_energy_density_hess_quad_kernel(
    F: wp.array(dtype=mat33),
    p: wp.array(dtype=mat43),
    mu: wp.array(dtype=float),
    dh_dX: wp.array(dtype=mat43),
    hess_quad: wp.array(dtype=float),
) -> None:
    tid = wp.tid()
    hess_quad[tid] = func.arap_energy_density_hess_quad_func(
        F=F[tid], p=p[tid], mu=mu[tid], dh_dX=dh_dX[tid]
    )
