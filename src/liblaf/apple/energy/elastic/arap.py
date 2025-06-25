from typing import no_type_check, override

import jax
import warp as wp
from jaxtyping import Float

from liblaf.apple import func, sim, struct, utils
from liblaf.apple.typed.warp import mat33, mat43

from .elastic import Elastic


@struct.pytree
class ARAP(Elastic):
    r"""As-Rigid-As-Possible.

    $$
    \Psi = \frac{\mu}{2} \|F - R\|_F^2 = \frac{\mu}{2} (I_2 - 2 I_1 + 3)
    $$
    """

    @property
    def mu(self) -> Float[jax.Array, " c"]:
        return self.actor.cell_data["mu"]

    @override
    @utils.jit_method(inline=True)
    def energy_density(
        self, field: sim.Field, /, params: sim.GlobalParams
    ) -> Float[jax.Array, "c q"]:
        region: sim.Region = field.region
        F: Float[jax.Array, "c q J J"] = region.deformation_gradient(field.values)
        F: Float[jax.Array, "cq J J"] = region.squeeze_cq(F)
        Psi: Float[jax.Array, " cq"]
        (Psi,) = arap_energy_density_warp(F, self.mu)
        Psi: Float[jax.Array, "c q"] = region.unsqueeze_cq(Psi)
        return Psi

    @override
    @utils.jit_method(inline=True)
    def first_piola_kirchhoff_stress(
        self, field: sim.Field, /, params: sim.GlobalParams
    ) -> Float[jax.Array, "c q J J"]:
        region: sim.Region = field.region
        F: Float[jax.Array, "c q J J"] = region.deformation_gradient(field.values)
        F: Float[jax.Array, "cq J J"] = region.squeeze_cq(F)
        PK1: Float[jax.Array, "cq J J"]
        (PK1,) = arap_first_piola_kirchhoff_stress_warp(F, self.mu)
        PK1: Float[jax.Array, "c q J J"] = region.unsqueeze_cq(PK1)
        return PK1

    @override
    @utils.jit_method(inline=True)
    def energy_density_hess_diag(
        self, field: sim.Field, /, params: sim.GlobalParams
    ) -> Float[jax.Array, "c q a J"]:
        hess_diag: Float[jax.Array, "cells 4 3"]
        region: sim.Region = field.region
        F: Float[jax.Array, "c q J J"] = region.deformation_gradient(field.values)
        F: Float[jax.Array, "cq J J"] = region.squeeze_cq(F)
        dhdX: Float[jax.Array, "cq a J"] = region.squeeze_cq(region.dhdX)
        hess_diag: Float[jax.Array, "cq a J"]
        (hess_diag,) = arap_energy_density_hess_diag_warp(F, self.mu, dhdX)
        hess_diag: Float[jax.Array, "c q a J"] = region.unsqueeze_cq(hess_diag)
        return hess_diag

    @override
    @utils.jit_method(inline=True)
    def energy_density_hess_quad(
        self, field: sim.Field, p: sim.Field, /, params: sim.GlobalParams
    ) -> Float[jax.Array, "c q"]:
        region: sim.Region = field.region
        F: Float[jax.Array, "c q J J"] = region.deformation_gradient(field.values)
        F: Float[jax.Array, "cq J J"] = region.squeeze_cq(F)
        dhdX: Float[jax.Array, "cq a J"] = region.squeeze_cq(region.dhdX)
        hess_quad: Float[jax.Array, " cq"]
        (hess_quad,) = arap_energy_density_hess_quad_warp(
            F, region.scatter(p.values), self.mu, dhdX
        )
        hess_quad: Float[jax.Array, "c q"] = region.unsqueeze_cq(hess_quad)
        return hess_quad


@no_type_check
@utils.jax_kernel
def arap_energy_density_warp(
    F: wp.array(dtype=mat33), mu: wp.array(dtype=float), Psi: wp.array(dtype=float)
) -> None:
    tid = wp.tid()
    Psi[tid] = func.elastic.arap_energy_density(F=F[tid], mu=mu[tid])


@no_type_check
@utils.jax_kernel
def arap_first_piola_kirchhoff_stress_warp(
    F: wp.array(dtype=mat33),
    mu: wp.array(dtype=float),
    PK1: wp.array(dtype=mat33),
) -> None:
    tid = wp.tid()
    PK1[tid] = func.elastic.arap_first_piola_kirchhoff_stress(F=F[tid], mu=mu[tid])


@no_type_check
@utils.jax_kernel
def arap_energy_density_hess_diag_warp(
    F: wp.array(dtype=mat33),
    mu: wp.array(dtype=float),
    dh_dX: wp.array(dtype=mat43),
    hess_diag: wp.array(dtype=mat43),
) -> None:
    tid = wp.tid()
    hess_diag[tid] = func.elastic.arap_energy_density_hess_diag(
        F=F[tid], mu=mu[tid], dh_dX=dh_dX[tid]
    )


@no_type_check
@utils.jax_kernel
def arap_energy_density_hess_quad_warp(
    F: wp.array(dtype=mat33),
    p: wp.array(dtype=mat43),
    mu: wp.array(dtype=float),
    dh_dX: wp.array(dtype=mat43),
    hess_quad: wp.array(dtype=float),
) -> None:
    tid = wp.tid()
    hess_quad[tid] = func.elastic.arap_energy_density_hess_quad(
        F=F[tid], p=p[tid], mu=mu[tid], dh_dX=dh_dX[tid]
    )
