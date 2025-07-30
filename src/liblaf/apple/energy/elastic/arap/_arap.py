from typing import override

import jax
from jaxtyping import Float

from liblaf.apple import sim, utils
from liblaf.apple.energy.elastic._elastic import Elastic, Field

from . import kernel


class Arap(Elastic):
    r"""As-Rigid-As-Possible.

    $$
    \Psi = \frac{\mu}{2} \|F - R\|_F^2 = \frac{\mu}{2} (I_2 - 2 I_1 + 3)
    $$
    """

    @property
    def mu(self) -> Float[jax.Array, " c"]:
        return self.actor.cell_data["mu"]

    @override
    @utils.jit(inline=True)
    def energy_density(
        self, field: Field, /, params: sim.GlobalParams
    ) -> Float[jax.Array, "c q"]:
        region: sim.Region = self.region
        F: Float[jax.Array, "c q J J"] = region.deformation_gradient(field)
        F: Float[jax.Array, "cq J J"] = region.squeeze_cq(F)
        Psi: Float[jax.Array, " cq"]
        (Psi,) = kernel.arap_energy_density_kernel(F, self.mu)
        Psi: Float[jax.Array, "c q"] = region.unsqueeze_cq(Psi)
        return Psi

    @override
    @utils.jit(inline=True)
    def first_piola_kirchhoff_stress(
        self, field: Field, /, params: sim.GlobalParams
    ) -> Float[jax.Array, "c q J J"]:
        region: sim.Region = self.region
        F: Float[jax.Array, "c q J J"] = region.deformation_gradient(field)
        F: Float[jax.Array, "cq J J"] = region.squeeze_cq(F)
        PK1: Float[jax.Array, "cq J J"]
        (PK1,) = kernel.arap_first_piola_kirchhoff_stress_kernel(F, self.mu)
        PK1: Float[jax.Array, "c q J J"] = region.unsqueeze_cq(PK1)
        return PK1

    @override
    @utils.jit(inline=True)
    def energy_density_hess_diag(
        self, field: Field, /, params: sim.GlobalParams
    ) -> Float[jax.Array, "c q a J"]:
        region: sim.Region = self.region
        hess_diag: Float[jax.Array, "cells 4 3"]
        F: Float[jax.Array, "c q J J"] = region.deformation_gradient(field)
        F: Float[jax.Array, "cq J J"] = region.squeeze_cq(F)
        dhdX: Float[jax.Array, "cq a J"] = region.squeeze_cq(region.dhdX)
        hess_diag: Float[jax.Array, "cq a J"]
        (hess_diag,) = kernel.arap_energy_density_hess_diag_kernel(F, self.mu, dhdX)
        hess_diag: Float[jax.Array, "c q a J"] = region.unsqueeze_cq(hess_diag)
        return hess_diag

    @override
    @utils.jit(inline=True)
    def energy_density_hess_quad(
        self, field: Field, p: Field, /, params: sim.GlobalParams
    ) -> Float[jax.Array, "c q"]:
        region: sim.Region = self.region
        F: Float[jax.Array, "c q J J"] = region.deformation_gradient(field)
        F: Float[jax.Array, "cq J J"] = region.squeeze_cq(F)
        dhdX: Float[jax.Array, "cq a J"] = region.squeeze_cq(region.dhdX)
        hess_quad: Float[jax.Array, " cq"]
        (hess_quad,) = kernel.arap_energy_density_hess_quad_kernel(
            F, region.scatter(p), self.mu, dhdX
        )
        hess_quad: Float[jax.Array, "c q"] = region.unsqueeze_cq(hess_quad)
        return hess_quad
