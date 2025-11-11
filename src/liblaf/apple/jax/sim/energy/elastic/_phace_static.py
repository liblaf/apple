from typing import Self, override

import jax.numpy as jnp
from jax import Array
from jaxtyping import Float
from liblaf.peach import tree

from liblaf.apple.jax import math
from liblaf.apple.jax.sim.energy.elastic._elastic import Elastic
from liblaf.apple.jax.sim.region import Region


@tree.define
class PhaceStatic(Elastic):
    mu: Float[Array, " c"] = tree.array()
    lambda_: Float[Array, " c"] = tree.array()

    @override
    @classmethod
    def from_region(cls, region: Region, **kwargs) -> Self:
        return cls(
            region=region,
            lambda_=region.cell_data["lambda"],
            mu=region.cell_data["mu"],
            **kwargs,
        )

    @override
    def energy_density(self, F: Float[Array, "c q J J"]) -> Float[Array, "c q"]:
        lambda_: Float[Array, " c #q"] = self.lambda_[:, jnp.newaxis]
        mu: Float[Array, " c #q"] = self.mu[:, jnp.newaxis]
        R: Float[Array, "c q J J"]
        R, _ = math.polar_rv(F)
        J: Float[Array, "c q"] = jnp.linalg.det(F)
        Psi_ARAP: Float[Array, "c q"] = mu * math.fro_norm_square(F - R)
        Psi_volume_preserving: Float[Array, "c q"] = lambda_ * (J - 1.0) ** 2
        Psi: Float[Array, "c q"] = 2.0 * Psi_ARAP + Psi_volume_preserving
        return Psi
