from typing import Self

import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from liblaf.apple.jax import math, tree
from liblaf.apple.jax.sim.energy.elastic._elastic import Elastic
from liblaf.apple.jax.sim.region import Region


@tree.pytree
class ARAP(Elastic):
    mu: Float[Array, " c"]

    @classmethod
    def from_region(cls, region: Region, **kwargs) -> Self:
        return cls(region=region, mu=region.cell_data["mu"], **kwargs)

    def energy_density(self, F: Float[Array, "c q J J"]) -> Float[Array, "c q"]:
        mu: Float[Array, " c #q"] = self.mu[:, jnp.newaxis]
        R: Float[Array, "c q J J"]
        R, _ = math.polar_rv(F)
        return 0.5 * mu * math.frobenius_norm_square(F - R)
