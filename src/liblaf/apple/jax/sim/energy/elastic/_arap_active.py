from collections.abc import Callable
from typing import Self, override

import attrs
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from liblaf.apple.jax import math, tree
from liblaf.apple.jax.sim.energy.elastic._elastic import Elastic
from liblaf.apple.jax.sim.region import Region
from liblaf.apple.jax.typing import UpdatesData, UpdatesIndex, Vector

from . import utils


@tree.pytree
class ARAPActive(Elastic):
    activation: Float[Array, "c 6"] = tree.array()
    mu: Float[Array, " c"] = tree.array()

    @classmethod
    def from_region(cls, region: Region, **kwargs) -> Self:
        return cls(
            region=region,
            activation=region.cell_data["activation"],
            mu=region.cell_data["mu"],
            **kwargs,
        )

    @override
    def energy_density(self, F: Float[Array, "c q J J"]) -> Float[Array, "c q"]:
        A: Float[Array, " c #q J J"] = utils.make_activation(self.activation)[
            :, jnp.newaxis, :, :
        ]
        mu: Float[Array, " c #q"] = self.mu[:, jnp.newaxis]
        R: Float[Array, "c q J J"]
        R, _ = math.polar_rv(F)
        return mu * math.frobenius_norm_square(F - R @ A)

    def mixed_derivative_prod_activation(self, u: Vector, p: Vector) -> Vector:
        def jac(q: Float[Array, "c 6"]) -> Vector:
            energy: Self = attrs.evolve(self, activation=q)
            data: UpdatesData
            index: UpdatesIndex
            data, index = energy.jac(u)
            jac: Vector = jax.ops.segment_sum(data, index, num_segments=u.shape[0])
            return jac

        vjp: Callable[[Vector], Float[Array, "c 6"]]
        _, vjp = jax.vjp(jac, self.activation)

        output: Float[Array, "c 6"]
        (output,) = vjp(p)
        return output
