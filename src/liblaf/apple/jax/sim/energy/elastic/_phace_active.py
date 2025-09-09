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
class PhaceActive(Elastic):
    activation: Float[Array, "c J J"] = tree.array()
    lambda_: Float[Array, " c"] = tree.array()
    mu: Float[Array, " c"] = tree.array()

    @override
    @classmethod
    def from_region(cls, region: Region, **kwargs) -> Self:
        return cls(
            region=region,
            activation=region.cell_data["activation"],
            lambda_=region.cell_data["lambda"],
            mu=region.cell_data["mu"],
            **kwargs,
        )

    @override
    def energy_density(self, F: Float[Array, "c q J J"]) -> Float[Array, "c q"]:
        A: Float[Array, " c #q J J"] = utils.make_activation(self.activation)[
            :, jnp.newaxis, :, :
        ]
        lambda_: Float[Array, " c #q"] = self.lambda_[:, jnp.newaxis]
        mu: Float[Array, " c #q"] = self.mu[:, jnp.newaxis]
        R: Float[Array, "c q J J"]
        R, _ = math.polar_rv(F)
        J: Float[Array, "c q"] = jnp.linalg.det(F)
        Psi_ARAP: Float[Array, "c q"] = mu * math.frobenius_norm_square(F - R @ A)
        Psi_volume_preserving: Float[Array, "c q"] = lambda_ * (J - 1.0) ** 2
        Psi: Float[Array, "c q"] = Psi_ARAP + Psi_volume_preserving
        return Psi

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
