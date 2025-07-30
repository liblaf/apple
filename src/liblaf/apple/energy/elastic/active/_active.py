from typing import Self, override

import einops
import jax.numpy as jnp
from jaxtyping import Array, Float

from liblaf.apple import sim, struct, utils
from liblaf.apple.energy.elastic._elastic import Elastic


class Active(Elastic):
    passive: Elastic = struct.field()

    @classmethod
    def from_passive(cls, passive: Elastic) -> Self:
        return cls(actor=passive.actor, passive=passive)

    @property
    @override
    @utils.jit(inline=True)
    def region(self) -> sim.Region:
        region: sim.Region = self.passive.region
        activation: Float[Array, "c J J"] = self.actor.cell_data["activation"]
        act_inv: Float[Array, "c J J"] = jnp.linalg.inv(activation)
        dhdX: Float[Array, "c q a J"] = einops.einsum(
            region.dhdX, act_inv, "c q a I, c I J -> c q a J"
        )
        return region.replace(dhdX=dhdX)
