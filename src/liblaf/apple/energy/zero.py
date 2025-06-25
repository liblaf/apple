from typing import Self, override

import jax
import jax.numpy as jnp
from jaxtyping import Float

from liblaf.apple import sim, struct, utils

type FloatScalar = Float[jax.Array, ""]


@struct.pytree
class EnergyZero(sim.Energy):
    @classmethod
    def from_actor(cls, actor: sim.Actor) -> Self:
        return cls(actors=struct.NodeContainer([actor]))

    @override
    @utils.jit_method(inline=True)
    def fun(self, x: struct.ArrayDict, /, params: sim.GlobalParams) -> FloatScalar:
        return jnp.zeros(())

    @override
    @utils.jit_method(inline=True)
    def jac(self, x: struct.ArrayDict, /, params: sim.GlobalParams) -> struct.ArrayDict:
        return struct.ArrayDict({self.actor.id: jnp.zeros_like(x[self.actor.id])})

    @override
    @utils.jit_method(inline=True)
    def hess_diag(
        self, x: struct.ArrayDict, /, params: sim.GlobalParams
    ) -> struct.ArrayDict:
        return struct.ArrayDict({self.actor.id: jnp.zeros_like(x[self.actor.id])})

    @override
    @utils.jit_method(inline=True)
    def hess_quad(
        self, x: struct.ArrayDict, p: struct.ArrayDict, /, params: sim.GlobalParams
    ) -> FloatScalar:
        return jnp.zeros(())
