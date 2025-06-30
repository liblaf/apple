from typing import Self, override

import jax
import jax.numpy as jnp
from jaxtyping import Float

from liblaf.apple import sim, struct, utils

type FloatScalar = Float[jax.Array, ""]


@struct.pytree
class EnergyZero(sim.Energy):
    _actors: struct.NodeContainer[sim.Actor] = struct.container(
        factory=struct.NodeContainer
    )

    @classmethod
    def from_actors(cls, *actors: sim.Actor) -> Self:
        return cls(_actors=struct.NodeContainer(actors))

    @property
    @override
    def actors(self) -> struct.NodeContainer[sim.Actor]:
        return self._actors

    @override
    def with_actors(self, actors: struct.NodeContainer[sim.Actor]) -> Self:
        return self.evolve(_actors=actors)

    @override
    @utils.jit_method(inline=True)
    def fun(self, x: struct.ArrayDict, /, params: sim.GlobalParams) -> FloatScalar:
        return jnp.zeros(())

    @override
    @utils.jit_method(inline=True)
    def jac(self, x: struct.ArrayDict, /, params: sim.GlobalParams) -> struct.ArrayDict:
        return struct.ArrayDict(
            {actor.id: jnp.zeros_like(x[actor.id]) for actor in self.actors.values()}
        )

    @override
    @utils.jit_method(inline=True)
    def hess_diag(
        self, x: struct.ArrayDict, /, params: sim.GlobalParams
    ) -> struct.ArrayDict:
        return struct.ArrayDict(
            {actor.id: jnp.zeros_like(x[actor.id]) for actor in self.actors.values()}
        )

    @override
    @utils.jit_method(inline=True)
    def hess_quad(
        self, x: struct.ArrayDict, p: struct.ArrayDict, /, params: sim.GlobalParams
    ) -> FloatScalar:
        return jnp.zeros(())
