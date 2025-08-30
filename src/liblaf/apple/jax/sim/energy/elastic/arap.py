import attrs
import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import Float

from liblaf.apple import struct
from liblaf.apple.jax import math
from liblaf.apple.jax.sim.energy.elastic._elastic import Elastic
from liblaf.apple.types.jax._types import Vector


@struct.pytree
class ARAP(Elastic):
    activation: Float[Array, "c J J"]
    mu: Float[Array, " c"]

    def energy_density(self, F: Float[Array, "c q J J"]) -> Float[Array, "c q"]:
        A: Float[Array, "c J J"] = self.activation
        mu: Float[Array, " c"] = self.mu
        R: Float[Array, "c q J J"]
        R, _ = math.polar_rv(F)
        RA: Float[Array, "c q J J"] = jnp.matmul(R, A[:, jnp.newaxis, :, :])
        return mu[:, jnp.newaxis, jnp.newaxis] * jnp.sum((F - RA) ** 2)

    def mixed_derivative_prod(self, u: Vector, p: Vector) -> Vector:
        def fun(x: Array, q: Array) -> Array:
            energy = attrs.evolve(self, activation=q)
            return energy.fun(x)

        def jac_q(x: Array) -> Array:
            return jax.grad(fun, argnums=1)(x, self.activation)

        _, outputs = jax.jvp(jac_q, (u,), (p,))
        return outputs
