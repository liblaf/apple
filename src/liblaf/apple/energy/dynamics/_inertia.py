from typing import override

import flax
import flax.struct
import jax
import jax.numpy as jnp
from jaxtyping import Float

from liblaf.apple import physics, utils


class Inertia(physics.Energy):
    field_id: str = "field"
    mass: Float[jax.Array, ""] = flax.struct.field(default=jnp.asarray(1.0))
    time_step: Float[jax.Array, ""] = flax.struct.field(default=jnp.asarray(1.0 / 30.0))

    @utils.jit
    @override
    def fun(self, field: physics.Field) -> Float[jax.Array, ""]:
        x: Float[jax.Array, "points dim"] = field.values
        mass: Float[jax.Array, "points dim"] = jnp.broadcast_to(self.mass, x.shape)
        x_tilde: Float[jax.Array, "points dim"] = (
            x
            + self.time_step * field.velocities
            + self.time_step**2 * field.forces / mass
        )
        fun: Float[jax.Array, ""] = 0.5 * jnp.sum(mass * (x - x_tilde) ** 2)
        fun /= self.time_step**2
        return fun

    @utils.jit
    @override
    def jac(self, field: physics.Field) -> Float[jax.Array, " DoF"]:
        x: Float[jax.Array, "points dim"] = field.values
        mass: Float[jax.Array, "points dim"] = jnp.broadcast_to(self.mass, x.shape)
        x_tilde: Float[jax.Array, "points dim"] = (
            x
            + self.time_step * field.velocities
            + self.time_step**2 * field.forces / mass
        )
        jac: Float[jax.Array, "points dim"] = mass * (x - x_tilde)
        jac /= self.time_step**2
        return jac.ravel()

    @utils.jit
    @override
    def hess_diag(self, field: physics.Field) -> Float[jax.Array, " DoF"]:
        x: Float[jax.Array, "points dim"] = field.values
        mass: Float[jax.Array, "points dim"] = jnp.broadcast_to(self.mass, x.shape)
        return mass.ravel()

    @utils.jit
    @override
    def hess_quad(self, field: physics.Field, p: physics.Field) -> Float[jax.Array, ""]:
        x: Float[jax.Array, "points dim"] = field.values
        mass: Float[jax.Array, "points dim"] = jnp.broadcast_to(self.mass, x.shape)
        return jnp.sum(mass * p.values**2)
