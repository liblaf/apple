from typing import override

import flax
import flax.struct
import jax
import jax.numpy as jnp
from jaxtyping import Float

from liblaf.apple import math, physics, utils


class Inertia(physics.Energy):
    id: str = flax.struct.field(pytree_node=False, default="inertia")
    mass: Float[jax.Array, ""] = flax.struct.field(
        default_factory=lambda: jnp.asarray(1.0)
    )
    time_step: Float[jax.Array, ""] = flax.struct.field(
        default_factory=lambda: jnp.asarray(1.0 / 30.0)
    )

    @override
    @utils.jit
    def fun(self, field: physics.Field) -> Float[jax.Array, ""]:
        x: Float[jax.Array, "points dim"] = field.values
        mass: Float[jax.Array, "points dim"] = self.broadcast_mass(field)
        x_tilde: Float[jax.Array, "points dim"] = (
            field.values_prev
            + self.time_step * field.velocities
            + self.time_step**2 * field.forces / mass
        )
        fun: Float[jax.Array, ""] = 0.5 * jnp.sum(mass * (x - x_tilde) ** 2)
        fun /= self.time_step**2
        return fun

    @override
    @utils.jit
    def jac(self, field: physics.Field) -> Float[jax.Array, " DoF"]:
        x: Float[jax.Array, "points dim"] = field.values
        mass: Float[jax.Array, "points dim"] = self.broadcast_mass(field)
        x_tilde: Float[jax.Array, "points dim"] = (
            field.values_prev
            + self.time_step * field.velocities
            + self.time_step**2 * field.forces / mass
        )
        jac: Float[jax.Array, "points dim"] = mass * (x - x_tilde)
        jac /= self.time_step**2
        return jac.ravel()

    @override
    @utils.jit
    def hess_diag(self, field: physics.Field) -> Float[jax.Array, " DoF"]:
        mass: Float[jax.Array, "points dim"] = self.broadcast_mass(field)
        hess_diag: Float[jax.Array, "points dim"] = mass / self.time_step**2
        return hess_diag.ravel()

    @override
    @utils.jit
    def hess_quad(self, field: physics.Field, p: physics.Field) -> Float[jax.Array, ""]:
        mass: Float[jax.Array, "points dim"] = self.broadcast_mass(field)
        hess_quad: Float[jax.Array, ""] = jnp.sum(mass * p.values**2)
        hess_quad /= self.time_step**2
        return hess_quad

    def broadcast_mass(self, field: physics.Field) -> Float[jax.Array, "points dim"]:
        return math.broadcast_to(self.mass[:, None], field.values.shape)
