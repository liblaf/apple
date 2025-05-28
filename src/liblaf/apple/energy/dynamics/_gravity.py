from typing import override

import flax
import flax.struct
import jax
import jax.numpy as jnp
from jaxtyping import Float

from liblaf.apple import math, physics, utils


class Gravity(physics.Energy):
    id: str = flax.struct.field(pytree_node=False, default="gravity")
    mass: Float[jax.Array, ""] = flax.struct.field(
        default_factory=lambda: jnp.asarray(1.0)
    )
    gravity: Float[jax.Array, " dim"] = flax.struct.field(
        default_factory=lambda: jnp.asarray([0.0, -9.81, 0.0])
    )

    @override
    @utils.jit
    def fun(self, field: physics.Field) -> Float[jax.Array, ""]:
        x: Float[jax.Array, "points dim"] = field.values
        mass: Float[jax.Array, " points"] = jnp.broadcast_to(
            self.mass, (field.n_points,)
        )
        fun: Float[jax.Array, ""] = -jnp.sum(mass * jnp.dot(x, self.gravity))
        return fun

    @override
    @utils.jit
    def jac(self, field: physics.Field) -> Float[jax.Array, " DoF"]:
        mass: Float[jax.Array, " points"] = math.broadcast_to(
            self.mass, (field.n_points,)
        )
        jac: Float[jax.Array, "points dim"] = -mass[:, None] * self.gravity
        return jac.ravel()

    @override
    @utils.jit
    def hess_diag(self, field: physics.Field) -> Float[jax.Array, " DoF"]:
        return jnp.zeros((field.n_dof,))

    @override
    @utils.jit
    def hess_quad(self, field: physics.Field, p: physics.Field) -> Float[jax.Array, ""]:
        return jnp.asarray(0.0)
