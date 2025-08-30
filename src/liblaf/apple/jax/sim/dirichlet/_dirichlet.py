import jax.numpy as jnp
from jaxtyping import Array, Float, Integer

from liblaf.apple.jax import tree
from liblaf.apple.jax.typing import Vector


def _default_index() -> Integer[Array, "*dirichlet"]:
    return jnp.empty((0,), dtype=int)


def _default_values() -> Float[Array, "*dirichlet"]:
    return jnp.empty((0,), dtype=float)


@tree.pytree
class Dirichlet:
    index: Integer[Array, "*dirichlet"] = tree.array(factory=_default_index)
    values: Float[Array, "*dirichlet"] = tree.array(factory=_default_values)

    def apply(self, x: Vector) -> Vector:
        return x.at[self.index].set(self.values)

    def mask(self, x: Vector) -> Vector:
        return x.at[self.index].set(True)

    def zero(self, x: Vector) -> Vector:
        return x.at[self.index].set(0.0)
