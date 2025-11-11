import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float, Integer
from liblaf.peach import tree

from liblaf.apple.jax.typing import Vector


def _default_index() -> Integer[Array, " dirichlet"]:
    return jnp.empty((0,), dtype=int)


def _default_values() -> Float[Array, " dirichlet"]:
    return jnp.empty((0,), dtype=float)


@tree.define
class Dirichlet:
    n_dofs: int = tree.field(default=0)
    index: Integer[Array, " dirichlet"] = tree.array(factory=_default_index)
    values: Float[Array, " dirichlet"] = tree.array(factory=_default_values)
    free_index: Integer[Array, " free"] = tree.array(factory=_default_index)

    @property
    def n_dirichlet(self) -> int:
        return self.index.size

    @property
    def n_free(self) -> int:
        return self.free_index.size

    def apply(self, x: Vector) -> Vector:
        return self.set(x, self.values)

    def get(self, x: Vector) -> Float[Array, " dirichlet"]:
        x_flat: Array = x.flatten()
        return x_flat[self.index]

    def get_free(self, x: Vector) -> Float[Array, " free"]:
        x_flat: Array = x.flatten()
        return x_flat[self.free_index]

    def mask(self, x: Vector) -> Vector:
        return self.set(x, True)  # noqa: FBT003

    def set(self, x: Vector, values: ArrayLike) -> Vector:
        x_flat: Array = x.flatten()
        y_flat: Array = x_flat.at[self.index].set(values)
        return y_flat.reshape(x.shape)

    def set_free(self, x: Vector, values: ArrayLike) -> Vector:
        x_flat: Array = x.flatten()
        y_flat: Array = x_flat.at[self.free_index].set(values)
        return y_flat.reshape(x.shape)

    def zero(self, x: Vector) -> Vector:
        return self.set(x, 0.0)
