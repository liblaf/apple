import jax.numpy as jnp
from jax import Array
from jaxtyping import ArrayLike, Float


def make_activation(activation: Float[Array, "c A"]) -> Float[Array, "c J J"]:
    n_cells: int = activation.shape[0]
    A: Float[Array, "c J J"] = jnp.empty((n_cells, 3, 3), dtype=activation.dtype)
    A = A.at[:, 0, 0].set(activation[:, 0])
    A = A.at[:, 0, 1].set(activation[:, 1])
    A = A.at[:, 0, 2].set(activation[:, 2])
    A = A.at[:, 1, 1].set(activation[:, 3])
    A = A.at[:, 1, 2].set(activation[:, 4])
    A = A.at[:, 2, 2].set(activation[:, 5])
    A = A.at[:, 1, 0].set(activation[:, 1])
    A = A.at[:, 2, 0].set(activation[:, 2])
    A = A.at[:, 2, 1].set(activation[:, 4])
    return A


def as_activation(a: ArrayLike) -> Float[Array, "c J J"]:
    return jnp.reshape(a, (-1, 3, 3))
