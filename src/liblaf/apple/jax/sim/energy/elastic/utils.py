import jax.numpy as jnp
from jax import Array
from jaxtyping import ArrayLike, Float


def make_activation(activation: Float[Array, "c 6"]) -> Float[Array, "c 3 3"]:
    n_cells: int = activation.shape[0]
    A: Float[Array, "c 3 3"] = jnp.empty((n_cells, 3, 3), activation.dtype)
    A = A.at[:, 0, 0].set(activation[:, 0])
    A = A.at[:, 1, 1].set(activation[:, 1])
    A = A.at[:, 2, 2].set(activation[:, 2])
    A = A.at[:, 0, 1].set(activation[:, 3])
    A = A.at[:, 0, 2].set(activation[:, 4])
    A = A.at[:, 1, 2].set(activation[:, 5])
    A = A.at[:, 1, 0].set(activation[:, 3])
    A = A.at[:, 2, 0].set(activation[:, 4])
    A = A.at[:, 2, 1].set(activation[:, 5])
    A += jnp.identity(3, activation.dtype)
    return A


def as_activation(a: ArrayLike) -> Float[Array, "c 6"]:
    return jnp.reshape(a, (-1, 6))
