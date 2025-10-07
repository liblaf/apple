import jax.numpy as jnp
from jax import Array
from jaxtyping import DTypeLike, Float


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
    # A += jnp.identity(3, activation.dtype)
    return A


def rest_activation(n_cells: int = 1, dtype: DTypeLike = float) -> Float[Array, "c 6"]:
    activation: Float[Array, "c 6"] = jnp.zeros((n_cells, 6), dtype)
    activation = activation.at[:, 0].set(1.0)
    activation = activation.at[:, 1].set(1.0)
    activation = activation.at[:, 2].set(1.0)
    return activation


def transform_activation(
    activation: Float[Array, "#c 6"], orientation: Float[Array, "#c 3 3"]
) -> Float[Array, "c 6"]:
    activation_mat: Float[Array, "c 3 3"] = make_activation(activation)
    transformed_mat: Float[Array, "c 3 3"] = (
        orientation.mT @ activation_mat @ orientation
    )
    n_cells: int = transformed_mat.shape[0]
    transformed: Float[Array, "c 6"] = jnp.empty((n_cells, 6), activation.dtype)
    transformed = transformed.at[:, 0].set(transformed_mat[:, 0, 0])
    transformed = transformed.at[:, 1].set(transformed_mat[:, 1, 1])
    transformed = transformed.at[:, 2].set(transformed_mat[:, 2, 2])
    transformed = transformed.at[:, 3].set(transformed_mat[:, 0, 1])
    transformed = transformed.at[:, 4].set(transformed_mat[:, 0, 2])
    transformed = transformed.at[:, 5].set(transformed_mat[:, 1, 2])
    return transformed
