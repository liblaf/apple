import jax.numpy as jnp
from jaxtyping import Array, Float


def normalize(v: Float[Array, "*batch dim"]) -> Float[Array, "*batch dim"]:
    norm: Float[Array, "*batch 1"] = jnp.linalg.norm(v, axis=-1, keepdims=True)
    return v / jnp.where(norm == 0.0, 1.0, norm)
