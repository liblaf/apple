import jax.numpy as jnp
from jaxtyping import Array, Float


def fro_norm_square(a: Float[Array, "*batch I I"]) -> Float[Array, "*batch"]:
    return jnp.sum(jnp.square(a), axis=(-2, -1))
