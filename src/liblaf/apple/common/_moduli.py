import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Float


def lame_converter(
    E: Float[ArrayLike, "..."], nu: Float[ArrayLike, "..."]
) -> tuple[Float[Array, "..."], Float[Array, "..."]]:
    la: Float[Array, " ..."] = jnp.asarray(E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)))
    mu: Float[Array, " ..."] = jnp.asarray(E / (2.0 * (1.0 + nu)))
    return la, mu
