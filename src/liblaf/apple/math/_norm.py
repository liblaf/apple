import jax
import jax.numpy as jnp
from jaxtyping import Float


def norm_sqr(x: Float[jax.Array, "..."]) -> Float[jax.Array, ""]:
    return jnp.vdot(x, x)
