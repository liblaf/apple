import jax
import jax.numpy as jnp
from jaxtyping import Float


def rosen(x: Float[jax.Array, " N"]) -> Float[jax.Array, ""]:
    return jnp.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)
