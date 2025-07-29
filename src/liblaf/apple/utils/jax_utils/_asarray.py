import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike


def asarray(x: ArrayLike) -> Array:
    """Super-fast (100+x faster) replacement of `jnp.asarray()`."""
    if isinstance(x, jax.Array):
        return x
    return jnp.asarray(x)
