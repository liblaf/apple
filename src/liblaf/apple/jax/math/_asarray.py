import jax.numpy as jnp
from jaxtyping import Array
from numpy.typing import ArrayLike

from liblaf import grapes


@grapes.wraps(jnp.asarray)
def asarray(a: ArrayLike, *args, **kwargs) -> Array:
    return jnp.asarray(a, *args, **kwargs)
