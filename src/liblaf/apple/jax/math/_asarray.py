import jax.numpy as jnp
from jaxtyping import Array
from numpy.typing import ArrayLike


def asarray(a: ArrayLike, **kwargs) -> Array:
    return jnp.asarray(a, **kwargs)
