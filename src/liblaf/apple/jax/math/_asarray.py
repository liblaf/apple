import jax.numpy as jnp
from jaxtyping import Array
from numpy.typing import ArrayLike

from liblaf import grapes


@grapes.clone_param_spec(jnp.asarray)
def asarray(a: ArrayLike, *args, **kwargs) -> Array:
    return jnp.asarray(a, *args, **kwargs)
