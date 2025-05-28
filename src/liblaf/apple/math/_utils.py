from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Shaped
from numpy.typing import ArrayLike

from liblaf.apple import utils


@utils.jit(static_argnames=("shape",))
def broadcast_to(
    arr: Shaped[ArrayLike, "..."], shape: Sequence[int]
) -> Shaped[jax.Array, "..."]:
    print("Jit ...")
    arr = jnp.asarray(arr)
    if arr.shape == shape:
        return arr
    if arr.size == np.prod(shape):
        return arr.reshape(shape)
    while arr.ndim < len(shape):
        arr = jnp.expand_dims(arr, axis=-1)
    return jnp.broadcast_to(arr, shape)
