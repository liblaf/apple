from collections.abc import Callable

import equinox as eqx
import jax.flatten_util
from jaxtyping import Array, ArrayLike

from liblaf.apple.jax import math


def flatten[T](obj: T) -> tuple[Array, Callable[[Array], T]]:
    data: T
    meta: T
    data, meta = eqx.partition(obj, eqx.is_array)
    flat: Array
    unravel: Callable[[Array], T]
    flat, unravel = jax.flatten_util.ravel_pytree(data)

    def unflatten(a: ArrayLike, /) -> T:
        a = math.asarray(a, dtype=flat.dtype)
        data: T = unravel(a)
        data = eqx.combine(data, meta)
        return data

    return flat, unflatten
