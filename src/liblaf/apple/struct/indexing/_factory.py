from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Integer

from ._index import Index


def as_index(
    index: Integer[ArrayLike, "*dim"] | Integer[Index, "*dim"],
) -> Integer[Index, "*dim"]:
    if isinstance(index, Index):
        return index
    index: Integer[Array, "*dim"] = jnp.asarray(index)
    index_flat: Integer[Array, " N"] = index.ravel()
    return Index(index=index_flat, shape=index.shape)


def make_index(shape: Sequence[int], *, offset: int = 0) -> Index:
    return Index(
        index=jnp.arange(offset, offset + np.prod(shape), dtype=int), shape=shape
    )
