import jax.numpy as jnp
from numpy.typing import ArrayLike

from ._array import IndexArray
from ._index import Index
from ._slice import IndexSlice


def make_index(index: slice | ArrayLike | None = None) -> Index:
    if index is None:
        return IndexSlice()
    if isinstance(index, slice):
        return IndexSlice(index)
    return IndexArray(jnp.asarray(index))
