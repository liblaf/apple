import functools
from collections.abc import Sequence

import jax
import numpy as np

from liblaf import grapes
from liblaf.apple.struct.indexing._range import IndexRange

from ._array import IndexArray
from .index import Index


@functools.singledispatch
def as_index(*args, **kwargs) -> Index:
    raise grapes.error.DispatchLookupError(as_index, *args, **kwargs)


@as_index.register(Index)
def _(index: Index) -> Index:
    return index


@as_index.register(jax.Array)
def _(index: jax.Array) -> Index:
    return IndexArray(index)


def make_index(shape: Sequence[int], *, offset: int = 0) -> Index:
    return IndexRange(_range=range(offset, offset + np.prod(shape)), _shape=shape)
