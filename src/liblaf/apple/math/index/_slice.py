from typing import Any, override

import jax.numpy as jnp

from liblaf.apple import struct

from ._array import IndexArray
from ._index import Index


class IndexSlice(Index):
    _slice: slice = struct.static(default=slice(None))  # pyright: ignore[reportAssignmentType]

    @override
    def __getitem__(self, item: Any) -> "Index":
        if isinstance(item, slice):
            start: int = (
                self._index(item.start) if item.start is not None else self.start
            )
            stop: int = self._index(item.stop) if item.stop is not None else self.stop
            return self.evolve(_slice=slice(start, stop))
        return self.array[item]

    @property
    @override
    def index(self) -> Any:
        return self._slice

    @property
    def array(self) -> IndexArray:
        return IndexArray(jnp.arange(self.start, self.stop))

    @property
    def start(self) -> int:
        if self._slice.start is None:
            return 0
        return self._slice.start

    @property
    def stop(self) -> int:
        if self._slice.stop is None:
            return -1
        return self._slice.stop

    def _index(self, i: int) -> int:
        if i < 0:
            return self.stop + i
        if i >= 0:
            return self.start + i
        raise ValueError(i)
