from typing import Any, Self, override

import jax
from jaxtyping import Integer

from liblaf.apple import struct

from ._index import Index


class IndexArray(Index):
    _index: Integer[jax.Array, " N"] = struct.array()

    def __getitem__(self, item: Any) -> Self:
        return self.evolve(_index=self._index[item])

    @property
    @override
    def index(self) -> Any:
        return self._index
