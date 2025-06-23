from collections.abc import Sequence
from typing import Any, Self, override

import jax
from jaxtyping import Integer

from liblaf.apple.struct import tree

from .index import Index


@tree.pytree
class IndexArray(Index):
    _index: Integer[jax.Array, "..."] = tree.array()

    def __getitem__(self, index: Any) -> Self:
        return self.evolve(_index=self._index[index])

    @property
    @override
    def index(self) -> Integer[jax.Array, "..."]:
        return self.integers

    @property
    @override
    def integers(self) -> Integer[jax.Array, "..."]:
        return self._index

    @property
    @override
    def shape(self) -> Sequence[int]:
        return self.integers.shape
