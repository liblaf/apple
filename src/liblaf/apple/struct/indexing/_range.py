from collections.abc import Sequence
from typing import Any, override

import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, Integer

from liblaf.apple.struct import tree

from ._array import IndexArray
from .index import Index


@tree.pytree
class IndexRange(Index):
    _range: range = tree.static()
    _shape: Sequence[int] = tree.static()

    def __getitem__(self, index: Any, /) -> "Index":
        return self.to_array()[index]

    @property
    @override
    def index(self) -> Integer[jax.Array, "..."]:
        return self.integers

    @property
    @override
    def integers(self) -> Integer[jax.Array, "..."]:
        return jnp.arange(self.start, self.stop, self.step).reshape(self.shape)

    @property
    @override
    def shape(self) -> Sequence[int]:
        return self._shape

    def add(self, x: ArrayLike, y: ArrayLike) -> jax.Array:
        x = jnp.ravel(x)
        y = jnp.ravel(y)
        return x.at[self.slice].add(y).reshape(self.shape)

    @override
    def get(self, x: ArrayLike) -> jax.Array:
        x = jnp.ravel(x)
        return x[self.slice].reshape(self.shape)

    @override
    def set(self, x: ArrayLike, y: ArrayLike) -> jax.Array:
        x = jnp.ravel(x)
        y = jnp.ravel(y)
        return x.at[self.slice].set(y).reshape(self.shape)

    @property
    def slice(self) -> slice:
        return slice(self.start, self.stop, self.step)

    @property
    def start(self) -> int | None:
        return self._range.start

    @property
    def stop(self) -> int | None:
        return self._range.stop

    @property
    def step(self) -> int | None:
        return self._range.step

    def to_array(self) -> IndexArray:
        return IndexArray(self.integers)
