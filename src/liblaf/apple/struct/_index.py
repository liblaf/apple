from collections.abc import Sequence
from typing import Any, override

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike, Integer

from ._pytree import PyTree, array

type IndexLike = Index | Integer[ArrayLike, "..."]


class Index(PyTree):
    def __call__(self, x: ArrayLike, /) -> jax.Array:
        x = jnp.asarray(x)
        return x[self.index]

    def __getitem__(self, index: Any) -> "Index":
        raise NotImplementedError

    @property
    def integers(self) -> Integer[jax.Array, "..."]:
        raise NotImplementedError

    @property
    def index(self) -> Any:
        return self.integers

    def concat(self, other: IndexLike, /) -> "Index":
        other = as_index(other)
        return IndexIntegers(jnp.concat([self.integers, other.integers]))

    def add(self, x: jax.Array, y: ArrayLike, /) -> jax.Array:
        return x.at[self.index].add(y)

    def set(self, x: jax.Array, y: ArrayLike, /) -> jax.Array:
        return x.at[self.index].set(y)

    def ravel(self) -> "Index":
        return IndexIntegers(self.integers.ravel())


class IndexIntegers(Index):
    _array: Integer[jax.Array, "..."] = array(default=jnp.empty((), dtype=int))

    @override
    def __getitem__(self, index: Any) -> Index:
        return self.evolve(_array=self._array[index])

    @property
    @override
    def integers(self) -> Integer[jax.Array, "..."]:
        return self._array


def as_index(x: IndexLike) -> Index:
    if isinstance(x, Index):
        return x
    return IndexIntegers(jnp.asarray(x))


def concat_index(*args: IndexLike | None) -> Index:
    indices: list[Index] = [as_index(arg) for arg in args if arg is not None]
    if not indices:
        return IndexIntegers()
    result: Index = indices[0]
    for other in indices[1:]:
        result = result.concat(other)
    return result


def make_index(shape: int | Sequence[int], start: int = 0) -> Index:
    return IndexIntegers(
        _array=jnp.arange(start, start + np.prod(shape)).reshape(shape)
    )
