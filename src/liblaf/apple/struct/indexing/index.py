from collections.abc import Sequence
from typing import Any

import jax
from jaxtyping import ArrayLike, Integer

from liblaf.apple.struct import tree


@tree.pytree
class Index(tree.PyTreeMixin):
    def __getitem__(self, index: Any, /) -> "Index":
        raise NotImplementedError

    @property
    def index(self) -> Any:
        raise NotImplementedError

    @property
    def integers(self) -> Integer[jax.Array, "..."]:
        raise NotImplementedError

    @property
    def shape(self) -> Sequence[int]:
        raise NotImplementedError

    def add(self, x: jax.Array, y: ArrayLike) -> jax.Array:
        return x.at[self.index].add(y)

    def get(self, x: jax.Array) -> jax.Array:
        return x[self.index]

    def set(self, x: jax.Array, y: ArrayLike) -> jax.Array:
        return x.at[self.index].set(y)
