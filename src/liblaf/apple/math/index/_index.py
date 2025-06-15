from typing import Any

import jax
from jaxtyping import ArrayLike

from liblaf.apple import struct, utils


class Index(struct.PyTree):
    @utils.jit
    def __call__(self, x: jax.Array) -> jax.Array:
        return x[self.index]

    def __getitem__(self, item: Any) -> "Index":
        raise NotImplementedError

    @property
    def index(self) -> Any:
        raise NotImplementedError

    @utils.jit
    def add(self, x: jax.Array, y: ArrayLike) -> jax.Array:
        return x.at[self.index].add(y)

    @utils.jit
    def set(self, x: jax.Array, y: ArrayLike) -> jax.Array:
        return x.at[self.index].set(y)
