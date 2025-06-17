from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, Integer

from liblaf.apple import struct


class DofMap(struct.PyTree):
    def __getitem__(self, idx: Any) -> "DofMap":
        from ._integer import DofMapInteger

        return DofMapInteger(self.integers)[idx]

    @property
    def index(self) -> Any:
        return self.integers

    @property
    def integers(self) -> Integer[jax.Array, "..."]:
        raise NotImplementedError

    @eqx.filter_jit
    def add(self, x: ArrayLike, y: ArrayLike) -> jax.Array:
        x = jnp.asarray(x)
        return x.at[self.index].add(y)

    @eqx.filter_jit
    def get(self, x: ArrayLike) -> jax.Array:
        x = jnp.asarray(x)
        return x[self.index]

    @eqx.filter_jit
    def set(self, x: ArrayLike, y: ArrayLike) -> jax.Array:
        x = jnp.asarray(x)
        return x.at[self.index].set(y)
