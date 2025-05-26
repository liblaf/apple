from collections.abc import Sequence

import jax
from jax.typing import ArrayLike
from jaxtyping import PRNGKeyArray


class Random:
    key: PRNGKeyArray

    def __init__(self, seed: ArrayLike = 0) -> None:
        self.key = jax.random.key(seed)

    @property
    def subkey(self) -> PRNGKeyArray:
        subkey: PRNGKeyArray
        self.key, subkey = jax.random.split(self.key)
        return subkey

    def uniform(self, shape: Sequence[int] = (), **kwargs) -> jax.Array:
        return jax.random.uniform(self.subkey, shape=shape, **kwargs)
