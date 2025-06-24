import abc
from collections.abc import Sequence
from typing import Self

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike, Integer

from liblaf.apple import struct

from .typed import IndexUpdateRef


@struct.pytree
class DOFs(struct.PyTreeMixin, abc.ABC):
    shape: Sequence[int] = struct.static()

    @classmethod
    def union(cls, *dofs: "DOFs") -> "DOFs":
        from .array import DOFsArray

        if not dofs:
            return DOFsArray(shape=(), _array=jnp.empty((0,), dtype=int))
        array: Integer[jax.Array, " N"] = jnp.concat(
            [jnp.asarray(d).ravel() for d in dofs]
        )
        return DOFsArray(shape=array.shape, _array=array)

    @abc.abstractmethod
    def __jax_array__(self) -> Integer[jax.Array, " N"]:
        raise NotImplementedError

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    def ravel(self) -> Self:
        return self.evolve(shape=(np.prod(self.shape),))

    # region Modifications

    @abc.abstractmethod
    def at(self, x: ArrayLike, /) -> IndexUpdateRef:
        raise NotImplementedError

    def set(self, x: ArrayLike, y: ArrayLike, /) -> jax.Array:
        ref: IndexUpdateRef = self.at(x)
        y_flat: jax.Array = jnp.ravel(y)
        return ref.set(y_flat).reshape(jnp.shape(x))

    def add(self, x: ArrayLike, y: ArrayLike, /) -> jax.Array:
        ref: IndexUpdateRef = self.at(x)
        y_flat: jax.Array = jnp.ravel(y)
        return ref.add(y_flat).reshape(jnp.shape(x))

    def get(self, x: ArrayLike, /) -> jax.Array:
        ref: IndexUpdateRef = self.at(x)
        return ref.get().reshape(self.shape)

    # endregion Modifications
