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
    shape: Sequence[int] = struct.static(default=(0,), kw_only=True)

    @classmethod
    def from_mask(cls, mask: ArrayLike, /) -> "DOFs":
        from .array import DOFsArray

        mask = jnp.asarray(mask).ravel()
        if not mask.any():
            return DOFsArray()
        idx: Integer[jax.Array, " DOF"] = jnp.nonzero(mask)[0]
        return DOFsArray(idx, shape=idx.shape)

    @classmethod
    def union(cls, *dofs: "DOFs") -> "DOFs":
        from .array import DOFsArray

        if not dofs:
            return DOFsArray()
        idx: Integer[jax.Array, " N"] = jnp.concat(
            [jnp.asarray(d).ravel() for d in dofs]
        )
        return DOFsArray(idx, shape=idx.shape)

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
