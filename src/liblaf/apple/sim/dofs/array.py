from typing import override

import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, Integer

from liblaf.apple import struct

from .dofs import DOFs
from .typed import IndexUpdateRef


@struct.pytree
class DOFsArray(DOFs):
    _array: Integer[jax.Array, " N"] = struct.array(
        factory=lambda: jnp.empty((0,), dtype=int)
    )

    @override
    def __jax_array__(self) -> Integer[jax.Array, " N"]:
        return self._array.reshape(self.shape)

    @override
    def at(self, x: ArrayLike, /) -> IndexUpdateRef:
        x_flat: jax.Array = jnp.ravel(x)
        return x_flat.at[self._array]
