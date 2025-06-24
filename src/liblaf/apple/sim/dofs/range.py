from typing import override

import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, Integer

from liblaf.apple import struct

from .dofs import DOFs
from .typed import IndexUpdateRef


@struct.pytree
class DOFsRange(DOFs):
    _range: range = struct.static()

    @override
    def __jax_array__(self) -> Integer[jax.Array, " N"]:
        return jnp.arange(self.start, self.stop, self.step).reshape(self.shape)

    @override
    def at(self, x: ArrayLike, /) -> IndexUpdateRef:
        x_flat: jax.Array = jnp.ravel(x)
        return x_flat.at[self.slice]

    @property
    def slice(self) -> slice:
        return slice(self.start, self.stop, self.step)

    @property
    def start(self) -> int:
        return self._range.start

    @property
    def stop(self) -> int:
        return self._range.stop

    @property
    def step(self) -> int:
        return self._range.step
