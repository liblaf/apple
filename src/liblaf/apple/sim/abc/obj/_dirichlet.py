from typing import overload

import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, Float

from liblaf.apple import struct
from liblaf.apple.sim.abc.field import Field


class Dirichlet(struct.PyTree):
    index: struct.DofMap = struct.field(default=None, converter=struct.as_dof_map)
    values: Float[jax.Array, " dirichlet"] = struct.array(default=None)

    @overload
    def apply(self, x: Field, /) -> Field: ...
    @overload
    def apply(self, x: ArrayLike, /) -> jax.Array: ...
    def apply(self, x: Field | ArrayLike, /) -> Field | jax.Array:
        return self._set(x, self.values)

    @overload
    def zero(self, x: Field, /) -> Field: ...
    @overload
    def zero(self, x: ArrayLike, /) -> jax.Array: ...
    def zero(self, x: Field | ArrayLike, /) -> Field | jax.Array:
        return self._set(x, 0.0)

    def _set(self, x: Field | ArrayLike, /, values: ArrayLike) -> Field | jax.Array:
        if isinstance(x, Field):
            if self.index is None:
                return x
            return x.with_values(self.index.set(x.values, values))
        x = jnp.asarray(x)
        if self.index is None:
            return x
        return self.index.set(x, values)
