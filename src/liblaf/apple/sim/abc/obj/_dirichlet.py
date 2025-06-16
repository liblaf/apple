from typing import Self, overload

import attrs
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, Float

from liblaf.apple import struct
from liblaf.apple.sim.abc.field import Field


class Dirichlet(struct.PyTree):
    index: struct.Index = struct.data(
        default=None, converter=attrs.converters.optional(struct.as_index)
    )
    values: Float[jax.Array, " dirichlet"] = struct.array(default=None)

    @classmethod
    def concat(cls, *args: "Dirichlet | None") -> Self:
        args = tuple(d for d in args if d is not None)
        if not args:
            return cls()
        index: struct.Index = struct.concat_index(*(d.index.ravel() for d in args))
        values: jax.Array = jnp.concatenate([d.values.ravel() for d in args])
        return cls(index=index, values=values)

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
