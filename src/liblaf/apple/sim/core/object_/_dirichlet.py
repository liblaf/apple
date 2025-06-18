from typing import overload

import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike

from liblaf.apple import struct
from liblaf.apple.sim.core.field import AbstractField, FieldLike


class Dirichlet(struct.PyTree):
    index: struct.DofMap = struct.data(default=None)
    values: jax.Array = struct.array(default=None)

    @overload
    def apply(self, field: AbstractField, /) -> AbstractField: ...
    @overload
    def apply(self, field: ArrayLike, /) -> jax.Array: ...
    def apply(self, field: FieldLike, /) -> FieldLike:
        return self._set(field, self.values)

    @overload
    def zero(self, field: AbstractField, /) -> AbstractField: ...
    @overload
    def zero(self, field: ArrayLike, /) -> jax.Array: ...
    def zero(self, field: FieldLike, /) -> FieldLike:
        return self._set(field, 0.0)

    def _set(self, field: FieldLike, values: ArrayLike) -> FieldLike:
        if isinstance(field, AbstractField):
            return field.with_values(
                self.index.set(field.values.ravel(), values).reshape(field.shape)
            )
        field = jnp.asarray(field)
        return self.index.set(field.ravel(), values).reshape(field.shape)
