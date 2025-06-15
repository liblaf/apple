from collections.abc import Sequence
from typing import Self

import chex
import jax
import jax.numpy as jnp
from jaxtyping import Float
from numpy.typing import ArrayLike

from liblaf.apple import math, struct
from liblaf.apple.sim import region as _r

from ._field import Field


class FieldConcrete(Field):
    _region: _r.Region = struct.data(default=None)
    _shape_dtype: jax.ShapeDtypeStruct = struct.static(default=None)
    _values: Float[jax.Array, " points *dim"] = struct.array(default=None)

    @property
    def region(self) -> _r.Region:
        return self._region

    @property
    def dim(self) -> Sequence[int]:
        return self._shape_dtype.shape

    @property
    def dtype(self) -> jnp.dtype:
        return self._shape_dtype.dtype

    @property
    def values(self) -> Float[jax.Array, " points *dim"]:
        return self._values

    def with_values(
        self, values: Float[ArrayLike, " points *dim"] | None = None
    ) -> Self:
        if values is None:
            return self
        values = math.broadcast_to(values, self.shape)
        chex.assert_shape(values, self.shape)
        return self.evolve(_values=values)
