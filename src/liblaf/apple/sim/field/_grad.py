from collections.abc import Sequence
from typing import Self, override

import chex
import jax
import jax.numpy as jnp
from jaxtyping import DTypeLike, Float
from numpy.typing import ArrayLike

from liblaf.apple import struct
from liblaf.apple.sim import region as _r

from ._field import Field


class FieldGrad(Field):
    _region: _r.Region = struct.data(default=None)
    _values: Float[jax.Array, "cells q *dim"] = struct.array(default=None)

    @classmethod
    def from_region(
        cls,
        region: _r.Region,
        values: Float[ArrayLike, "cells q *dim"] | None = None,
        *,
        dim: int | Sequence[int] = (3,),
        dtype: DTypeLike = float,
    ) -> Self:
        self: Self = cls(_shape_dtype=jax.ShapeDtypeStruct(dim, dtype), _region=region)
        if values is not None:
            self = self.with_values(values)
        return self

    @property
    @override
    def region(self) -> _r.Region:
        return self._region

    @property
    @override
    def dim(self) -> tuple[int, ...]:
        return self.values.shape[2:]

    @property
    @override
    def dtype(self) -> jnp.dtype:
        return self.values.dtype

    @property
    @override
    def values(self) -> Float[jax.Array, "cells q *dim"]:
        return self._values

    @property
    def integral(self) -> Float[jax.Array, "*dim"]:
        return self.region.integrate(self.values)

    def with_values(
        self, values: Float[ArrayLike, "cells q *dim"] | None = None
    ) -> Self:
        values = jnp.asarray(values)
        chex.assert_shape(values, (self.n_cells, self.quadrature.n_points, *self.dim))
        return self.evolve(_values=values)
