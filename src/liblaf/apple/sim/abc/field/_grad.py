from collections.abc import Sequence
from typing import Self, override

import chex
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, DTypeLike

from liblaf import grapes
from liblaf.apple import math
from liblaf.apple.sim.abc.region import Region

from ._field import Field, FieldLike


class FieldGrad(Field):
    @classmethod
    def from_region(
        cls,
        region: Region,
        values: FieldLike = 0.0,
        *,
        dim: int | Sequence[int] | None = None,
        dtype: DTypeLike | None = None,
    ) -> Self:
        values = jnp.asarray(values)
        if values.ndim == 0:
            values = math.broadcast_to(
                values, (region.n_cells, region.quadrature.n_points, 1)
            )
        if values.shape[:2] != (region.n_cells, region.quadrature.n_points):
            values = math.broadcast_to(
                values, (region.n_cells, region.quadrature.n_points, *values.shape)
            )
        if values.ndim == 2:
            values = jnp.expand_dims(values, axis=-1)
        if dim is not None:
            dim = grapes.as_sequence(dim)
            values = math.broadcast_to(
                values, (region.n_cells, region.quadrature.n_points, *dim)
            )
        if dtype is not None:
            values = jnp.asarray(values, dtype=dtype)
        return cls(
            _region=region,
            _shape_dtype=jax.ShapeDtypeStruct(values.shape[2:], values.dtype),
            _values=values,
        )

    @property
    @override
    def shape(self) -> Sequence[int]:
        return (self.n_cells, self.quadrature.n_points, *self.dim)

    @override
    def from_values(self, values: ArrayLike, /) -> Self:
        values = jnp.asarray(values)
        chex.assert_shape(values, (self.n_cells, self.quadrature.n_points, ...))
        return self.replace(
            _shape_dtype=jax.ShapeDtypeStruct(values.shape[2:], dtype=values.dtype),
            _values=values,
        )
