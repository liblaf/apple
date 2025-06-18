from collections.abc import Sequence
from typing import Self, override

import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, Float, Integer

from liblaf.apple import struct
from liblaf.apple.sim.core.region import Region

from ._abc import AbstractField
from ._field import FieldLike


class FieldGrad(AbstractField):
    _values: Float[jax.Array, "cells q *dim"] = struct.array(default=None)

    @classmethod
    def from_region(cls, region: Region, values: FieldLike | None = None) -> Self:
        if values is None:
            values = jnp.zeros((region.n_points, region.quadrature.n_points, 1))
        self: Self = cls(_region=region)
        self = self.from_values(values)
        return self

    @property
    @override
    def values(self) -> Float[jax.Array, "cells q *dim"]:
        return self._values

    @property
    @override
    def dim(self) -> Sequence[int]:
        return self.values.shape[2:]

    @property
    @override
    def integral(self) -> Float[jax.Array, "*dim"]:
        return self.region.integrate(self.values)

    @property
    def boundary(self) -> Self:
        region: Region = self.region.boundary
        return self.evolve(_region=region).with_values(
            self.values[self.region.original_cell_id]
        )

    def extract_cells(
        self, ind: Integer[ArrayLike, " sub_cells"], *, invert: bool = False
    ) -> Self:
        region: Region = self.region.extract_cells(ind=ind, invert=invert)
        return self.evolve(_region=region).with_values(
            self.values[self.region.original_cell_id]
        )

    @override
    def from_values(self, values: FieldLike, /) -> Self:
        values = jnp.asarray(values)
        values = jnp.broadcast_to(
            values, (self.region.n_cells, self.quadrature.n_points, *values.shape[2:])
        )
        return self.evolve(_values=values)
