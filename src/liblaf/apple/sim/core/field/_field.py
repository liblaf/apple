from collections.abc import Sequence
from typing import TYPE_CHECKING, Self, override

import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, Float, Integer

from liblaf.apple import struct
from liblaf.apple.sim.core.region import Region

from ._abc import AbstractField, FieldLike

if TYPE_CHECKING:
    from ._grad import FieldGrad


class Field(AbstractField):
    _values: Float[jax.Array, " points *dim"] = struct.array(default=None)

    @classmethod
    def from_region(cls, region: Region, values: FieldLike | None = None) -> Self:
        if values is None:
            values = jnp.zeros((region.n_points, 1))
        self: Self = cls(_region=region)
        self = self.from_values(values)
        return self

    # region Structure

    @property
    def values(self) -> Float[jax.Array, " points *dim"]:
        return self._values

    # endregion Structure

    # region Shape

    @property
    @override
    def dim(self) -> Sequence[int]:
        return self.values.shape[1:]

    # endregion Shape

    # region Geometric Operations

    @property
    def boundary(self) -> "Field":
        region: Region = self.region.boundary
        return self.evolve(_region=region).with_values(
            self.values[self.region.original_point_id]
        )

    def extract_cells(
        self, ind: Integer[ArrayLike, " sub_cells"], *, invert: bool = False
    ) -> "Field":
        region: Region = self.region.extract_cells(ind=ind, invert=invert)
        return self.evolve(_region=region).with_values(
            self.values[self.region.original_point_id]
        )

    # endregion Geometric Operations

    # region Operators

    @property
    @override
    def deformation_gradient(self) -> "FieldGrad":
        from ._grad import FieldGrad

        return FieldGrad.from_region(
            region=self.region, values=self.region.deformation_gradient(self.values)
        )

    @property
    @override
    def grad(self) -> "FieldGrad":
        from ._grad import FieldGrad

        return FieldGrad.from_region(
            self.region, values=self.region.gradient(self.values)
        )

    # endregion Operators

    # region ArrayMixin

    @override
    def from_values(self, values: FieldLike, /) -> Self:
        values = jnp.asarray(values)
        values = jnp.broadcast_to(values, (self.region.n_points, *values.shape[1:]))
        return self.evolve(_values=values)

    # endregion ArrayMixin
