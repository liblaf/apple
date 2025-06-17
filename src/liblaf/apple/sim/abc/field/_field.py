from collections.abc import Sequence
from typing import TYPE_CHECKING, Self, override

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike, DTypeLike, Float, Integer

from liblaf import grapes
from liblaf.apple import math, struct
from liblaf.apple.sim.abc.element import Element
from liblaf.apple.sim.abc.geometry import Geometry
from liblaf.apple.sim.abc.quadrature import Scheme
from liblaf.apple.sim.abc.region import Region

if TYPE_CHECKING:
    from ._grad import FieldGrad


type FieldLike = Float[ArrayLike, " points *dim"] | "Field"


def broadcast_to(
    values: FieldLike, *, n_points: int
) -> Float[jax.Array, " points *dim"]:
    values = jnp.asarray(values)
    if values.ndim == 0:
        return jnp.broadcast_to(values, (n_points, 1))
    if values.shape[0] != n_points:
        values = jnp.broadcast_to(values, (n_points, *values.shape))
    if values.ndim == 1:
        values = jnp.expand_dims(values, axis=-1)
    return values


class Field(struct.ArrayMixin, struct.PyTree):
    _region: Region = struct.field(default=None)
    _shape_dtype: jax.ShapeDtypeStruct = struct.static(
        default=jax.ShapeDtypeStruct(shape=(1,), dtype=float)
    )
    _values: Float[jax.Array, "points *dim"] = struct.array(default=None)

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
            values = math.broadcast_to(values, (region.n_points, 1))
        if values.shape[0] != region.n_points:
            values = math.broadcast_to(values, (region.n_points, *values.shape))
        if values.ndim == 1:
            values = jnp.expand_dims(values, axis=-1)
        if dim is not None:
            dim = grapes.as_sequence(dim)
            values = math.broadcast_to(values, (region.n_points, *dim))
        if dtype is not None:
            values = jnp.asarray(values, dtype=dtype)
        return cls(
            _region=region,
            _shape_dtype=jax.ShapeDtypeStruct(values.shape[1:], values.dtype),
            _values=values,
        )

    def __jax_array__(self) -> Float[jax.Array, "points *dim"]:
        return self.values

    # region Underlying

    @property
    def element(self) -> Element:
        return self.region.element

    @property
    def geometry(self) -> Geometry:
        return self.region.geometry

    @property
    def quadrature(self) -> Scheme:
        return self.region.quadrature

    @property
    def region(self) -> Region:
        return jax.lax.stop_gradient(self._region)

    # endregion Underlying

    # region Shape

    @property
    def dim(self) -> Sequence[int]:
        return self._shape_dtype.shape

    @property
    def dtype(self) -> jnp.dtype:
        return self._shape_dtype.dtype

    @property
    def n_cells(self) -> int:
        return self.region.n_cells

    @property
    def n_dof(self) -> int:
        return int(np.prod(self.shape))

    @property
    def n_points(self) -> int:
        return self.region.n_points

    @property
    def shape(self) -> Sequence[int]:
        return (self.n_points, *self.dim)

    # endregion Shape

    # region Array

    @property
    def cells(self) -> Integer[jax.Array, "cells a"]:
        return self.region.cells

    @property
    def points(self) -> Float[jax.Array, "points J"]:
        return self.region.points

    @property
    def values(self) -> Float[jax.Array, "points *dim"]:
        return self._values

    # endregion Array

    # region Function Space

    @property
    def h(self) -> Float[jax.Array, "q a"]:
        return self.region.h

    @property
    def dhdr(self) -> Float[jax.Array, "q a J"]:
        return self.region.dhdr

    @property
    def dXdr(self) -> Float[jax.Array, "c q I J"]:
        return self.region.dXdr

    @property
    def drdX(self) -> Float[jax.Array, "c q J I"]:
        return self.region.drdX

    @property
    def dV(self) -> Float[jax.Array, "c q"]:
        return self.region.dV

    @property
    def dhdX(self) -> Float[jax.Array, "c q a J"]:
        return self.region.dhdX

    # endregion Function Space

    # region Geometric Operations

    @property
    def boundary(self) -> "Field":
        region: Region = self.region.boundary
        return self.replace(_region=region).with_values(
            self.values[self.region.original_point_id]
        )

    def extract_cells(
        self, ind: Integer[ArrayLike, " sub_cells"], *, invert: bool = False
    ) -> "Field":
        region: Region = self.region.extract_cells(ind=ind, invert=invert)
        return self.replace(_region=region).with_values(
            self.values[self.region.original_point_id]
        )

    # endregion Geometric Operations

    # region Operators

    @property
    def deformation_gradient(self) -> "FieldGrad":
        from ._grad import FieldGrad

        return FieldGrad.from_region(
            region=self.region,
            values=self.region.deformation_gradient(self.values),
            dim=(*self.dim, self.geometry.dim),
            dtype=self.dtype,
        )

    @property
    def integral(self) -> Float[jax.Array, "*dim"]:
        return self.region.integrate(self.values)

    @property
    def grad(self) -> "FieldGrad":
        from ._grad import FieldGrad

        return FieldGrad.from_region(
            self.region,
            values=self.region.gradient(self.values),
            dim=(*self.dim, self.geometry.dim),
            dtype=self.dtype,
        )

    # endregion Operators

    @classmethod
    def broadcast(
        cls,
        region: Region,
        values: FieldLike,
        /,
        dim: int | Sequence[int] | None = None,
        dtype: DTypeLike | None = None,
    ) -> Float[jax.Array, " points *dim"]:
        values = jnp.asarray(values)
        if values.ndim == 0:
            values = math.broadcast_to(values, (region.n_points, 1))
        if values.shape[0] != region.n_points:
            values = math.broadcast_to(values, (region.n_points, *values.shape))
        if values.ndim == 1:
            values = jnp.expand_dims(values, axis=-1)
        if dim is not None:
            dim = grapes.as_sequence(dim)
            values = math.broadcast_to(values, (region.n_points, *dim))
        if dtype is not None:
            values = jnp.asarray(values, dtype=dtype)
        return values

    @override
    def from_values(self, values: ArrayLike, /) -> Self:
        values = jnp.asarray(values)
        chex.assert_shape(values, (self.n_points, ...))
        return self.replace(
            _shape_dtype=jax.ShapeDtypeStruct(values.shape[1:], dtype=values.dtype),
            _values=values,
        )

    def with_values(self, values: FieldLike | None = None, /) -> Self:
        if values is None:
            return self
        values = jnp.asarray(values, dtype=self.dtype)
        values = math.broadcast_to(values, self.shape)
        return self.replace(_values=values)
