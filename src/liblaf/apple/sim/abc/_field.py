import math
from collections.abc import Sequence
from typing import Self, override

import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, DTypeLike, Float, Integer

from liblaf.apple import math as _m
from liblaf.apple import struct

from ._element import Element
from ._geometry import Geometry
from ._quadrature import Scheme
from ._region import Region


class Field(struct.ArrayMixin, struct.PyTree):
    _region: Region = struct.data(default=None)
    _shape_dtype: jax.ShapeDtypeStruct = struct.static(
        default=jax.ShapeDtypeStruct(shape=(3,), dtype=float)
    )
    _values: Float[jax.Array, "points *dim"] = struct.array(default=None)

    @classmethod
    def from_region(
        cls,
        region: Region,
        values: Float[ArrayLike, "points *dim"] = 0.0,
        *,
        dim: int | Sequence[int],
        dtype: DTypeLike = float,
    ) -> Self:
        self: Self = cls(
            _region=region, _shape_dtype=jax.ShapeDtypeStruct(shape=dim, dtype=dtype)
        )
        if values is not None:
            self = self.with_values(values)
        return self

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
        return self._region

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
        return self.n_points * math.prod(self.dim)

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
        raise NotImplementedError

    def extract_cells(
        self, ind: Integer[ArrayLike, " sub_cells"], *, invert: bool = False
    ) -> "Field":
        raise NotImplementedError

    def warp_by_vector(
        self, displacement: Float[ArrayLike, "points *dim"] | None
    ) -> Geometry:
        raise NotImplementedError

    # endregion Geometric Operations

    # region Operators

    @property
    def deformation_gradient(self) -> "FieldGrad":
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
        return FieldGrad.from_region(
            self.region,
            values=self.region.gradient(self.values),
            dim=(*self.dim, self.geometry.dim),
            dtype=self.dtype,
        )

    # endregion Operators

    def with_values(
        self, values: 'Float[ArrayLike, "points *dim"] | Field | None' = None, /
    ) -> Self:
        if values is None:
            return self
        values = jnp.asarray(values)
        values = _m.broadcast_to(values, self.shape)
        return self.evolve(_values=values)


class FieldGrad(Field):
    @property
    @override
    def shape(self) -> Sequence[int]:
        return (self.n_cells, self.quadrature.dim, *self.dim)
