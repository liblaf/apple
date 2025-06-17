from collections.abc import MutableMapping
from typing import Self

import einops
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, Float, Integer

from liblaf.apple import struct, utils

from .element import Element
from .geometry import Geometry, GeometryAttributes
from .quadrature import Scheme


class Region(struct.PyTree):
    _geometry: Geometry = struct.field(default=None)
    _quadrature: Scheme = struct.field(default=None)

    _h: Float[jax.Array, "q a"] = struct.array(default=None)
    _dhdr: Float[jax.Array, "q a J"] = struct.array(default=None)
    _dXdr: Float[jax.Array, "c q I J"] = struct.array(default=None)
    _drdX: Float[jax.Array, "c q J I"] = struct.array(default=None)
    _dV: Float[jax.Array, "c q"] = struct.array(default=None)
    _dhdX: Float[jax.Array, "c q a J"] = struct.array(default=None)

    @classmethod
    def from_geometry(
        cls, geometry: Geometry, quadrature: Scheme | None = None, *, grad: bool = True
    ) -> Self:
        if quadrature is None:
            quadrature = geometry.quadrature
        self: Self = cls(_geometry=geometry, _quadrature=quadrature)
        if grad:
            self = self.with_grad()
        return self

    # region Structure

    @property
    def element(self) -> Element:
        return self.geometry.element

    @property
    def geometry(self) -> Geometry:
        return self._geometry

    @property
    def quadrature(self) -> Scheme:
        return self._quadrature

    # endregion Structure

    # region Shape

    @property
    def dim(self) -> int:
        return self.geometry.dim

    @property
    def n_cells(self) -> int:
        return self.geometry.n_cells

    @property
    def n_points(self) -> int:
        return self.geometry.n_points

    # endregion Shape

    # region Array

    @property
    def cells(self) -> Integer[jax.Array, "cells a"]:
        return self.geometry.cells

    @property
    def points(self) -> Float[jax.Array, "points J"]:
        return self.geometry.points

    # endregion Array

    # region Attributes

    @property
    def cell_data(self) -> GeometryAttributes:
        return self.geometry.cell_data

    @property
    def field_data(self) -> GeometryAttributes:
        return self.geometry.field_data

    @property
    def original_cell_id(self) -> Integer[jax.Array, "cells"]:
        return self.geometry.original_cell_id

    @property
    def original_point_id(self) -> Integer[jax.Array, "points"]:
        return self.geometry.original_point_id

    @property
    def point_data(self) -> GeometryAttributes:
        return self.geometry.point_data

    @property
    def user_dict(self) -> MutableMapping:
        return self.geometry.user_dict

    # endregion Attributes

    # region Function Space

    @property
    def h(self) -> Float[jax.Array, "q a"]:
        return self._h

    @property
    def dhdr(self) -> Float[jax.Array, "q a J"]:
        return self._dhdr

    @property
    def dXdr(self) -> Float[jax.Array, "c q I J"]:
        return self._dXdr

    @property
    def drdX(self) -> Float[jax.Array, "c q J I"]:
        return self._drdX

    @property
    def dV(self) -> Float[jax.Array, "c q"]:
        return self._dV

    @property
    def dhdX(self) -> Float[jax.Array, "c q a J"]:
        return self._dhdX

    @utils.jit
    def with_grad(self) -> Self:
        h: Float[jax.Array, "q a"] = jnp.asarray(
            [self.element.function(q) for q in self.quadrature.points]
        )
        dhdr: Float[jax.Array, "q a J"] = jnp.asarray(
            [self.element.gradient(q) for q in self.quadrature.points]
        )
        dXdr: Float[jax.Array, "c q I J"] = einops.einsum(
            self.scatter(self.points), dhdr, "c a I, q a J -> c q I J"
        )
        drdX: Float[jax.Array, "c q J I"] = jnp.linalg.inv(dXdr)
        dV: Float[jax.Array, "c q"] = (
            jnp.linalg.det(dXdr) * self.quadrature.weights[None, :]
        )
        dhdX: Float[jax.Array, "c q a J"] = einops.einsum(
            dhdr, drdX, "q a I, c q I J -> c q a J"
        )
        return self.replace(
            _h=h, _dhdr=dhdr, _dXdr=dXdr, _drdX=drdX, _dV=dV, _dhdX=dhdX
        )

    # endregion Function Space

    # region Geometric Operations

    @property
    def boundary(self) -> "Region":
        return self.replace(
            _geometry=self.geometry.boundary,
            _quadrature=None,
            _h=None,
            _dhdr=None,
            _dXdr=None,
            _drdX=None,
            _dV=None,
            _dhdX=None,
        )

    def extract_cells(
        self, ind: Integer[ArrayLike, " sub_cells"], *, invert: bool = False
    ) -> Self:
        raise NotImplementedError

    # endregion Geometric Operations

    # region Operators

    def deformation_gradient(
        self, values: Float[ArrayLike, "points dim"]
    ) -> Float[jax.Array, "cells q dim J"]:
        return self.gradient(values) + jnp.identity(self.dim)[None, None, :, :]

    def gather(
        self, values: Float[ArrayLike, "cells a *dim"]
    ) -> Float[jax.Array, "points *dim"]:
        values = jnp.asarray(values)
        return jax.ops.segment_sum(
            einops.rearrange(values, "c a ... -> (c a) ..."),
            einops.rearrange(self.cells, "c a -> (c a)"),
            num_segments=self.n_points,
        )

    def gradient(
        self, values: Float[ArrayLike, "points *dim"]
    ) -> Float[jax.Array, "cells q *dim J"]:
        return einops.einsum(
            self.scatter(values), self.dhdX, "c a ..., c q a J -> c q ... J"
        )

    def integrate(
        self, values: Float[ArrayLike, "cells q *dim"]
    ) -> Float[jax.Array, "*dim"]:
        values = jnp.asarray(values)
        return einops.einsum(values, self.dV, "c q ... , c q -> ...")

    def scatter(
        self, values: Float[ArrayLike, "points *dim"]
    ) -> Float[jax.Array, "cells a *dim"]:
        values = jnp.asarray(values)
        return values[self.cells]

    # endregion Operators
