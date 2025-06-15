import math
from collections.abc import Sequence
from typing import Self

import jax
import jax.numpy as jnp
from jaxtyping import Float, Integer
from numpy.typing import ArrayLike

from liblaf.apple import struct
from liblaf.apple.sim import element as _e
from liblaf.apple.sim import geometry as _g
from liblaf.apple.sim import quadrature as _q
from liblaf.apple.sim import region as _r


class Field(struct.ArrayMixin, struct.Node):
    def __jax_array__(self) -> Float[jax.Array, "points *dim"]:
        return self.values

    # region Underlying

    @property
    def element(self) -> _e.Element:
        return self.region.element

    @property
    def geometry(self) -> _g.Geometry:
        return self.region.geometry

    @property
    def quadrature(self) -> _q.Scheme:
        return self.region.quadrature

    @property
    def region(self) -> _r.Region:
        raise NotImplementedError

    # endregion Underlying

    # region Shape

    @property
    def dim(self) -> Sequence[int]:
        raise NotImplementedError

    @property
    def dtype(self) -> jnp.dtype:
        raise NotImplementedError

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
        raise NotImplementedError

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

    def extract(
        self, ind: Integer[ArrayLike, " sub_cells"], *, invert: bool = False
    ) -> "Field":
        raise NotImplementedError

    def warp(self, displacement: Float[ArrayLike, "points *dim"] | None) -> _g.Geometry:
        raise NotImplementedError

    # endregion Geometric Operations

    # region Operators

    @property
    def grad(self) -> "Field":
        raise NotImplementedError

    # endregion Operators

    def with_values(self, values: Float[ArrayLike, "points *dim"]) -> Self:
        raise NotImplementedError
