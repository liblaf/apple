import abc
from collections.abc import Sequence
from typing import Self, override

import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, Float, Integer

from liblaf.apple import struct
from liblaf.apple.sim.core.element import Element
from liblaf.apple.sim.core.geometry import Geometry
from liblaf.apple.sim.core.quadrature import Scheme
from liblaf.apple.sim.core.region import Region

type FieldLike = AbstractField | ArrayLike


class AbstractField(struct.PyTree, struct.ArrayMixin):
    _region: Region = struct.data(default=None)

    @property
    @abc.abstractmethod
    def values(self) -> jax.Array: ...

    # region Structure

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

    # endregion Structure

    # region Shape

    @property
    def dim(self) -> Sequence[int]:
        raise NotImplementedError

    @property
    def dtype(self) -> jnp.dtype:
        return self.values.dtype

    @property
    def n_cells(self) -> int:
        return self.region.n_cells

    @property
    def n_dof(self) -> int:
        return self.values.size

    @property
    def n_points(self) -> int:
        return self.region.n_points

    @property
    def shape(self) -> Sequence[int]:
        return self.values.shape

    @property
    def ndim(self) -> int:
        return self.values.ndim

    # endregion Shape

    # region Array

    @property
    def cells(self) -> Integer[jax.Array, "cells a"]:
        return self.region.cells

    @property
    def points(self) -> Float[jax.Array, "points J"]:
        return self.region.points

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
    def boundary(self) -> "AbstractField":
        raise NotImplementedError

    def extract_cells(
        self, ind: Integer[ArrayLike, " sub_cells"], *, invert: bool = False
    ) -> "AbstractField":
        raise NotImplementedError

    # endregion Geometric Operations

    # region Operators

    @property
    def deformation_gradient(self) -> "AbstractField":
        raise NotImplementedError

    @property
    def integral(self) -> Float[jax.Array, "*dim"]:
        raise NotImplementedError

    @property
    def grad(self) -> "AbstractField":
        raise NotImplementedError

    # endregion Operators

    # region ArrayMixin

    @override
    def __jax_array__(self) -> jax.Array:
        return self.values

    # endregion ArrayMixin

    def with_values(self, values: FieldLike | None = None, /) -> Self:
        if values is None:
            return self
        values = jnp.asarray(values, dtype=self.dtype)
        values = jnp.broadcast_to(values, self.shape)
        return self.evolve(_values=values)
