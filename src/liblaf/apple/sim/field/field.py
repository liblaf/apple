from collections.abc import Sequence
from typing import Self

import chex
import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike, Float, Integer

from liblaf.apple import struct, utils
from liblaf.apple.sim.element import Element
from liblaf.apple.sim.geometry import Geometry
from liblaf.apple.sim.quadrature import Scheme
from liblaf.apple.sim.region import Region


@struct.pytree
class Field(struct.PyTreeMixin):
    region: Region = struct.data(default=None)
    values: Float[jax.Array, "points *dim"] = struct.array(default=None)

    @classmethod
    def from_region(cls, region: Region, values: ArrayLike) -> Self:
        values = jnp.asarray(values)
        values = jnp.broadcast_to(values, (region.n_points, *values.shape[1:]))
        chex.assert_shape(values, (region.n_points, ...))
        return cls(region=region, values=values)

    # region ArrayMixin

    def from_values(self, values: ArrayLike, /) -> Self:
        return self.evolve(values=values)

    # endregion ArrayMixin

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

    # endregion Structure

    # region Numbers

    @property
    def dim(self) -> Sequence[int]:
        return self.values.shape[1:]

    @property
    def n_cells(self) -> int:
        return self.region.n_cells

    @property
    def n_dofs(self) -> int:
        return self.values.size

    @property
    def n_points(self) -> int:
        return self.region.n_points

    # endregion Numbers

    # region Arrays

    @property
    def cells(self) -> Integer[jax.Array, "{self.n_cells} {self.element.n_points}"]:
        return self.region.cells

    @property
    def points(self) -> Integer[jax.Array, "{self.n_points} {self.region.dim}"]:
        return self.region.points

    @property
    @utils.validate
    def h(self) -> Float[jax.Array, "q a"]:
        return self.region.h

    @property
    @utils.validate
    def dhdr(self) -> Float[jax.Array, "q a J"]:
        return self.region.dhdr

    @property
    @utils.validate
    def dXdr(self) -> Float[jax.Array, "c q J J"]:
        return self.region.dXdr

    @property
    @utils.validate
    def drdX(self) -> Float[jax.Array, "c q J J"]:
        return self.region.drdX

    @property
    @utils.validate
    def dV(self) -> Float[jax.Array, "c q"]:
        return self.region.dV

    @property
    @utils.validate
    def dhdX(self) -> Float[jax.Array, "c q a J"]:
        return self.region.dhdX

    # endregion Arrays

    # region Operators

    @utils.jit_method(inline=True)
    def deformation_gradient(
        self,
    ) -> Float[
        jax.Array,
        "{self.n_cells} {self.quadrature.n_points} {self.region.dim} {self.region.dim}",
    ]:
        return self.region.deformation_gradient(self.values)

    @utils.jit_method(inline=True)
    def grad(
        self,
    ) -> Float[
        jax.Array, "{self.n_cells} {self.quadrature.n_points} *dim {self.region.dim}"
    ]:
        return self.region.gradient(self.values)

    # endregion Operators
