from collections.abc import MutableMapping
from typing import Self

import einops
import jax
import jax.numpy as jnp
from jaxtyping import Float, Integer
from numpy.typing import ArrayLike

from liblaf.apple import struct, utils
from liblaf.apple.sim import element as _e
from liblaf.apple.sim import geometry as _g
from liblaf.apple.sim import quadrature as _q


class Region(struct.Node):
    _geometry: _g.Geometry = struct.data(default=None)
    _quadrature: _q.Scheme = struct.data(default=None)

    # region Structure

    @property
    def element(self) -> _e.Element:
        return self.geometry.element

    @property
    def geometry(self) -> _g.Geometry:
        return self._geometry

    @property
    def quadrature(self) -> _q.Scheme:
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
    def cell_data(self) -> _g.GeometryAttributes:
        return self.geometry.cell_data

    @property
    def field_data(self) -> _g.GeometryAttributes:
        return self.geometry.field_data

    @property
    def point_data(self) -> _g.GeometryAttributes:
        return self.geometry.point_data

    @property
    def user_dict(self) -> MutableMapping:
        return self.geometry.user_dict

    # endregion Attributes

    # region Function Space

    @property
    @utils.jit
    def h(self) -> Float[jax.Array, "q a"]:
        with jax.ensure_compile_time_eval():
            return jnp.asarray(
                [self.element.function(q) for q in self.quadrature.points]
            )

    @property
    @utils.jit
    def dhdr(self) -> Float[jax.Array, "q a J"]:
        with jax.ensure_compile_time_eval():
            return jnp.asarray(
                [self.element.gradient(q) for q in self.quadrature.points]
            )

    @property
    @utils.jit
    def dXdr(self) -> Float[jax.Array, "c q I J"]:
        with jax.ensure_compile_time_eval():
            return einops.einsum(
                self.points[self.cells], self.dhdr, "c a I, q a J -> c q I J"
            )

    @property
    @utils.jit
    def drdX(self) -> Float[jax.Array, "c q J I"]:
        with jax.ensure_compile_time_eval():
            return jnp.linalg.inv(self.dXdr)

    @property
    @utils.jit
    def dV(self) -> Float[jax.Array, "c q"]:
        with jax.ensure_compile_time_eval():
            return jnp.linalg.det(self.dXdr) * self.quadrature.weights[None, :]

    @property
    @utils.jit
    def dhdX(self) -> Float[jax.Array, "c q a J"]:
        with jax.ensure_compile_time_eval():
            return einops.einsum(self.dhdr, self.drdX, "q a I, c q I J -> c q a J")

    # endregion Function Space

    # region Geometric Operations

    @property
    def boundary(self) -> "Region":
        from ._boundary import RegionBoundary

        return RegionBoundary.from_region(self)

    def extract(
        self, ind: Integer[ArrayLike, " sub_cells"], *, invert: bool = False
    ) -> "Region":
        raise NotImplementedError

    def warp(self, displacement: Float[ArrayLike, " points J"]) -> Self:
        raise NotImplementedError

    # endregion Geometric Operations

    # region Operators

    def deformation_gradient(
        self, values: Float[ArrayLike, "points dim"]
    ) -> Float[jax.Array, "cells q dim J"]:
        return self.gradient(values) + jnp.identity(self.element.dim)[None, None, :, :]

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
