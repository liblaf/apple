from typing import Self

import einops
import flax.struct
import jax
import jax.numpy as jnp
import pyvista as pv
from jaxtyping import ArrayLike, Float, Integer

from liblaf.apple import math
from liblaf.apple.sim import domain as _d
from liblaf.apple.sim import function_space as _s
from liblaf.apple.sim import geometry as _g


class Field(flax.struct.PyTreeNode, frozen=False):
    dim: int = flax.struct.field(pytree_node=False, default=3)
    space: _s.FunctionSpace = flax.struct.field(default=None)
    _values: Float[jax.Array, "points dim"] = flax.struct.field(default=None)

    @classmethod
    def from_space(
        cls,
        space: _s.FunctionSpace,
        values: Float[jax.Array, "points dim"],
        *,
        dim: int = 3,
    ) -> Self:
        return cls(dim=dim, space=space, _values=values)

    # region Inherited

    @property
    def area(self) -> Float[jax.Array, " cells"]:
        return self.space.area

    @property
    def cells(self) -> Integer[jax.Array, "cells ..."]:
        return self.space.cells

    @property
    def domain(self) -> _d.Domain:
        return self.space.domain

    @property
    def geometry(self) -> _g.Geometry:
        return self.space.geometry

    @property
    def mesh(self) -> pv.DataSet:
        return self.space.mesh

    @property
    def n_cells(self) -> int:
        return self.space.n_cells

    @property
    def n_dof(self) -> int:
        return self.n_points * self.dim

    @property
    def n_points(self) -> int:
        return self.space.n_points

    @property
    def points(self) -> Float[jax.Array, "points 3"]:
        return self.space.points

    @property
    def volume(self) -> Float[jax.Array, " cells"]:
        return self.space.volume

    @property
    def w(self) -> Float[jax.Array, ""]:
        return self.space.w

    @property
    def h(self) -> Float[jax.Array, " a"]:
        return self.space.h

    @property
    def dh_dr(self) -> Float[jax.Array, "a J=3"]:
        return self.space.dh_dr

    @property
    def dX_dr(self) -> Float[jax.Array, "c I=3 J=3"]:
        return self.space.dX_dr

    @property
    def dr_dX(self) -> Float[jax.Array, "c I=3 J=3"]:
        return self.space.dr_dX

    @property
    def dV(self) -> Float[jax.Array, " c"]:
        return self.space.dV

    @property
    def dh_dX(self) -> Float[jax.Array, "c a J=3"]:
        return self.space.dh_dX

    # endregion Inherited

    @property
    def values(self) -> Float[jax.Array, "points dim"]:
        return self._values

    @values.setter
    def values(self, values: ArrayLike) -> None:
        values = jnp.asarray(values)
        values = math.broadcast_to(
            values, self.values.shape, math.BroadcastMode.LEADING
        )
        self._values = values

    @property
    def grad(self) -> Float[jax.Array, "cells dim J=3"]:
        return einops.einsum(
            self.values[self.cells], self.dh_dX, "c a dim, c a J -> c dim J"
        )

    @property
    def deformation_gradient(self) -> Float[jax.Array, "cells 3 3"]:
        return self.grad + jnp.identity(3)

    def deformation_gradient_jvp(self, p: Self) -> Float[jax.Array, "cells 3 3"]:
        return einops.einsum(
            p.values[self.cells], self.dh_dX, "c a dim, c a J -> c dim J"
        )

    def deformation_gradient_vjp(
        self, p: Float[jax.Array, "cells 3 3"]
    ) -> Float[jax.Array, "cells a dim"]:
        p = jnp.asarray(p)
        return einops.einsum(self.dh_dX, p, "c a J, c dim J -> c a dim")
