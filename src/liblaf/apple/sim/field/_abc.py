import functools
import math
import operator
from collections.abc import Sequence
from typing import Self

import einops
import jax
import jax.numpy as jnp
import pyvista as pv
from jaxtyping import Float, Integer
from numpy.typing import ArrayLike

from liblaf import grapes
from liblaf.apple import struct
from liblaf.apple.sim import domain as _d
from liblaf.apple.sim import function_space as _s
from liblaf.apple.sim import geometry as _g


class Field(struct.Node):
    dim: Sequence[int] = struct.static(default=(3,), converter=grapes.as_sequence)
    space: _s.FunctionSpace = struct.data(default=None)
    values: Float[jax.Array, "points *dim"] = struct.data(default=None)

    @classmethod
    def from_space(
        cls,
        space: _s.FunctionSpace,
        values: Float[ArrayLike, "points *dim"] = 0.0,
        *,
        dim: int | Sequence[int] = (3,),
    ) -> Self:
        self: Self = cls(dim=dim, space=space)
        self = self.with_values(values=values)
        return self

    def __jax_array__(self) -> Float[jax.Array, "points *dim"]:
        return self.values

    def _op(self, op: str, /, *args, **kwargs) -> Self:
        values: jax.Array = getattr(operator, op)(self.values, *args, **kwargs)
        return self.evolve(values=values)

    __add__ = functools.partialmethod(_op, "__add__")
    __sub__ = functools.partialmethod(_op, "__sub__")
    __mul__ = functools.partialmethod(_op, "__mul__")
    __matmul__ = functools.partialmethod(_op, "__matmul__")
    __truediv__ = functools.partialmethod(_op, "__truediv__")
    __floordiv__ = functools.partialmethod(_op, "__floordiv__")
    __mod__ = functools.partialmethod(_op, "__mod__")
    __pow__ = functools.partialmethod(_op, "__pow__")
    __lshift__ = functools.partialmethod(_op, "__lshift__")
    __rshift__ = functools.partialmethod(_op, "__rshift__")
    __and__ = functools.partialmethod(_op, "__and__")
    __xor__ = functools.partialmethod(_op, "__xor__")
    __or__ = functools.partialmethod(_op, "__or__")

    __neg__ = functools.partialmethod(_op, "__neg__")
    __pos__ = functools.partialmethod(_op, "__pos__")
    __abs__ = functools.partialmethod(_op, "__abs__")
    __invert__ = functools.partialmethod(_op, "__invert__")

    # region Delegation

    @property
    def area(self) -> Float[jax.Array, " cells"]:
        return self.space.area

    @property
    def boundary(self) -> "Field":
        raise NotImplementedError

    @property
    def cells(self) -> Integer[jax.Array, "cells a"]:
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
        return self.n_points * math.prod(self.dim)

    @property
    def n_points(self) -> int:
        return self.space.n_points

    @property
    def points(self) -> Float[jax.Array, "points 3"]:
        return self.space.points

    @property
    def values_scatter(self) -> Float[jax.Array, "cells a *dim"]:
        return self.space.scatter(self.values)

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

    # endregion Delegation

    @property
    def deformation_gradient(self) -> "Field":
        return self.grad + jnp.identity(3)

    @property
    def grad(self) -> "Field":
        values: Float[jax.Array, " cells *dim"] = einops.einsum(
            self.values_scatter, self.dh_dX, "c a ..., c a J -> c ... J"
        )
        return Field.from_space(space=self.space.grad, values=values)

    @property
    def integration(self) -> Float[jax.Array, "*dim"]:
        # TODO: make this more general
        return einops.einsum(self.values_scatter, self.dV, "c ..., c -> ...")

    def deformation_gradient_jvp(self, p: Self) -> "Field":
        values: Float[jax.Array, "cells 3 3"] = einops.einsum(
            p.values_scatter, self.dh_dX, "c a dim, c a J -> c dim J"
        )
        return Field.from_space(space=self.space.grad, values=values)

    def deformation_gradient_vjp(self, p: "Field") -> Self:
        values: Float[jax.Array, "cells a dim=3"] = einops.einsum(
            self.dh_dX, p.values, "c a J, c dim J -> c a dim"
        )
        return self.evolve(values=self.space.gather(values=values))

    def with_values(self, values: Float[ArrayLike, " points *dim"]) -> Self:
        values: Float[jax.Array, " points *dim"] = jnp.broadcast_to(
            jnp.asarray(values), (self.n_points, *self.dim)
        )
        return self.evolve(values=values)
