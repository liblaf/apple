import math
from collections.abc import Sequence
from typing import Self, override

import einops
import jax
import jax.numpy as jnp
import pyvista as pv
from jaxtyping import DTypeLike, Float, Integer
from numpy.typing import ArrayLike

from liblaf.apple import math as _m
from liblaf.apple import struct
from liblaf.apple.sim import element as _e
from liblaf.apple.sim import geometry as _g
from liblaf.apple.sim import quadrature as _q
from liblaf.apple.sim import region as _r


class Field(struct.ArrayMixin, struct.Node):
    region: _r.Region = struct.data(default=None)
    shape_dtype: jax.ShapeDtypeStruct = struct.static(
        default=jax.ShapeDtypeStruct((3,), float)
    )
    _values: Float[jax.Array, " points *dim"] = struct.data(
        default=None, alias="_values"
    )

    @classmethod
    def from_region(
        cls,
        region: _r.Region,
        values: Float[ArrayLike, " points *dim"] = 0.0,
        *,
        dim: int | Sequence[int] = (3,),
        dtype: DTypeLike = float,
    ) -> Self:
        self: Self = cls(region=region, shape_dtype=jax.ShapeDtypeStruct(dim, dtype))
        self = self.with_values(values=values)
        return self

    def __jax_array__(self) -> Float[jax.Array, " points *dim"]:
        return self.values

    # region Delegation

    @property
    def area(self) -> Float[jax.Array, " cells"]:
        return self.region.area

    @property
    def boundary(self) -> "Field":
        raise NotImplementedError

    @property
    def cells(self) -> Integer[jax.Array, "cells a"]:
        return self.region.cells

    @property
    def dim(self) -> Sequence[int]:
        return self.shape_dtype.shape

    @property
    def dtype(self) -> jnp.dtype:
        return self.shape_dtype.dtype

    @property
    def element(self) -> _e.Element:
        return self.region.element

    @property
    def geometry(self) -> _g.Geometry:
        return self.region.geometry

    @property
    def mesh(self) -> pv.DataSet:
        return self.region.mesh

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
    def points(self) -> Float[jax.Array, "points 3"]:
        return self.region.points

    @property
    def quadrature(self) -> _q.Scheme:
        return self.region.quadrature

    @property
    def values(self) -> Float[jax.Array, " points *dim"]:
        return self._values

    @property
    def values_scatter(self) -> Float[jax.Array, "cells a *dim"]:
        return self.region.scatter(self.values)

    @property
    def volume(self) -> Float[jax.Array, " cells"]:
        return self.region.volume

    @property
    def weights(self) -> Float[jax.Array, " q"]:
        return self.region.weights

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

    # endregion Delegation

    @property
    def deformation_gradient(self) -> "FieldGrad":
        return self.grad + jnp.identity(3)

    @property
    def grad(self) -> "FieldGrad":
        return FieldGrad.from_region(
            region=self.region.grad,
            values=self.region.gradient(self.values),
            dim=(*self.dim, self.quadrature.dim),
            dtype=self.dtype,
        )

    @property
    def integration(self) -> Float[jax.Array, "*dim"]:
        return self.region.integrate(self.values)

    def deformation_gradient_jvp(self, p: Self) -> "FieldGrad":
        return FieldGrad.from_region(
            region=self.region.grad,
            values=self.region.gradient(p.values),
            dim=(*self.dim, self.quadrature.dim),
            dtype=self.dtype,
        )

    def deformation_gradient_vjp(self, p: "FieldGrad") -> Self:
        values: Float[jax.Array, "cells a *dim=3"] = einops.einsum(
            self.dhdX, p.values, "c q a J, c q ... J -> c a ..."
        )
        return self.evolve(values=self.region.gather(values))

    def with_values(self, values: Float[ArrayLike, " points *dim"]) -> Self:
        values: Float[jax.Array, " points *dim"] = jnp.broadcast_to(
            jnp.asarray(values), (self.n_points, *self.dim)
        )
        return self.evolve(_values=values)


class FieldGrad(Field):
    region: _r.RegionGrad = struct.data(default=None)

    @property
    def n_dof(self) -> int:
        return self.n_cells * self.quadrature.n_points * math.prod(self.dim)

    @override
    def with_values(self, values: Float[ArrayLike, " cells q *dim"]) -> Self:
        values: Float[jax.Array, " cells q *dim"] = _m.broadcast_to(
            jnp.asarray(values), (self.n_cells, self.quadrature.n_points, *self.dim)
        )
        return self.evolve(_values=values)
