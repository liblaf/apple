from typing import Self

import einops
import flax.struct
import jax
import jax.numpy as jnp
import pyvista as pv
from jaxtyping import Float, Integer

from liblaf.apple.sim import domain as _d
from liblaf.apple.sim import geometry as _g


class FunctionSpace(flax.struct.PyTreeNode):
    domain: _d.Domain = flax.struct.field(default=None)

    # cached properties
    w: Float[jax.Array, ""] = flax.struct.field(default=None)
    h: Float[jax.Array, " a"] = flax.struct.field(default=None)
    dh_dr: Float[jax.Array, "a J=3"] = flax.struct.field(default=None)
    dX_dr: Float[jax.Array, "c I=3 J=3"] = flax.struct.field(default=None)
    dr_dX: Float[jax.Array, "c I=3 J=3"] = flax.struct.field(default=None)
    dV: Float[jax.Array, " c"] = flax.struct.field(default=None)
    dh_dX: Float[jax.Array, "c a J=3"] = flax.struct.field(default=None)

    @classmethod
    def from_domain(
        cls,
        domain: _d.Domain,
        *,
        dX_dr: bool = True,
        dr_dX: bool = True,
        dV: bool = True,
        dh_dX: bool = True,
    ) -> Self:
        self: Self = cls(domain=domain)
        if dX_dr:
            self = self.with_dX_dr()
        if dr_dX:
            self = self.with_dr_dX()
        if dV:
            self = self.with_dV()
        if dh_dX:
            self = self.with_dh_dX()
        return self

    # region Inherited

    @property
    def area(self) -> Float[jax.Array, " cells"]:
        return self.domain.area

    @property
    def cells(self) -> Integer[jax.Array, "cells ..."]:
        return self.domain.cells

    @property
    def geometry(self) -> _g.Geometry:
        return self.domain.geometry

    @property
    def mesh(self) -> pv.DataSet:
        return self.domain.mesh

    @property
    def n_cells(self) -> int:
        return self.domain.n_cells

    @property
    def n_points(self) -> int:
        return self.domain.n_points

    @property
    def points(self) -> Float[jax.Array, "points 3"]:
        return self.domain.points

    @property
    def volume(self) -> Float[jax.Array, " cells"]:
        return self.domain.volume

    # endregion Inherited

    # region FEM

    def with_dX_dr(self) -> Self:
        if self.dX_dr is not None:
            return self
        dX_dr: Float[jax.Array, "c I=3 J=3"] = einops.einsum(
            self.points[self.cells], self.dh_dr, "c a I, a J -> c I J"
        )
        return self.replace(dX_dr=dX_dr)

    def with_dr_dX(self) -> Self:
        if self.dr_dX is not None:
            return self
        new: Self = self.with_dX_dr()
        dr_dX: Float[jax.Array, "c I=3 J=3"] = jnp.linalg.inv(new.dX_dr)
        return new.replace(dr_dX=dr_dX)

    def with_dV(self) -> Self:
        if self.dV is not None:
            return self
        new: Self = self.with_dX_dr()
        dV: Float[jax.Array, " c"] = jnp.linalg.det(new.dX_dr) * new.w
        return new.replace(dV=dV)

    def with_dh_dX(self) -> Self:
        if self.dh_dX is not None:
            return self
        new: Self = self.with_dr_dX()
        dh_dX: Float[jax.Array, "c a J=3"] = einops.einsum(
            new.dh_dr, new.dr_dX, "a I, c I J -> c a J"
        )
        return new.replace(dh_dX=dh_dX)

    # endregion FEM
