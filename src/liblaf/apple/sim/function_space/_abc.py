from typing import Self

import einops
import jax
import jax.numpy as jnp
import pyvista as pv
from jaxtyping import Float, Integer

from liblaf.apple import struct, utils
from liblaf.apple.sim import domain as _d
from liblaf.apple.sim import geometry as _g


class FunctionSpace(struct.Node):
    domain: _d.Domain = struct.data(default=None)

    # cached properties
    w: Float[jax.Array, ""] = struct.data(default=None)
    h: Float[jax.Array, " a"] = struct.data(default=None)
    dh_dr: Float[jax.Array, "a J=3"] = struct.data(default=None)
    dX_dr: Float[jax.Array, "c I=3 J=3"] = struct.data(default=None)
    dr_dX: Float[jax.Array, "c I=3 J=3"] = struct.data(default=None)
    dV: Float[jax.Array, " c"] = struct.data(default=None)
    dh_dX: Float[jax.Array, "c a J=3"] = struct.data(default=None)

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

    # region Delegation

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
    def grad(self) -> "FunctionSpace":
        raise NotImplementedError

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

    # endregion Delegation

    # region FEM

    @utils.jit
    def scatter(
        self, values: Float[jax.Array, " points *dim"]
    ) -> Float[jax.Array, "cells a *dim"]:
        values = jnp.asarray(values)
        return values[self.cells]

    @utils.jit
    def gather(
        self, values: Float[jax.Array, "cells a *dim"]
    ) -> Float[jax.Array, " points *dim"]:
        return jax.ops.segment_sum(
            einops.rearrange(
                values, "cells points_per_cell ... -> (cells points_per_cell) ..."
            ),
            einops.rearrange(
                self.cells, "cells points_per_cell -> (cells points_per_cell)"
            ),
            num_segments=self.n_points,
        )

    def with_dX_dr(self) -> Self:
        if self.dX_dr is not None:
            return self
        if self.dh_dr is None:
            return self
        dX_dr: Float[jax.Array, "c I=3 J=3"] = einops.einsum(
            self.points[self.cells], self.dh_dr, "c a I, a J -> c I J"
        )
        return self.evolve(dX_dr=dX_dr)

    def with_dr_dX(self) -> Self:
        if self.dr_dX is not None:
            return self
        new: Self = self.with_dX_dr()
        if new.dX_dr is None:
            return new
        dr_dX: Float[jax.Array, "c I=3 J=3"] = jnp.linalg.inv(new.dX_dr)
        return new.evolve(dr_dX=dr_dX)

    def with_dV(self) -> Self:
        if self.dV is not None:
            return self
        new: Self = self.with_dX_dr()
        if new.dX_dr is None or new.w is None:
            return new
        dV: Float[jax.Array, " c"] = jnp.linalg.det(new.dX_dr) * new.w
        return new.evolve(dV=dV)

    def with_dh_dX(self) -> Self:
        if self.dh_dX is not None:
            return self
        new: Self = self.with_dr_dX()
        if new.dh_dr is None or new.dr_dX is None:
            return new
        dh_dX: Float[jax.Array, "c a J=3"] = einops.einsum(
            new.dh_dr, new.dr_dX, "a I, c I J -> c a J"
        )
        return new.evolve(dh_dX=dh_dX)

    # endregion FEM
