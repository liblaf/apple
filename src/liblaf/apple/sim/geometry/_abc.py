from typing import Self

import jax
import jax.numpy as jnp
import pyvista as pv
from jaxtyping import ArrayLike, Float, Integer

from liblaf.apple import struct
from liblaf.apple.sim import element as _e


class Geometry(struct.PyTree):
    _mesh: pv.DataSet = struct.static(default=None)

    @classmethod
    def from_pyvista(cls, mesh: pv.DataSet) -> Self:
        self: Self = cls(_mesh=mesh)
        return self

    # region Shape

    @property
    def dim(self) -> int:
        return self.points.shape[1]

    @property
    def n_cells(self) -> int:
        return self.mesh.n_cells

    @property
    def n_points(self) -> int:
        return self.mesh.n_points

    # endregion Shape

    @property
    def boundary(self) -> "Geometry":
        raise NotImplementedError

    @property
    def cells(self) -> Integer[jax.Array, "cells a"]:
        raise NotImplementedError

    @property
    def element(self) -> _e.Element:
        raise NotImplementedError

    @property
    def mesh(self) -> pv.DataSet:
        return self._mesh

    @property
    def points(self) -> Float[jax.Array, "points J"]:
        with jax.ensure_compile_time_eval():
            return jnp.asarray(self.mesh.points)

    def extract(
        self, ind: Integer[ArrayLike, " sub_cells"], *, invert: bool = False
    ) -> "Geometry":
        raise NotImplementedError

    def warp(self, displacement: Float[ArrayLike, "points J"]) -> "Geometry":
        mesh: pv.DataSet = self.mesh
        mesh.point_data["displacement"] = displacement
        mesh = mesh.warp_by_vector("displacement", inplace=False)
        return self.evolve(_mesh=mesh)
