from typing import Self

import jax
import pyvista as pv
from jaxtyping import Float, Integer
from numpy.typing import ArrayLike

from liblaf.apple import struct


class Geometry(struct.Node):
    mesh: pv.DataSet = struct.static(default=None)

    area: Float[jax.Array, " cells"] = struct.array(default=None)
    cells: Integer[jax.Array, "cells ..."] = struct.array(default=None)
    points: Float[jax.Array, "points 3"] = struct.array(default=None)
    volume: Float[jax.Array, " cells"] = struct.array(default=None)

    @classmethod
    def from_pyvista(
        cls,
        mesh: pv.DataSet,
        *,
        cell_sizes: bool = True,
        cells: bool = True,
        points: bool = True,
    ) -> Self:
        self: Self = cls(mesh=mesh)
        if cell_sizes:
            self = self.with_cell_sizes()
        if cells:
            self = self.with_cells()
        if points:
            self = self.with_points()
        return self

    @property
    def boundary(self) -> "Geometry":
        raise NotImplementedError

    @property
    def n_cells(self) -> int:
        return self.mesh.n_cells

    @property
    def n_points(self) -> int:
        return self.mesh.n_points

    def warp(self, x: Float[ArrayLike, " DoF"]) -> Self:
        mesh: pv.DataSet = self.mesh
        mesh.point_data["warp"] = x
        mesh = mesh.warp_by_vector("warp")
        return type(self).from_pyvista(mesh)

    def with_cell_sizes(self) -> Self:
        if self.area is not None and self.volume is not None:
            return self
        mesh: pv.DataSet = self.mesh.compute_cell_sizes()
        return self.evolve(
            mesh=mesh, area=mesh.cell_data["Area"], volume=mesh.cell_data["Volume"]
        )

    def with_cells(self) -> Self:
        if self.cells is not None:
            return self
        raise NotImplementedError

    def with_points(self) -> Self:
        if self.points is not None:
            return self
        return self.evolve(points=self.mesh.points)
