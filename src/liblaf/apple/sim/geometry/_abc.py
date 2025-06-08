from typing import Self

import flax.struct
import jax
import jax.numpy as jnp
import pyvista as pv
from jaxtyping import Float, Integer


class Geometry(flax.struct.PyTreeNode):
    mesh: pv.DataSet = flax.struct.field(pytree_node=False, default=None)

    # cached properties
    area: Float[jax.Array, " cells"] = flax.struct.field(default=None)
    cells: Integer[jax.Array, "cells ..."] = flax.struct.field(default=None)
    points: Float[jax.Array, "points 3"] = flax.struct.field(default=None)
    volume: Float[jax.Array, " cells"] = flax.struct.field(default=None)

    @classmethod
    def from_mesh(
        cls,
        mesh: pv.DataSet,
        *,
        area: bool = False,
        cells: bool = True,
        points: bool = True,
        volume: bool = False,
    ) -> Self:
        self: Self = cls(mesh=mesh)
        if area:
            self = self.with_area()
        if cells:
            self = self.with_cells()
        if points:
            self = self.with_points()
        if volume:
            self = self.with_volume()
        return self

    @property
    def n_cells(self) -> int:
        return self.mesh.n_cells

    @property
    def n_points(self) -> int:
        return self.mesh.n_points

    def with_area(self) -> Self:
        if self.area is not None:
            return self
        mesh: pv.DataSet = self.mesh.compute_cell_sizes()
        return self.replace(mesh=mesh, area=jnp.asarray(self.mesh.cell_data["Area"]))

    def with_cells(self) -> Self:
        if self.cells is not None:
            return self
        raise NotImplementedError

    def with_points(self) -> Self:
        if self.points is not None:
            return self
        return self.replace(points=jnp.asarray(self.mesh.points))

    def with_volume(self) -> Self:
        if self.volume is not None:
            return self
        mesh: pv.DataSet = self.mesh.compute_cell_sizes()
        return self.replace(
            mesh=mesh, volume=jnp.asarray(self.mesh.cell_data["Volume"])
        )
