from typing import Self

import attrs
import jax
import jax.numpy as jnp
import pyvista as pv
from jaxtyping import Float, Integer


@attrs.define
class Geometry:
    mesh: pv.UnstructuredGrid
    id: str = "geometry"

    @property
    def cells(self) -> Integer[jax.Array, "cells 4"]:
        return jnp.asarray(self.mesh.cells_dict[pv.CellType.TETRA], dtype=int)

    @property
    def length(self) -> float:
        return self.mesh.length

    @property
    def n_cells(self) -> int:
        return self.mesh.n_cells

    @property
    def n_points(self) -> int:
        return self.mesh.n_points

    @property
    def point_data(self) -> pv.DataSetAttributes:
        return self.mesh.point_data

    @property
    def cell_data(self) -> pv.DataSetAttributes:
        return self.mesh.cell_data

    @property
    def points(self) -> Float[jax.Array, "points 3"]:
        return jnp.asarray(self.mesh.points)

    def warp(self, displacements: Float[jax.Array, "points 3"]) -> Self:
        mesh: pv.UnstructuredGrid = self.mesh.copy()
        mesh.point_data["displacements"] = displacements
        mesh.warp_by_vector("displacements", inplace=True)
        return type(self)(mesh=mesh, id=self.id)
