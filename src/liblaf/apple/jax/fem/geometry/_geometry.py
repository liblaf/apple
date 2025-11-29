from __future__ import annotations

import jax.numpy as jnp
import pyvista as pv
from jaxtyping import Array, Float, Integer
from liblaf.peach import tree

from liblaf.apple.constants import POINT_ID
from liblaf.apple.jax.fem.element import Element


@tree.define
class Geometry:
    mesh: pv.DataSet = tree.field()

    @classmethod
    def from_pyvista(cls, mesh: pv.DataObject) -> Geometry:
        from ._tetra import GeometryTetra
        from ._triangle import GeometryTriangle

        if isinstance(mesh, pv.PolyData):
            return GeometryTriangle.from_pyvista(mesh)
        if isinstance(mesh, pv.UnstructuredGrid):
            return GeometryTetra.from_pyvista(mesh)
        raise NotImplementedError

    @property
    def element(self) -> Element:
        raise NotImplementedError

    @property
    def n_cells(self) -> int:
        return self.cells.shape[0]

    @property
    def cell_data(self) -> pv.DataSetAttributes:
        return self.mesh.cell_data

    @property
    def cells(self) -> Integer[Array, "c a"]:
        raise NotImplementedError

    @property
    def cells_global(self) -> Integer[Array, "c a"]:
        return self.point_id[self.cells]

    @property
    def point_data(self) -> pv.DataSetAttributes:
        return self.mesh.point_data

    @property
    def point_id(self) -> Integer[Array, "p J"]:
        return jnp.asarray(self.point_data[POINT_ID])

    @property
    def points(self) -> Float[Array, "p J"]:
        return jnp.asarray(self.mesh.points)
