from __future__ import annotations

import jarp
import jax.numpy as jnp
import pyvista as pv
from jaxtyping import Array, Float, Integer

from liblaf.apple.consts import GLOBAL_POINT_ID
from liblaf.apple.jax.fem.element import Element


@jarp.define
class Geometry:
    mesh: pv.DataSet = jarp.field()

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
        return self.cells_local.shape[0]

    @property
    def cell_data(self) -> pv.DataSetAttributes:
        return self.mesh.cell_data

    @property
    def cells_global(self) -> Integer[Array, "c a"]:
        return self.global_point_id[self.cells_local]

    @property
    def cells_local(self) -> Integer[Array, "c a"]:
        raise NotImplementedError

    @property
    def global_point_id(self) -> Integer[Array, "p J"]:
        return jnp.asarray(self.point_data[GLOBAL_POINT_ID])

    @property
    def point_data(self) -> pv.DataSetAttributes:
        return self.mesh.point_data

    @property
    def points(self) -> Float[Array, "p J"]:
        return jnp.asarray(self.mesh.points)
