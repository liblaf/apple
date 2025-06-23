from typing import Self, override

import jax.numpy as jnp
import numpy as np
import pyvista as pv

from .geometry import Geometry


class GeometryTriangle(Geometry):
    @classmethod
    def from_pyvista(cls, mesh: pv.PolyData) -> Self:
        self: Self = (
            cls(points=jnp.asarray(mesh.points), cells=jnp.asarray(mesh.regular_faces))
            .update_cell_data(mesh.cell_data)
            .update_point_data(mesh.point_data)
        )
        return self

    @override
    def to_pyvista(self, *, attributes: bool = True) -> pv.PolyData:
        mesh: pv.PolyData = pv.PolyData.from_regular_faces(
            np.asarray(self.points), np.asarray(self.cells)
        )
        if attributes:
            mesh.cell_data.update(self.cell_data)
            mesh.point_data.update(self.point_data)
        return mesh
