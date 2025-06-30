from typing import Self, override

import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv

from liblaf.apple import struct
from liblaf.apple.sim.element import Element, ElementTetra

from .geometry import Geometry
from .triangle import GeometryTriangle


@struct.pytree
class GeometryTetra(Geometry):
    @classmethod
    def from_pyvista(cls, mesh: pv.UnstructuredGrid) -> Self:
        return cls(
            points=jnp.asarray(mesh.points),
            cells=jnp.asarray(mesh.cells_dict[pv.CellType.TETRA]),
        ).copy_attributes(mesh)

    @property
    @override
    def element(self) -> Element:
        with jax.ensure_compile_time_eval():
            return ElementTetra()

    @override
    def boundary(self, *, attributes: bool = True) -> GeometryTriangle:
        mesh: pv.UnstructuredGrid = self.to_pyvista(attributes=attributes)
        mesh.cell_data["cell-id"] = np.arange(mesh.n_cells)
        mesh.point_data["point-id"] = np.arange(mesh.n_points)
        surface: pv.PolyData = mesh.extract_surface()
        result: GeometryTriangle = GeometryTriangle.from_pyvista(surface)
        return result

    @override
    def to_pyvista(self, *, attributes: bool = True) -> pv.DataSet:
        mesh: pv.UnstructuredGrid = pv.UnstructuredGrid(
            {pv.CellType.TETRA: np.asarray(self.cells)}, np.asarray(self.points)
        )
        if attributes:
            mesh.point_data.update(self.point_data)
            mesh.cell_data.update(self.cell_data)
        return mesh
