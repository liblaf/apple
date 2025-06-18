from typing import Self, override

import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv

from liblaf.apple.sim.core import Geometry
from liblaf.apple.sim.element import ElementTetra

from ._triangle import GeometryTriangle


class GeometryTetra(Geometry):
    @classmethod
    @override
    def from_pyvista(cls, mesh: pv.UnstructuredGrid, /) -> Self:
        self: Self = cls(
            points=jnp.asarray(mesh.points),
            cells=jnp.asarray(mesh.cells_dict[pv.CellType.TETRA]),
        )
        self = self.copy_attributes(mesh)
        return self

    @property
    @override
    def element(self) -> ElementTetra:
        with jax.ensure_compile_time_eval():
            return ElementTetra()

    @property
    @override
    def structure(self) -> pv.UnstructuredGrid:
        mesh = pv.UnstructuredGrid(
            {pv.CellType.TETRA: np.asarray(self.cells)}, np.asarray(self.points)
        )
        return mesh

    @property
    @override
    def boundary(self) -> GeometryTriangle:
        mesh: pv.UnstructuredGrid = self.structure
        mesh.point_data["point-id"] = np.arange(mesh.n_points)
        mesh.cell_data["cell-id"] = np.arange(mesh.n_cells)
        surface: pv.PolyData = mesh.extract_surface()
        geometry: GeometryTriangle = GeometryTriangle.from_pyvista(surface)
        geometry = geometry.copy_attributes(self)
        return geometry
