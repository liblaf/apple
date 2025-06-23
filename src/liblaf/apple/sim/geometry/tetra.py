from typing import override

import numpy as np
import pyvista as pv

from .geometry import Geometry
from .triangle import GeometryTriangle


class GeometryTetra(Geometry):
    @property
    @override
    def boundary(self) -> GeometryTriangle:
        mesh: pv.UnstructuredGrid = self.to_pyvista(attributes=True)
        mesh.cell_data["cell-id"] = np.arange(mesh.n_cells)
        mesh.point_data["point-id"] = np.arange(mesh.n_points)
        surface: pv.PolyData = mesh.extract_surface()
        result: GeometryTriangle = GeometryTriangle.from_pyvista(surface)
        return result
