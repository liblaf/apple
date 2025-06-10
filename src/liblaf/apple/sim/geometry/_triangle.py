from typing import Self, override

import pyvista as pv

from liblaf.apple import struct

from ._abc import Geometry


class GeometryTriangle(Geometry):
    mesh: pv.PolyData = struct.static(default=None)

    @override
    def with_cells(self) -> Self:
        if self.cells is not None:
            return self
        return self.evolve(cells=self.mesh.regular_faces)
