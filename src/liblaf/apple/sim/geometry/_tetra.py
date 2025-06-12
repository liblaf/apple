from typing import Self, override

import jax
import numpy as np
import pyvista as pv
from jaxtyping import Integer

from liblaf.apple import struct

from ._abc import Geometry
from ._triangle import GeometryTriangle


class GeometryTetra(Geometry):
    mesh: pv.UnstructuredGrid = struct.static(default=None)

    @property
    def boundary(self) -> "GeometryTetraSurface":
        return GeometryTetraSurface.from_tetra(self)

    @override
    def with_cells(self) -> Self:
        if self.cells is not None:
            return self
        return self.evolve(cells=self.mesh.cells_dict[pv.CellType.TETRA])


class GeometryTetraSurface(GeometryTriangle):
    is_view: bool = struct.class_var(default=True, init=False)
    original_point_ids: Integer[jax.Array, " points"] = struct.array(default=None)

    @classmethod
    def from_tetra(cls, tetra: GeometryTetra) -> Self:
        mesh: pv.UnstructuredGrid = tetra.mesh
        mesh.point_data["point-id"] = np.arange(mesh.n_points)
        surface: pv.PolyData = mesh.extract_surface()
        self: Self = cls(
            refs=(tetra,),
            mesh=surface,
            original_point_ids=surface.point_data["point-id"],
        )
        self = self.with_cell_sizes().with_cells().with_points()
        return self
