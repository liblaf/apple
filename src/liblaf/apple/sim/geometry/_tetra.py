from typing import Self, override

import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv
from jaxtyping import Integer

from liblaf.apple import struct
from liblaf.apple.sim.abc import Geometry
from liblaf.apple.sim.element import ElementTetra

from ._triangle import GeometryTriangle


class GeometryTetra(Geometry):
    _pyvista: pv.UnstructuredGrid = struct.static(default=None)

    @classmethod
    @override
    def from_pyvista(cls, mesh: pv.UnstructuredGrid, /) -> Self:
        return cls(_pyvista=mesh)

    @property
    @override
    def element(self) -> ElementTetra:
        with jax.ensure_compile_time_eval():
            return ElementTetra()

    @property
    @override
    def pyvista(self) -> pv.UnstructuredGrid:
        return self._pyvista

    @property
    @override
    def cells(self) -> Integer[jax.Array, "cells a=4"]:
        with jax.ensure_compile_time_eval():
            return jnp.asarray(self.pyvista.cells_dict[pv.CellType.TETRA])

    @property
    @override
    def boundary(self) -> GeometryTriangle:
        mesh: pv.UnstructuredGrid = self.pyvista.copy()
        mesh.point_data["point-id"] = np.arange(mesh.n_points)
        mesh.cell_data["cell-id"] = np.arange(mesh.n_cells)
        surface: pv.PolyData = mesh.extract_surface()
        return GeometryTriangle.from_pyvista(surface)
