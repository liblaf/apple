from typing import Self, override

import jax
import jax.numpy as jnp
import pyvista as pv
from jaxtyping import Integer

from liblaf.apple import struct
from liblaf.apple.sim import element as _e

from ._abc import Geometry
from ._triangle import GeometryTriangle


class GeometryTetra(Geometry):
    _mesh: pv.UnstructuredGrid = struct.static(default=None)

    @classmethod
    def from_pyvista(cls, mesh: pv.UnstructuredGrid) -> Self:
        self: Self = cls(_mesh=mesh)
        return self

    @property
    @override
    def boundary(self) -> GeometryTriangle:
        surface: pv.PolyData = self.mesh.extract_surface()
        return GeometryTriangle.from_pyvista(surface)

    @property
    @override
    def cells(self) -> Integer[jax.Array, "cells a=4"]:
        with jax.ensure_compile_time_eval():
            return jnp.asarray(self.mesh.cells_dict[pv.CellType.TETRA])

    @property
    @override
    def element(self) -> _e.ElementTetra:
        with jax.ensure_compile_time_eval():
            return _e.ElementTetra()

    @property
    @override
    def mesh(self) -> pv.UnstructuredGrid:
        return self._mesh
