from typing import Self, override

import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv
from jaxtyping import Integer
from numpy.typing import ArrayLike

from liblaf.apple import struct
from liblaf.apple.sim import element as _e

from ._geometry import Geometry


class GeometryTriangle(Geometry):
    _mesh: pv.PolyData = struct.static(default=None)

    @classmethod
    @override
    def from_pyvista(cls, mesh: pv.PolyData) -> Self:
        mesh = mesh.triangulate()
        self: Self = cls(_mesh=mesh)
        return self

    @property
    @override
    def element(self) -> _e.ElementTriangle:
        with jax.ensure_compile_time_eval():
            return _e.ElementTriangle()

    @property
    @override
    def pyvista(self) -> pv.PolyData:
        return self._mesh

    @property
    @override
    def n_cells(self) -> int:
        return self.pyvista.n_faces_strict

    @property
    @override
    def cells(self) -> Integer[jax.Array, "cells a=3"]:
        with jax.ensure_compile_time_eval():
            return jnp.asarray(self.pyvista.regular_faces)

    def extract(
        self, ind: Integer[ArrayLike, " sub_cells"], *, invert: bool = False
    ) -> Self:
        ind = np.asarray(ind)
        ug: pv.UnstructuredGrid = self.pyvista.extract_cells(ind, invert=invert)
        mesh: pv.PolyData = ug.extract_surface()
        return self.evolve(_pyvista=mesh)
