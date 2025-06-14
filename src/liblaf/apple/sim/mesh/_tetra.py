from typing import Self, override

import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv
from jaxtyping import Integer
from numpy.typing import ArrayLike

from liblaf.apple import struct
from liblaf.apple.sim import element as _e

from ._mesh import Mesh


class MeshTetra(Mesh):
    _pyvista: pv.UnstructuredGrid = struct.static(default=None)

    @property
    @override
    def cells(self) -> Integer[jax.Array, "cells a=3"]:
        with jax.ensure_compile_time_eval():
            return jnp.asarray(self.pyvista.cells_dict[pv.CellType.TETRA])

    @property
    @override
    def element(self) -> _e.ElementTetra:
        with jax.ensure_compile_time_eval():
            return _e.ElementTetra()

    @property
    @override
    def pyvista(self) -> pv.UnstructuredGrid:
        return self._pyvista

    @override
    def extract(
        self, ind: Integer[ArrayLike, " sub_cells"], *, invert: bool = False
    ) -> Self:
        ind = np.asarray(ind)
        mesh: pv.UnstructuredGrid = self.pyvista.extract_cells(ind, invert=invert)
        return self.evolve(_pyvista=mesh)
