from typing import Self, override

import jax
import jax.numpy as jnp
import pyvista as pv
from jaxtyping import Integer

from liblaf.apple import struct
from liblaf.apple.sim.abc import Geometry
from liblaf.apple.sim.element import ElementTriangle


class GeometryTriangle(Geometry):
    _pyvista: pv.PolyData = struct.static(default=None)

    @classmethod
    @override
    def from_pyvista(cls, mesh: pv.PolyData, /) -> Self:
        return cls(_pyvista=mesh)

    @property
    @override
    def element(self) -> ElementTriangle:
        with jax.ensure_compile_time_eval():
            return ElementTriangle()

    @property
    @override
    def pyvista(self) -> pv.PolyData:
        return self._pyvista

    @property
    @override
    def n_cells(self) -> int:
        return self.pyvista.n_faces_strict

    @property
    @override
    def cells(self) -> Integer[jax.Array, "cells a=3"]:
        with jax.ensure_compile_time_eval():
            return jnp.asarray(self.pyvista.regular_faces)
