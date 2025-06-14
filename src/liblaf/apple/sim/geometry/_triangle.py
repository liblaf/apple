from typing import Self, override

import jax
import jax.numpy as jnp
import pyvista as pv
from jaxtyping import Integer

from liblaf.apple import struct
from liblaf.apple.sim import element as _e

from ._abc import Geometry


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
    def dim(self) -> int:
        return 3

    @property
    @override
    def n_cells(self) -> int:
        return self.mesh.n_faces_strict

    @property
    @override
    def cells(self) -> Integer[jax.Array, "cells a=3"]:
        with jax.ensure_compile_time_eval():
            return jnp.asarray(self.mesh.regular_faces)

    @property
    @override
    def element(self) -> _e.ElementTriangle:
        with jax.ensure_compile_time_eval():
            return _e.ElementTriangle()

    @property
    @override
    def mesh(self) -> pv.PolyData:
        return self._mesh
