from typing import Self, override

import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv

from liblaf.apple.sim.core import Geometry
from liblaf.apple.sim.element import ElementTriangle


class GeometryTriangle(Geometry):
    @classmethod
    @override
    def from_pyvista(cls, mesh: pv.PolyData, /) -> Self:
        self: Self = cls(
            points=jnp.asarray(mesh.points), cells=jnp.asarray(mesh.regular_faces)
        )
        self = self.copy_attributes(mesh)
        return self

    @property
    @override
    def element(self) -> ElementTriangle:
        with jax.ensure_compile_time_eval():
            return ElementTriangle()

    @property
    @override
    def structure(self) -> pv.PolyData:
        return pv.PolyData.from_regular_faces(
            np.asarray(self.points), np.asarray(self.cells)
        )
