from typing import Self, override

import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv

from liblaf.apple import struct
from liblaf.apple.sim.element import Element, ElementTriangle

from .geometry import Geometry


@struct.pytree
class GeometryTriangle(Geometry):
    @classmethod
    def from_pyvista(cls, mesh: pv.PolyData) -> Self:
        return cls(
            points=jnp.asarray(mesh.points), cells=jnp.asarray(mesh.regular_faces)
        ).copy_attributes(mesh)

    @property
    @override
    def element(self) -> Element:
        with jax.ensure_compile_time_eval():
            return ElementTriangle()

    @override
    def to_pyvista(self, *, attributes: bool = True) -> pv.PolyData:
        mesh: pv.PolyData = pv.PolyData.from_regular_faces(
            np.asarray(self.points), np.asarray(self.cells)
        )
        if attributes:
            mesh.cell_data.update(self.cell_data)
            mesh.point_data.update(self.point_data)
        return mesh
