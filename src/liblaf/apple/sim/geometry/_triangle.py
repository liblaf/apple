from typing import Self, override

import flax.struct
import jax.numpy as jnp
import pyvista as pv

from ._abc import Geometry


class GeometryTriangle(Geometry):
    mesh: pv.PolyData = flax.struct.field(pytree_node=False, default=None)

    @override
    def with_cells(self) -> Self:
        if self.cells is not None:
            return self
        return self.replace(cells=jnp.asarray(self.mesh.regular_faces))
