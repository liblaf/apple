from typing import Self, override

import pyvista as pv
from jaxtyping import Array, Float, Integer
from liblaf.peach import tree

from liblaf.apple.jax import math
from liblaf.apple.jax.sim.element import ElementTetra
from liblaf.apple.jax.typing import float_, int_

from ._geometry import Geometry


@tree.define
class GeometryTetra(Geometry):
    @override
    @classmethod
    def from_pyvista(cls, mesh: pv.UnstructuredGrid) -> Self:  # pyright: ignore[reportIncompatibleMethodOverride]
        points: Float[Array, "p J"] = math.asarray(mesh.points, float_)
        cells: Integer[Array, "c a"] = math.asarray(
            mesh.cells_dict[pv.CellType.TETRA],  # pyright: ignore[reportArgumentType]
            int_,
        )
        self: Self = cls(points=points, cells=cells)
        self.copy_attributes(mesh)
        return self

    @property
    @override
    def element(self) -> ElementTetra:
        return ElementTetra()
