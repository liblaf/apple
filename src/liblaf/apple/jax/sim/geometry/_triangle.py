from typing import Self, override

import pyvista as pv
from jaxtyping import Array, Integer

from liblaf.apple.jax import math, tree
from liblaf.apple.jax.typing import float_, int_

from ._geometry import Geometry


@tree.pytree
class GeometryTriangle(Geometry):
    @override
    @classmethod
    def from_pyvista(cls, mesh: pv.PolyData) -> Self:  # pyright: ignore[reportIncompatibleMethodOverride]
        mesh = mesh.triangulate()  # pyright: ignore[reportAssignmentType]
        cells: Integer[Array, "c a"] = math.asarray(mesh.regular_faces, int_)
        self: Self = cls(points=math.asarray(mesh.points, float_), cells=cells)
        self.copy_attributes(mesh)
        return self
