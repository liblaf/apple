from typing import Self, override

import jax.numpy as jnp
import pyvista as pv
from jaxtyping import Array, Integer
from liblaf.peach import tree

from ._geometry import Geometry


@tree.define
class GeometryTriangle(Geometry):
    mesh: pv.PolyData = tree.field()  # pyright: ignore[reportIncompatibleVariableOverride]

    @override
    @classmethod
    def from_pyvista(cls, mesh: pv.PolyData) -> Self:  # pyright: ignore[reportIncompatibleMethodOverride]
        mesh = mesh.triangulate()  # pyright: ignore[reportAssignmentType]
        self: Self = cls(mesh=mesh)
        return self

    @property
    def cells(self) -> Integer[Array, "c a"]:
        return jnp.asarray(self.mesh.regular_faces)
