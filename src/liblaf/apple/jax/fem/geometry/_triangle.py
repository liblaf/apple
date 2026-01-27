from typing import Self, override

import jarp
import jax.numpy as jnp
import pyvista as pv
from jaxtyping import Array, Integer

from ._geometry import Geometry


@jarp.define
class GeometryTriangle(Geometry):
    mesh: pv.PolyData = jarp.static()  # pyright: ignore[reportIncompatibleVariableOverride]

    @override
    @classmethod
    def from_pyvista(cls, mesh: pv.PolyData) -> Self:  # pyright: ignore[reportIncompatibleMethodOverride]
        mesh = mesh.triangulate()  # pyright: ignore[reportAssignmentType]
        self: Self = cls(mesh=mesh)
        return self

    @property
    @override
    def cells_local(self) -> Integer[Array, "c a"]:
        return jnp.asarray(self.mesh.regular_faces)
