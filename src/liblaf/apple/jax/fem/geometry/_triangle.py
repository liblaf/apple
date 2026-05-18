from typing import Self, cast, override

import jax.numpy as jnp
import pyvista as pv
from jaxtyping import Array, Integer

from liblaf import jarp

from ._geometry import Geometry


@jarp.define
class GeometryTriangle(Geometry):
    mesh: pv.PolyData = jarp.static()  # pyright: ignore[reportIncompatibleVariableOverride]

    @override
    @classmethod
    def from_pyvista(cls, mesh: pv.DataObject) -> Self:
        mesh: pv.PolyData = cast("pv.PolyData", mesh)
        mesh: pv.PolyData = mesh.triangulate()
        self: Self = cls(mesh=mesh)
        return self

    @property
    @override
    def cells_local(self) -> Integer[Array, "c a"]:
        return jnp.asarray(self.mesh.regular_faces)
