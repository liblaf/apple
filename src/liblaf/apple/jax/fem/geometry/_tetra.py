from typing import Self, override

import jarp
import jax.numpy as jnp
import pyvista as pv
from jaxtyping import Array, Integer

from liblaf.apple.jax.fem.element import ElementTetra

from ._geometry import Geometry


@jarp.define
class GeometryTetra(Geometry):
    mesh: pv.UnstructuredGrid = jarp.field()  # pyright: ignore[reportIncompatibleVariableOverride]

    @override
    @classmethod
    def from_pyvista(cls, mesh: pv.UnstructuredGrid) -> Self:  # pyright: ignore[reportIncompatibleMethodOverride]
        self: Self = cls(mesh=mesh)
        return self

    @property
    @override
    def element(self) -> ElementTetra:
        return ElementTetra()

    @property
    @override
    def cells_local(self) -> Integer[Array, "c a"]:
        return jnp.asarray(self.mesh.cells_dict[pv.CellType.TETRA])  # pyright: ignore[reportArgumentType]
