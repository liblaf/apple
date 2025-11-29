from typing import Self, override

import jax.numpy as jnp
import pyvista as pv
from jaxtyping import Array, Integer
from liblaf.peach import tree

from liblaf.apple.jax.fem.element import ElementTetra

from ._geometry import Geometry


@tree.define
class GeometryTetra(Geometry):
    mesh: pv.UnstructuredGrid = tree.field()  # pyright: ignore[reportIncompatibleVariableOverride]

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
    def cells(self) -> Integer[Array, "c a"]:
        return jnp.asarray(self.mesh.cells_dict[pv.CellType.TETRA])  # pyright: ignore[reportArgumentType]
