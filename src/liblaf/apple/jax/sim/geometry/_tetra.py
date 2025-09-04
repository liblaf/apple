from typing import Self, override

import jax.numpy as jnp
import pyvista as pv
from jaxtyping import Array, Integer

from liblaf.apple.jax import tree
from liblaf.apple.jax.sim.element import ElementTetra

from ._geometry import Geometry


@tree.pytree
class GeometryTetra(Geometry):
    @override
    @classmethod
    def from_pyvista(cls, mesh: pv.UnstructuredGrid) -> Self:  # pyright: ignore[reportIncompatibleMethodOverride]
        cells_local: Integer[Array, "c a"] = jnp.asarray(
            mesh.cells_dict[pv.CellType.TETRA]  # pyright: ignore[reportArgumentType]
        )
        self: Self = cls(points=jnp.asarray(mesh.points), cells_local=cells_local)
        self.copy_attributes(mesh)
        if (point_id := self.point_data.get("point-id")) is not None:
            cells_global: Integer[Array, "c a"] = point_id[cells_local]
            self.cells_global = cells_global
        return self

    @property
    @override
    def element(self) -> ElementTetra:
        return ElementTetra()
