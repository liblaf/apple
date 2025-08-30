from typing import Self

import pyvista as pv
from jaxtyping import Array, Float, Integer

from liblaf.apple.jax import tree
from liblaf.apple.jax.sim.element import Element

from ._attributes import GeometryAttributes, as_array_dict


@tree.pytree
class Geometry:
    points: Float[Array, "p J"] = tree.array()
    cells_local: Integer[Array, "c a"] = tree.array()
    cells_global: Integer[Array, "c a"] = tree.array(default=None)

    point_data: GeometryAttributes = tree.field(
        factory=lambda: GeometryAttributes(association=pv.FieldAssociation.POINT)
    )
    cell_data: GeometryAttributes = tree.field(
        factory=lambda: GeometryAttributes(association=pv.FieldAssociation.CELL)
    )
    field_data: GeometryAttributes = tree.field(
        factory=lambda: GeometryAttributes(association=pv.FieldAssociation.NONE)
    )

    @classmethod
    def from_pyvista(cls, mesh: pv.DataObject) -> "Geometry":
        from ._tetra import GeometryTetra
        from ._triangle import GeometryTriangle

        if isinstance(mesh, pv.PolyData):
            return GeometryTriangle.from_pyvista(mesh)
        if isinstance(mesh, pv.UnstructuredGrid):
            return GeometryTetra.from_pyvista(mesh)
        raise NotImplementedError

    @property
    def element(self) -> Element:
        raise NotImplementedError

    def copy_attributes(self, other: Self | pv.DataObject) -> None:
        self.point_data.update(as_array_dict(other.point_data))
        self.cell_data.update(as_array_dict(other.cell_data))
        # self.field_data.update(as_array_dict(other.field_data))  # pyright: ignore[reportArgumentType]
