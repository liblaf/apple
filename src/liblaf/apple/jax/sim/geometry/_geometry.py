from typing import Self

import pyvista as pv
from jaxtyping import Array, Float, Integer
from liblaf.peach import tree

from liblaf.apple.jax import math
from liblaf.apple.jax.sim.element import Element
from liblaf.apple.jax.typing import int_

from ._attributes import GeometryAttributes, as_array_dict


@tree.define
class Geometry:
    points: Float[Array, "p J"] = tree.array()
    cells: Integer[Array, "c a"] = tree.array(default=None)

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

    @property
    def n_cells(self) -> int:
        return self.cells.shape[0]

    @property
    def dof_id(self) -> Integer[Array, " p"]:
        return math.asarray(self.point_data["dof-id"], int_)

    @property
    def cells_global(self) -> Integer[Array, "c a"]:
        return self.dof_id[self.cells]

    def copy_attributes(self, other: Self | pv.DataObject) -> None:
        self.point_data.update(as_array_dict(other.point_data))
        self.cell_data.update(as_array_dict(other.cell_data))
        # TODO: handle non-array field data properly
        # self.field_data.update(as_array_dict(other.field_data))
