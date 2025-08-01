import functools
from typing import Self

import pyvista as pv
import warp as wp
from jaxtyping import Array, ArrayLike, Float, Integer
from typing_extensions import deprecated

from liblaf import grapes
from liblaf.apple import struct
from liblaf.apple.sim.element import Element
from liblaf.apple.sim.quadrature import Scheme

from .attributes import GeometryAttributes


class Geometry(struct.PyTree):
    cells: Integer[Array, "cells a"] = struct.array(default=None)
    points: Float[Array, "points dim"] = struct.array(default=None)

    cell_data: GeometryAttributes = struct.container(
        factory=GeometryAttributes.factory(pv.FieldAssociation.CELL)
    )
    point_data: GeometryAttributes = struct.container(
        factory=GeometryAttributes.factory(pv.FieldAssociation.POINT)
    )
    field_data: GeometryAttributes = struct.container(
        factory=GeometryAttributes.factory(pv.FieldAssociation.NONE)
    )

    @classmethod
    def from_pyvista(cls, mesh: pv.DataSet) -> "Geometry":
        return geometry_from_pyvista(mesh)

    # region Numbers

    @property
    def dim(self) -> int:
        return self.points.shape[1]

    @property
    def n_cells(self) -> int:
        return self.cells.shape[0]

    @property
    def n_points(self) -> int:
        return self.points.shape[0]

    # endregion Numbers

    # region Structure

    @property
    def element(self) -> Element:
        raise NotImplementedError

    @property
    def quadrature(self) -> Scheme:
        return self.element.quadrature

    # endregion Structure

    # region Attributes

    @property
    def cell_id(self) -> Integer[Array, "cells"]:
        return self.cell_data.get("cell-id")  # pyright: ignore[reportReturnType]

    @property
    def point_id(self) -> Integer[Array, "points"]:
        return self.point_data.get("point-id")  # pyright: ignore[reportReturnType]

    # endregion Attributes

    # region Manipulation

    def copy_attributes(self, mesh: "pv.DataSet | Geometry", /) -> Self:
        self.point_data.update(mesh.point_data)
        self.cell_data.update(mesh.cell_data)
        self.field_data.update(mesh.field_data)
        return self

    @deprecated("Use `self.cell_data[name] = value` instead.")
    def set_cell_data(self, name: str, value: ArrayLike, /) -> Self:
        self.cell_data[name] = value
        return self

    @deprecated("Use `self.point_data[name] = value` instead.")
    def set_point_data(self, name: str, value: ArrayLike, /) -> Self:
        self.point_data[name] = value
        return self

    @deprecated("Use `self.field_data[name] = value` instead.")
    def set_field_data(self, name: str, value: ArrayLike, /) -> Self:
        self.field_data[name] = value
        return self

    @deprecated("Use `self.cell_data.update(...)` instead.")
    def update_cell_data(self, cell_data: struct.MappingLike, /) -> Self:
        self.cell_data.update(cell_data)
        return self

    @deprecated("Use `self.point_data.update(...)` instead.")
    def update_point_data(self, point_data: struct.MappingLike, /) -> Self:
        self.point_data.update(point_data)
        return self

    @deprecated("Use `self.field_data.update(...)` instead.")
    def update_field_data(self, field_data: struct.MappingLike, /) -> Self:
        self.field_data.update(field_data)
        return self

    # endregion Manipulation

    # region Geometric Operations

    def boundary(self, *, attributes: bool = True) -> "Geometry":
        raise NotImplementedError

    # endregion Geometric Operations

    # region Exchange

    def to_pyvista(self, *, attributes: bool = True) -> pv.DataSet:
        raise NotImplementedError

    def to_warp(self, **kwargs) -> wp.Mesh:
        return wp.Mesh(
            wp.from_jax(self.points, dtype=wp.vec3),
            wp.from_jax(self.cells.ravel(), dtype=wp.int32),
            **kwargs,
        )

    # endregion Exchange


@functools.singledispatch
def geometry_from_pyvista(*args, **kwargs) -> Geometry:
    raise grapes.error.DispatchLookupError(geometry_from_pyvista, args, kwargs)


@geometry_from_pyvista.register(pv.PolyData)
def _(mesh: pv.PolyData) -> Geometry:
    from .triangle import GeometryTriangle

    return GeometryTriangle.from_pyvista(mesh)


@geometry_from_pyvista.register(pv.UnstructuredGrid)
def _(mesh: pv.UnstructuredGrid) -> Geometry:
    from .tetra import GeometryTetra

    return GeometryTetra.from_pyvista(mesh)
