import functools
from collections.abc import Callable
from typing import Self

import jax
import pyvista as pv
import warp as wp
from jaxtyping import ArrayLike, Float, Integer

from liblaf.apple import struct
from liblaf.apple.sim.element import Element
from liblaf.apple.sim.quadrature import Scheme

from .attributes import GeometryAttributes


def attributes_factory(
    association: pv.FieldAssociation,
) -> Callable[..., GeometryAttributes]:
    return functools.partial(GeometryAttributes, association=association)


@struct.pytree
class Geometry(struct.PyTreeMixin):
    cells: Integer[jax.Array, "cells a"] = struct.array(default=None)
    points: Float[jax.Array, "points dim"] = struct.array(default=None)

    cell_data: GeometryAttributes = struct.container(
        factory=attributes_factory(pv.FieldAssociation.CELL)
    )
    point_data: GeometryAttributes = struct.container(
        factory=attributes_factory(pv.FieldAssociation.POINT)
    )

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
    def cell_id(self) -> Integer[jax.Array, "cells"]:
        return self.cell_data.get("cell-id")  # pyright: ignore[reportReturnType]

    @property
    def point_id(self) -> Integer[jax.Array, "points"]:
        return self.point_data.get("point-id")  # pyright: ignore[reportReturnType]

    # endregion Attributes

    # region Manipulation

    def set_cell_data(self, name: str, value: ArrayLike, /) -> Self:
        return self.update_cell_data(self.cell_data.set(name, value))

    def set_point_data(self, name: str, value: ArrayLike, /) -> Self:
        return self.update_point_data(self.point_data.set(name, value))

    def update_cell_data(self, cell_data: struct.MappingLike, /) -> Self:
        return self.evolve(cell_data=self.cell_data.update(cell_data))

    def update_point_data(self, point_data: struct.MappingLike, /) -> Self:
        return self.evolve(point_data=self.point_data.update(point_data))

    # endregion Manipulation

    # region Geometric Operations

    @property
    def boundary(self) -> "Geometry":
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
