from typing import Any, Self

import jax
import pyvista as pv
import warp as wp
from jaxtyping import Float, Integer
from numpy.typing import ArrayLike

from liblaf.apple import struct
from liblaf.apple.sim.core.element import Element
from liblaf.apple.sim.core.quadrature import Scheme

from ._attributes import GeometryAttributes, data_property


class Geometry(struct.PyTree):
    points: Float[jax.Array, "points dim"] = struct.array(default=None)
    cells: Integer[jax.Array, "cells a"] = struct.array(default=None)
    point_data: GeometryAttributes = struct.mapping(factory=GeometryAttributes)
    cell_data: GeometryAttributes = struct.mapping(factory=GeometryAttributes)
    field_data: GeometryAttributes = struct.mapping(factory=GeometryAttributes)
    user_dict: dict[Any, Any] = struct.static(factory=dict, converter=dict)

    @classmethod
    def from_pyvista(cls, mesh: pv.DataSet, /) -> Self:
        raise NotImplementedError

    # region Structure

    @property
    def element(self) -> Element:
        raise NotImplementedError

    @property
    def structure(self) -> pv.DataSet:
        raise NotImplementedError

    @property
    def quadrature(self) -> Scheme:
        return self.element.quadrature

    # endregion Structure

    # region Shape

    @property
    def dim(self) -> int:
        return self.points.shape[-1]

    @property
    def n_cells(self) -> int:
        return self.cells.shape[0]

    @property
    def n_points(self) -> int:
        return self.points.shape[0]

    # endregion Shape

    # region Attributes

    acceleration = data_property("acceleration", pv.FieldAssociation.POINT)
    displacement = data_property("displacement", pv.FieldAssociation.POINT)
    force = data_property("force", pv.FieldAssociation.POINT)
    jac = data_property("jac", pv.FieldAssociation.POINT)
    original_point_id = data_property("point-id", pv.FieldAssociation.POINT)
    point_mass = data_property("point-mass", pv.FieldAssociation.POINT)
    velocity = data_property("velocity", pv.FieldAssociation.POINT)

    cell_mass = data_property("cell-mass", pv.FieldAssociation.CELL)
    density = data_property("density", pv.FieldAssociation.CELL)
    original_cell_id = data_property("cell-id", pv.FieldAssociation.CELL)

    def copy_attributes(self, other: "Geometry | pv.DataSet") -> Self:
        return self.evolve(
            point_data=self.point_data.copy(other.point_data),
            cell_data=self.cell_data.copy(other.cell_data),
            field_data=self.field_data.copy(other.field_data),
            user_dict={**self.user_dict, **other.user_dict},
        )

    # endregion Attributes

    # region Geometric Operations

    @property
    def boundary(self) -> "Geometry":
        raise NotImplementedError

    def warp_by_vector(self, displacement: Float[ArrayLike, "#points dim"]) -> Self:
        points: Float[jax.Array, "points dim"] = self.points + displacement
        return self.evolve(points=points)

    # endregion Geometric Operations

    # region Exchange

    def to_pyvista(self) -> pv.DataSet:
        mesh: pv.DataSet = self.structure
        mesh.point_data.update(self.point_data)
        mesh.cell_data.update(self.cell_data)
        mesh.field_data.update(self.field_data)
        mesh.user_dict.update(self.user_dict)
        return mesh

    def to_warp(self, **kwargs) -> wp.Mesh:
        return wp.Mesh(
            wp.from_jax(self.points, dtype=wp.vec3),
            wp.from_jax(self.cells.ravel(), dtype=int),
            **kwargs,
        )

    # endregion Exchange
