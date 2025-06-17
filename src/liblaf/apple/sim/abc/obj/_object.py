from collections.abc import Sequence
from typing import TYPE_CHECKING, Self, override

import jax
from jaxtyping import DTypeLike, Float

from liblaf.apple import struct, utils
from liblaf.apple.sim.abc.element import Element
from liblaf.apple.sim.abc.field import Field, FieldLike
from liblaf.apple.sim.abc.geometry import Geometry, GeometryAttributes
from liblaf.apple.sim.abc.params import GlobalParams
from liblaf.apple.sim.abc.quadrature import Scheme
from liblaf.apple.sim.abc.region import Region

from ._dirichlet import Dirichlet

if TYPE_CHECKING:
    from liblaf.apple.sim.abc.operator import Operator


def field_property(name: str) -> property:
    def getter(self: "Object") -> Field:
        return self.fields[name]

    def setter(self: "Object", values: FieldLike) -> None:
        self.fields[name] = self.make_field(values)

    def deleter(self: "Object") -> None:
        del self.fields[name]

    return property(getter, setter, deleter)


class Object(struct.GraphNode):
    fields: struct.PyTreeDict[Field] = struct.pytree_dict()
    _region: Region = struct.field(default=None)

    op: "Operator[Self]" = struct.field(default=None)

    if TYPE_CHECKING:
        displacement: Field = struct.field(default=None)
        displacement_prev: Field = struct.field(default=None)
        velocity: Field = struct.field(default=None)
        force: Field = struct.field(default=None)
        mass: Field = struct.field(default=None)
    else:
        displacement = field_property("displacement")
        displacement_prev = field_property("displacement_prev")
        velocity = field_property("velocity")
        force = field_property("force")
        mass = field_property("mass")

    dirichlet: Dirichlet = struct.field(default=None)
    dof_map: struct.DofMap = struct.field(default=None)

    @classmethod
    def from_region(
        cls,
        region: Region,
        /,
        displacement: FieldLike | None = 0.0,
        velocity: FieldLike | None = 0.0,
        force: FieldLike | None = 0.0,
        mass: FieldLike | None = None,
    ) -> Self:
        self: Self = cls(_region=region)
        dim: int = region.dim
        if displacement is not None:
            self = self.set_field("displacement", displacement, dim=dim)
            self = self.set_field("displacement_prev", displacement, dim=dim)
        if velocity is not None:
            self = self.set_field("velocity", velocity, dim=dim)
        if force is not None:
            self = self.set_field("force", force, dim=dim)
        if mass is not None:
            self = self.set_field("mass", mass, dim=dim)
        return self

    # region Structure

    @property
    def element(self) -> Element:
        return self.region.element

    @property
    def geometry(self) -> Geometry:
        return self.region.geometry

    @property
    def quadrature(self) -> Scheme:
        return self.region.quadrature

    @property
    def region(self) -> Region:
        return self._region

    # endregion Structure

    # region Shape

    @property
    def dim(self) -> int:
        return self.region.dim

    @property
    def n_cells(self) -> int:
        return self.region.n_cells

    @property
    def n_dof(self) -> int:
        return self.displacement.n_dof

    @property
    def n_points(self) -> int:
        return self.region.n_points

    @property
    def shape(self) -> Sequence[int]:
        return self.displacement.shape

    # endregion Shape

    # region Attributes

    @property
    def cells(self) -> Float[jax.Array, "cells a"]:
        return self.region.cells

    @property
    def points(self) -> Float[jax.Array, "points dim"]:
        return self.region.points

    @property
    @utils.jit
    def positions(self) -> Float[jax.Array, "points dim"]:
        return self.points + self.displacement.values

    @property
    def point_data(self) -> GeometryAttributes:
        return self.geometry.point_data

    @property
    def cell_data(self) -> GeometryAttributes:
        return self.geometry.cell_data

    @property
    def field_data(self) -> GeometryAttributes:
        return self.geometry.field_data

    @property
    def user_dict(self) -> dict:
        return self.geometry.user_dict

    # endregion Attributes

    # region State Update

    def step(self, displacement: FieldLike | None, params: GlobalParams) -> Self:
        obj: Self = self.set_field("displacement_prev", self.displacement)
        obj = obj.update(
            displacement=displacement,
            velocity=displacement - self.displacement / params.time_step,
        )
        return obj

    def update(
        self,
        displacement: FieldLike | None = None,
        velocity: FieldLike | None = None,
        force: FieldLike | None = None,
    ) -> Self:
        obj: Self = self
        if displacement is None:
            obj = obj.set_field("displacement", displacement)
        if velocity is None:
            obj = obj.set_field("velocity", velocity)
        if force is None:
            obj = obj.set_field("force", force)
        return obj

    # endregion State Update

    # region Computational Graph

    @property
    @override
    def deps(self) -> struct.PyTreeDict:
        if self.op is None:
            return struct.PyTreeDict()
        return struct.PyTreeDict(self.op)

    def with_deps(self, nodes: struct.MappingLike, /) -> Self:
        nodes = struct.PyTreeDict(nodes)
        if self.op is None:
            return self
        op: Operator[Self] = nodes[self.op]
        return op.update(self.replace(op=op))

    # endregion Computational Graph

    # region Geometric Operation

    @property
    def boundary(self) -> "Object":
        from liblaf.apple.sim.operator import OperatorBoundary

        return OperatorBoundary.apply(self)

    # endregion Geometric Operation

    def make_field(
        self,
        values: FieldLike,
        /,
        dim: int | Sequence[int] | None = None,
        dtype: DTypeLike | None = None,
    ) -> Field:
        return Field.from_region(self.region, values, dim=dim, dtype=dtype)

    def set_field(
        self,
        name: str,
        values: FieldLike | None = None,
        /,
        dim: int | Sequence[int] | None = None,
        dtype: DTypeLike | None = None,
    ) -> Self:
        if values is None:
            return self
        if name in self.fields:
            if dim is None:
                dim = self.fields[name].dim
            if dtype is None:
                dtype = self.fields[name].dtype
        self.fields[name] = self.make_field(values, dim=dim, dtype=dtype)
        return self

    def export_geometry(self) -> Geometry:
        geometry: Geometry = self.geometry
        geometry = geometry.warp_by_vector(self.displacement.values)
        geometry.user_dict["id"] = self.id
        geometry.point_data.update({k: v.values for k, v in self.fields.items()})
        return geometry
