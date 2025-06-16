from collections.abc import Sequence
from typing import TYPE_CHECKING, Self, override

import jax
from jaxtyping import Float

from liblaf.apple import struct, utils
from liblaf.apple.sim.abc.element import Element
from liblaf.apple.sim.abc.field import Field, FieldLike
from liblaf.apple.sim.abc.geometry import Geometry
from liblaf.apple.sim.abc.params import GlobalParams
from liblaf.apple.sim.abc.quadrature import Scheme
from liblaf.apple.sim.abc.region import Region

from ._dirichlet import Dirichlet

if TYPE_CHECKING:
    from liblaf.apple.sim.abc.operator import Operator


class Object(struct.Node):
    op: "Operator[Self]" = struct.data(default=None)

    displacement: Field = struct.data(default=None)
    displacement_prev: Field = struct.data(default=None)
    velocity: Field = struct.data(default=None)
    force: Field = struct.data(default=None)
    mass: Field = struct.data(default=None)

    dirichlet: Dirichlet = struct.data(default=None)
    dof_index: struct.Index = struct.data(default=None)

    @classmethod
    def from_region(
        cls,
        region: Region,
        /,
        displacement: FieldLike | None = 0.0,
        velocity: FieldLike | None = 0.0,
        force: FieldLike | None = 0.0,
        mass: FieldLike | None = 1.0,
    ) -> Self:
        dim: int = region.dim
        if displacement is not None:
            displacement = Field.from_region(region, displacement, dim=dim)
        if velocity is not None:
            velocity = Field.from_region(region, velocity, dim=dim)
        if force is not None:
            force = Field.from_region(region, force, dim=dim)
        if mass is not None:
            mass = Field.from_region(region, mass, dim=1)
        return cls(
            displacement=displacement,  # pyright: ignore[reportArgumentType]
            displacement_prev=displacement,  # pyright: ignore[reportArgumentType]
            velocity=velocity,  # pyright: ignore[reportArgumentType]
            force=force,  # pyright: ignore[reportArgumentType]
            mass=mass,  # pyright: ignore[reportArgumentType]
        )

    # region Structure

    @property
    def element(self) -> Element:
        return self.displacement.element

    @property
    def geometry(self) -> Geometry:
        return self.displacement.geometry

    @property
    def quadrature(self) -> Scheme:
        return self.displacement.quadrature

    @property
    def region(self) -> Region:
        return self.displacement.region

    # endregion Structure

    # region Shape

    @property
    def dim(self) -> int:
        assert len(self.displacement.dim) == 1
        return self.displacement.dim[0]

    @property
    def n_cells(self) -> int:
        return self.displacement.n_cells

    @property
    def n_dof(self) -> int:
        return self.displacement.n_dof

    @property
    def n_points(self) -> int:
        return self.displacement.n_points

    @property
    def shape(self) -> Sequence[int]:
        return self.displacement.shape

    # endregion Shape

    # region Array

    @property
    def cells(self) -> Float[jax.Array, "cells a"]:
        return self.displacement.cells

    @property
    def points(self) -> Float[jax.Array, "points dim"]:
        return self.displacement.points

    @property
    @utils.jit
    def positions(self) -> Float[jax.Array, "points dim"]:
        return jax.lax.stop_gradient(self.points) + self.displacement.values

    # endregion Array

    # region State Update

    def prepare(self) -> Self:
        return self

    def step(self, displacement: FieldLike | None, params: GlobalParams) -> Self:
        obj: Self = self.evolve(displacement_prev=self.displacement)
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
        obj: Self = self.evolve(
            displacement=self.displacement.with_values(displacement),
            velocity=self.velocity.with_values(velocity),
            force=self.force.with_values(force),
        )
        return obj

    # endregion State Update

    # region Node

    @property
    @override
    def deps(self) -> struct.NodeCollection["Operator[Self]"]:
        if self.op is None:
            return struct.NodeCollection()
        return struct.NodeCollection(self.op)

    def with_deps(self, nodes: struct.NodesLike, /) -> Self:
        nodes = struct.NodeCollection(nodes)
        if self.op is None:
            return self
        op: Operator[Self] = nodes[self.op]
        return op.update(self)

    # endregion Node

    # region Geometric Operation

    @property
    def boundary(self) -> "Object":
        from liblaf.apple.sim.operator import OperatorBoundary

        return OperatorBoundary.apply(self)

    # endregion Geometric Operation

    def export_geometry(self) -> Geometry:
        geometry: Geometry = self.geometry
        geometry = geometry.warp_by_vector(self.displacement.values)
        geometry.user_dict["id"] = self.id
        geometry.point_data["displacement"] = self.displacement.values
        geometry.point_data["velocity"] = self.velocity.values
        geometry.point_data["force"] = self.force.values
        geometry.point_data["mass"] = self.mass.values
        geometry.point_data["dof-index"] = self.dof_index.index
        return geometry
