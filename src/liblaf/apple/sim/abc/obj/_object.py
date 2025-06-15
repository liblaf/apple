from typing import Self

import attrs
import jax
from jaxtyping import Float

from liblaf.apple import struct, utils
from liblaf.apple.sim.abc.element import Element
from liblaf.apple.sim.abc.field import Field, FieldLike
from liblaf.apple.sim.abc.geometry import Geometry
from liblaf.apple.sim.abc.quadrature import Scheme
from liblaf.apple.sim.abc.region import Region

from ._dirichlet import Dirichlet


class Object(struct.Node):
    id: str = struct.static(default=attrs.Factory(struct.uniq_id, takes_self=True))

    displacement: Field = struct.data(default=None)
    velocity: Field = struct.data(default=None)
    force: Field = struct.data(default=None)
    mass: Field = struct.data(default=None)

    dirichlet: Dirichlet = struct.data(default=None)
    dof_index: struct.Index = struct.data(default=None)

    @classmethod
    def from_fields(cls, displacement: Field, /) -> Self:
        self: Self = cls(displacement=displacement)
        return self

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

    def prepare(self) -> Self:
        return self

    def update(
        self,
        displacement: FieldLike | None = None,
        velocity: FieldLike | None = None,
        force: FieldLike | None = None,
    ) -> Self:
        obj: Self = self
        obj = obj.evolve(displacement=obj.displacement.with_values(displacement))
        obj = obj.evolve(velocity=obj.velocity.with_values(velocity))
        obj = obj.evolve(force=obj.force.with_values(force))
        return self.evolve()

    def with_displacement(self, displacement: FieldLike | None, /) -> Self:
        if displacement is None:
            return self
        if self.displacement is not None:
            displacement = self.displacement.with_values(displacement)
        return self.evolve(displacement=displacement)

    def with_velocity(self, velocity: FieldLike | None, /) -> Self:
        if velocity is None:
            return self
        if self.velocity is not None:
            velocity = self.velocity.with_values(velocity)
        elif self.displacement is not None:
            velocity = self.displacement.with_values(velocity)
        return self.evolve(velocity=velocity)

    def with_force(self, force: FieldLike | None, /) -> Self:
        if force is None:
            return self
        if self.force is not None:
            force = self.force.with_values(force)
        elif self.displacement is not None:
            force = self.displacement.with_values(force)
        return self.evolve(force=force)
