from collections.abc import Sequence
from typing import Protocol, Self, override

import jax
import jax.numpy as jnp
from jaxtyping import Float

from liblaf.apple import struct
from liblaf.apple.sim.core.field import Field, FieldLike
from liblaf.apple.sim.core.geometry import Geometry, GeometryAttributes
from liblaf.apple.sim.core.params import GlobalParams
from liblaf.apple.sim.core.region import Region

from ._dirichlet import Dirichlet


class Operator(Protocol):
    @property
    def id(self) -> str: ...
    def update[T: Object](self, obj: T) -> T: ...


class Object(struct.GraphNode):
    dirichlet: Dirichlet = struct.data(default=None)
    dof: struct.DofMap = struct.data(default=None)
    fields: struct.FrozenDict[Field] = struct.mapping(factory=struct.FrozenDict)
    op: Operator = struct.data(default=None)
    region: Region = struct.data(default=None)

    @classmethod
    def from_region(cls, region: Region) -> Self:
        region = _compute_mass(region)
        self: Self = cls(region=region)
        self = self.with_field("displacement", jnp.zeros((1, 3)))
        self = self.with_field("displacement_prev", jnp.zeros((1, 3)))
        self = self.with_field("velocity", jnp.zeros((1, 3)))
        self = self.with_field("force", jnp.zeros((1, 3)))
        self = self.with_field(
            "mass", region.point_data["mass"].reshape((region.n_points, 1))
        )
        return self

    # region Structure

    @property
    def geometry(self) -> Geometry:
        return self.region.geometry

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

    # region Fields

    @property
    def displacement(self) -> Field:
        return self.fields["displacement"]

    @property
    def displacement_prev(self) -> Field:
        return self.fields["displacement_prev"]

    @property
    def velocity(self) -> Field:
        return self.fields["velocity"]

    @property
    def force(self) -> Field:
        return self.fields["force"]

    @property
    def mass(self) -> Field:
        return self.fields["mass"]

    @property
    def positions(self) -> Float[jax.Array, " points 3"]:
        return self.region.points + self.displacement.values

    def with_field(self, name: str, field: FieldLike) -> Self:
        if field is None:
            return self
        if name in self.fields:
            field = self.fields[name].with_values(field)
        field = Field.from_region(self.region, field)
        return self.evolve(fields=self.fields.copy({name: field}))

    @property
    def point_data(self) -> GeometryAttributes:
        return self.region.point_data

    @property
    def cell_data(self) -> GeometryAttributes:
        return self.region.cell_data

    # endregion Fields

    # region Computational Graph

    @property
    @override
    def deps(self) -> struct.FrozenDict[struct.GraphNode]:
        return struct.FrozenDict(self.op)

    def step(self, displacement: FieldLike, /, params: GlobalParams) -> Self:
        obj: Self = self
        obj = obj.with_field(
            "velocity", (displacement - obj.displacement) / params.time_step
        )
        obj = obj.with_field("displacement_prev", obj.displacement)
        obj = obj.update(displacement)
        return obj

    def update(
        self,
        displacement: FieldLike | None = None,
        velocity: FieldLike | None = None,
        force: FieldLike | None = None,
    ) -> Self:
        obj: Self = self
        if displacement is not None:
            obj = obj.with_field("displacement", displacement)
        if velocity is not None:
            obj = obj.with_field("velocity", velocity)
        if force is not None:
            obj = obj.with_field("force", force)
        return obj

    @override
    def with_deps(self, deps: struct.MappingLike, /) -> Self:
        if self.op is None:
            return self
        deps = struct.FrozenDict(deps)
        op: Operator = deps[self.op]
        obj: Self = self.evolve(op=op)
        obj = op.update(self)
        return obj

    # endregion Computational Graph

    # region Geometric Operations

    @property
    def boundary(self) -> Self:
        raise NotImplementedError

    # endregion Geometric Operations

    def warp_by_disp(self) -> Geometry:
        geometry = self.geometry.warp_by_vector(self.displacement.values)
        geometry = geometry.evolve(
            point_data=geometry.point_data.copy(
                {
                    "displacement": self.displacement.values,
                    "velocity": self.velocity.values,
                    "force": self.force.values,
                }
            )
        )
        return geometry


def _compute_mass(region: Region) -> Region:
    cell_mass: Float[jax.Array, " cells"] = (
        region.cell_data["density"] * region.dV[:, 0]
    )
    cell_mass = cell_mass.reshape((region.n_cells,))
    point_mass: Float[jax.Array, " points"] = (
        region.gather(
            jnp.broadcast_to(
                cell_mass[:, None], (region.n_cells, region.element.n_points)
            )
        )
        / region.element.n_points
    )
    point_mass = point_mass.reshape((region.n_points,))
    return region.evolve(
        _geometry=region.geometry.evolve(
            cell_data=region.cell_data.copy({"mass": cell_mass}),
            point_data=region.point_data.copy({"mass": point_mass}),
        )
    )
