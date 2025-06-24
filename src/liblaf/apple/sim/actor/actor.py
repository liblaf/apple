from typing import Self

import pyvista as pv
from jaxtyping import ArrayLike, Float

from liblaf.apple import struct
from liblaf.apple.sim.dirichlet import Dirichlet
from liblaf.apple.sim.dofs import DOFs
from liblaf.apple.sim.geometry import Geometry, GeometryAttributes
from liblaf.apple.sim.region import Region


@struct.pytree
class Actor(struct.PyTreeNode):
    dirichlet: Dirichlet = struct.data(factory=Dirichlet)
    dofs: DOFs = struct.data(default=None)
    region: Region = struct.data(default=None)

    @classmethod
    def from_pyvista(cls, mesh: pv.DataSet) -> Self:
        geometry: Geometry = Geometry.from_pyvista(mesh)
        region: Region = Region.from_geometry(geometry)
        return cls.from_region(region)

    @classmethod
    def from_geometry(cls, geometry: Geometry) -> Self:
        region: Region = Region.from_geometry(geometry)
        return cls.from_region(region)

    @classmethod
    def from_region(cls, region: Region) -> Self:
        return cls(region=region)

    # region Structure

    @property
    def geometry(self) -> Geometry:
        return self.region.geometry

    # endregion Structure

    # region Numbers

    @property
    def dim(self) -> int:
        return self.region.dim

    @property
    def n_dirichlet(self) -> int:
        return self.dirichlet.size

    @property
    def n_dofs(self) -> int:
        return self.n_points * self.dim

    @property
    def n_points(self) -> int:
        return self.region.n_points

    # endregion Numbers

    # region Attributes

    @property
    def cell_data(self) -> GeometryAttributes:
        return self.geometry.cell_data

    @property
    def point_data(self) -> GeometryAttributes:
        return self.geometry.point_data

    # endregion Attributes

    # region State

    def prepare(
        self, displacement: Float[ArrayLike, "points dim"] | None = None
    ) -> Self:
        return self.update(displacement)

    def update(
        self,
        displacement: Float[ArrayLike, "points dim"] | None = None,
        velocity: Float[ArrayLike, "points dim"] | None = None,
    ) -> Self:
        actor: Self = self
        if displacement is not None:
            actor = actor.set_point_data("displacement", displacement)
        if velocity is not None:
            actor = actor.set_point_data("velocity", velocity)
        return actor

    # endregion State

    # region Geometric Operations

    def boundary(self) -> "Actor":
        raise NotImplementedError

    # endregion Geometric Operations

    # region Modifications

    def set_point_data(self, name: str, value: Float[ArrayLike, "points dim"]) -> Self:
        return self.tree_at(
            lambda self: self.point_data, replace=self.point_data.set(name, value)
        )

    def with_dofs(self, dofs: DOFs) -> Self:
        return self.evolve(dofs=dofs)

    # endregion Modifications
