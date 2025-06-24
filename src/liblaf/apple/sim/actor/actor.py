from typing import Self

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

    # region Geometric Operations

    def boundary(self) -> "Actor":
        raise NotImplementedError

    # endregion Geometric Operations

    # region Modifications

    def with_dofs(self, dofs: DOFs) -> Self:
        return self.evolve(dofs=dofs)

    # endregion Modifications
