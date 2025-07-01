from collections.abc import Mapping
from typing import Self

import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv
import warp as wp
from jaxtyping import ArrayLike, Bool, Float

from liblaf.apple import struct
from liblaf.apple.sim.dirichlet import Dirichlet
from liblaf.apple.sim.dofs import DOFs, DOFsArray
from liblaf.apple.sim.element import Element, ElementTriangle
from liblaf.apple.sim.field.field import Field
from liblaf.apple.sim.geometry import Geometry, GeometryAttributes
from liblaf.apple.sim.region import Region

from .protocol import ComponentProtocol


@struct.pytree
class Actor(struct.PyTreeNode):
    collision_mesh: wp.Mesh = struct.static(default=None)
    components: list[ComponentProtocol] = struct.data(factory=list)
    dirichlet: Dirichlet = struct.data(factory=Dirichlet)
    dofs: DOFs = struct.data(factory=DOFsArray)
    region: Region = struct.data(default=None)

    @classmethod
    def from_pyvista(
        cls, mesh: pv.DataSet, *, collision: bool = False, grad: bool = False
    ) -> Self:
        geometry: Geometry = Geometry.from_pyvista(mesh)
        return cls.from_geometry(geometry, collision=collision, grad=grad)

    @classmethod
    def from_geometry(
        cls, geometry: Geometry, *, collision: bool = False, grad: bool = False
    ) -> Self:
        region: Region = Region.from_geometry(geometry, grad=grad)
        return cls.from_region(region, collision=collision)

    @classmethod
    def from_region(cls, region: Region, *, collision: bool | None = None) -> Self:
        if collision is None:
            collision = isinstance(region.geometry.element, ElementTriangle)
        self: Self = cls(region=region)
        self = self.update(
            displacement=jnp.zeros((region.n_points, region.dim)),
            velocity=jnp.zeros((region.n_points, region.dim)),
            force=jnp.zeros((region.n_points, region.dim)),
        )
        if collision:
            self = self.with_collision_mesh()
        return self

    # region Structure

    @property
    def element(self) -> Element:
        return self.region.element

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
    def points(self) -> Float[jax.Array, "points dim"]:
        return self.geometry.points

    @property
    def cell_data(self) -> GeometryAttributes:
        return self.geometry.cell_data

    @property
    def point_data(self) -> GeometryAttributes:
        return self.geometry.point_data

    @property
    def displacement(self) -> Float[jax.Array, "points dim"]:
        return self.point_data["displacement"]

    @property
    def positions(self) -> Float[jax.Array, "points dim"]:
        return self.points + self.displacement

    @property
    def velocity(self) -> Float[jax.Array, "points dim"]:
        return self.point_data["velocity"]

    @property
    def force(self) -> Float[jax.Array, "points dim"]:
        return self.point_data["force"]

    @property
    def mass(self) -> Float[jax.Array, "points"]:
        return self.point_data["mass"]

    # endregion Attributes

    # region Procedure

    def pre_time_step(self) -> Self:
        return self

    def pre_optim_iter(
        self, displacement: Float[ArrayLike, "points dim"] | None = None
    ) -> Self:
        actor: Self = self.update(displacement)
        if self.collision_mesh is not None:
            self.collision_mesh.points = wp.from_jax(actor.positions, dtype=wp.vec3)
            self.collision_mesh.refit()
        return actor

    def update(
        self,
        displacement: Float[ArrayLike, "points dim"] | None = None,
        velocity: Float[ArrayLike, "points dim"] | None = None,
        force: Float[ArrayLike, "points dim"] | None = None,
    ) -> Self:
        actor: Self = self
        if displacement is not None:
            actor = actor.set_point_data("displacement", displacement)
        if velocity is not None:
            actor = actor.set_point_data("velocity", velocity)
        if force is not None:
            actor = actor.set_point_data("force", force)
        return actor

    # endregion Procedure

    # region Utilities

    def make_field(self, x: Float[ArrayLike, "points dim"]) -> Field:
        return Field.from_region(self.region, x)

    # endregion Utilities

    # region Geometric Operations

    def boundary(self) -> "Actor":
        raise NotImplementedError

    # endregion Geometric Operations

    # region Modifications

    def set_dirichlet(self, dirichlet: Dirichlet) -> Self:
        actor: Self = self
        actor = actor.evolve(dirichlet=dirichlet)
        actor = actor.update(displacement=dirichlet.apply(actor.displacement))
        return actor

    def set_point_data(self, name: str, value: Float[ArrayLike, "points dim"]) -> Self:
        point_data: GeometryAttributes = self.point_data.set(name, value)
        return self.tree_at(lambda self: self.point_data, point_data)

    def update_point_data(
        self,
        updates: Mapping[str, Float[ArrayLike, "points dim"]],
        /,
        **kwargs: Float[ArrayLike, "points dim"],
    ) -> Self:
        point_data: GeometryAttributes = self.point_data.update(updates, **kwargs)
        return self.tree_at(lambda self: self.point_data, replace=point_data)

    def with_collision_mesh(self) -> Self:
        if self.collision_mesh is not None:
            return self
        mesh: wp.Mesh = self.to_warp()
        return self.evolve(collision_mesh=mesh)

    def with_dofs(self, dofs: DOFs) -> Self:
        return self.evolve(dofs=dofs)

    # endregion Modifications

    # region Exchange

    def to_pyvista(self, *, attributes: bool = True) -> pv.DataSet:
        mesh: pv.DataSet = self.geometry.to_pyvista(attributes=attributes)
        if attributes:
            dirichlet_values: Float[np.ndarray, "points dim"] = np.zeros(
                (mesh.n_points, self.dim)
            )
            dirichlet_values = np.asarray(self.dirichlet.apply(dirichlet_values))
            dirichlet_mask: Bool[np.ndarray, "points dim"] = np.zeros(
                (mesh.n_points, self.dim), dtype=bool
            )
            dirichlet_mask = np.asarray(self.dirichlet.mask(dirichlet_mask), dtype=bool)
            mesh.point_data["dirichlet-values"] = dirichlet_values
            mesh.point_data["dirichlet-mask"] = dirichlet_mask
        return mesh

    def to_warp(self, **kwargs) -> wp.Mesh:
        mesh: wp.Mesh = wp.Mesh(
            wp.from_jax(self.positions, dtype=wp.vec3),
            wp.from_jax(self.geometry.cells.ravel(), dtype=wp.int32),
            **kwargs,
        )
        return mesh

    # endregion Exchange
