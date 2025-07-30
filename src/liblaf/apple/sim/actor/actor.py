from collections.abc import Mapping
from typing import Self

import jax.numpy as jnp
import numpy as np
import pyvista as pv
import warp as wp
from jaxtyping import Array, ArrayLike, Bool, Float, Shaped
from typing_extensions import deprecated

from liblaf.apple import struct, utils
from liblaf.apple.sim.dirichlet import Dirichlet
from liblaf.apple.sim.dofs import DOFs
from liblaf.apple.sim.element import Element
from liblaf.apple.sim.field import Field
from liblaf.apple.sim.geometry import Geometry, GeometryAttributes
from liblaf.apple.sim.region import Region


class Actor(struct.PyTreeNodeMutable):
    collision_mesh: wp.Mesh | None = struct.field(default=None, static=True)
    dirichlet_local: Dirichlet | None = struct.field(default=None)
    dofs_global: DOFs = struct.field(default=None)
    region: Region = struct.field(default=None)

    @classmethod
    def from_region(
        cls, region: Region, *, collision: bool = False, id_: str | None = None
    ) -> Self:
        self: Self = cls(region=region, id=id_)  # pyright: ignore[reportArgumentType]
        self = self.update(
            displacement=jnp.zeros((region.n_points, region.dim)),
            velocity=jnp.zeros((region.n_points, region.dim)),
            force=jnp.zeros((region.n_points, region.dim)),
        )
        if collision:
            self = self.with_collision_mesh()
        return self

    @classmethod
    def from_geometry(
        cls,
        geometry: Geometry,
        *,
        collision: bool = False,
        grad: bool = False,
        id_: str | None = None,
    ) -> Self:
        region: Region = Region.from_geometry(geometry, grad=grad)
        return cls.from_region(region, collision=collision, id_=id_)

    @classmethod
    def from_pyvista(
        cls,
        mesh: pv.DataSet,
        *,
        collision: bool = False,
        grad: bool = False,
        id_: str | None = None,
    ) -> Self:
        geometry: Geometry = Geometry.from_pyvista(mesh)
        return cls.from_geometry(geometry, collision=collision, grad=grad, id_=id_)

    # region Structure

    @property
    def dirichlet_global(self) -> Dirichlet | None:
        if self.dofs_global is None or self.dirichlet_local is None:
            return None
        return self.dirichlet_local.remap(self.dofs_global)

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
    def n_cells(self) -> int:
        return self.region.n_cells

    @property
    def n_dirichlet(self) -> int:
        if self.dirichlet_local is None:
            return 0
        return self.dirichlet_local.size

    @property
    def n_dofs(self) -> int:
        return self.n_points * self.dim

    @property
    def n_points(self) -> int:
        return self.region.n_points

    # endregion Numbers

    # region Attributes

    @property
    def points(self) -> Float[Array, "points dim"]:
        return self.geometry.points

    @property
    def cell_data(self) -> GeometryAttributes:
        return self.geometry.cell_data

    @property
    def point_data(self) -> GeometryAttributes:
        return self.geometry.point_data

    @property
    def field_data(self) -> GeometryAttributes:
        return self.geometry.field_data

    @property
    def displacement(self) -> Float[Array, "points dim"]:
        return self.point_data["displacement"]

    @property
    def positions(self) -> Float[Array, "points dim"]:
        return self.points + self.displacement

    @property
    def velocity(self) -> Float[Array, "points dim"]:
        return self.point_data["velocity"]

    @property
    def force_ext(self) -> Float[Array, "points dim"]:
        return self.point_data["force-ext"]

    @property
    def mass(self) -> Float[Array, "points"]:
        return self.point_data["mass"]

    # endregion Attributes

    # region Procedure

    def pre_time_step(self) -> Self:
        return self

    def pre_optim_iter(
        self, displacement: Float[ArrayLike, "points dim"] | None = None
    ) -> Self:
        self.update(displacement=displacement)
        if self.collision_mesh is not None:
            self.collision_mesh.points = wp.from_jax(self.positions, dtype=wp.vec3)
            self.collision_mesh.refit()
        return self

    @utils.jit(inline=True, validate=False)
    @deprecated("Actor.pre_optim_iter_jit() is deprecated.")
    def pre_optim_iter_jit(
        self, displacement: Float[ArrayLike, "points dim"] | None = None
    ) -> Self:
        actor: Self = self.update(displacement)
        return actor

    @deprecated("Actor.pre_optim_iter_no_jit() is deprecated.")
    def pre_optim_iter_no_jit(
        self,
        displacement: Float[ArrayLike, "points dim"] | None = None,  # noqa: ARG002
    ) -> Self:
        if self.collision_mesh is not None:
            self.collision_mesh.points = wp.from_jax(self.positions, dtype=wp.vec3)
            self.collision_mesh.refit()
        return self

    def update(
        self,
        displacement: Float[ArrayLike, "points dim"] | None = None,
        velocity: Float[ArrayLike, "points dim"] | None = None,
        force: Float[ArrayLike, "points dim"] | None = None,
    ) -> Self:
        if displacement is not None:
            self.point_data["displacement"] = displacement
        if velocity is not None:
            self.point_data["velocity"] = velocity
        if force is not None:
            self.point_data["force"] = force
        return self

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

    @deprecated("Use `self.point_data[name] = value` instead.")
    def set_point_data(self, name: str, value: Shaped[ArrayLike, "points dim"]) -> Self:
        self.point_data[name] = value
        return self

    @deprecated("Use `self.field_data[name] = value` instead.")
    def set_field_data(self, name: str, value: Shaped[ArrayLike, "..."]) -> Self:
        self.field_data[name] = value
        return self

    @deprecated("Use `self.point_data.update(...)` instead.")
    def update_point_data(
        self,
        updates: Mapping[str, Shaped[ArrayLike, "points ..."]],
        /,
        **kwargs: Shaped[ArrayLike, "points ..."],
    ) -> Self:
        self.point_data.update(updates, **kwargs)
        return self

    @deprecated("Use `self.field_data.update(...)` instead.")
    def update_field_data(
        self,
        updates: Mapping[str, Shaped[ArrayLike, "..."]],
        /,
        **kwargs: Shaped[ArrayLike, "..."],
    ) -> Self:
        self.field_data.update(updates, **kwargs)
        return self

    def with_collision_mesh(self) -> Self:
        if self.collision_mesh is not None:
            return self
        mesh: wp.Mesh = self.to_warp()
        return self.replace(collision_mesh=mesh)

    def with_dirichlet(self, dirichlet_local: Dirichlet) -> Self:
        actor: Self = self
        actor = actor.replace(dirichlet_local=dirichlet_local)
        actor = actor.update(displacement=dirichlet_local.apply(actor.displacement))
        return actor

    def with_dofs(self, dofs_global: DOFs) -> Self:
        return self.replace(dofs_global=dofs_global)

    # endregion Modifications

    # region Exchange

    def to_pyvista(self, *, attributes: bool = True) -> pv.DataSet:
        mesh: pv.DataSet = self.geometry.to_pyvista(attributes=attributes)
        if attributes:
            dirichlet_mask: Bool[np.ndarray, "points dim"] = np.zeros(
                (mesh.n_points, self.dim), dtype=bool
            )
            dirichlet_values: Float[np.ndarray, "points dim"] = np.zeros(
                (mesh.n_points, self.dim)
            )
            if self.dirichlet_local is not None:
                dirichlet_mask = np.asarray(
                    self.dirichlet_local.mask(dirichlet_mask), dtype=bool
                )
                dirichlet_values = np.asarray(
                    self.dirichlet_local.apply(dirichlet_values)
                )
            mesh.point_data["dirichlet-mask"] = dirichlet_mask
            mesh.point_data["dirichlet-values"] = dirichlet_values
        return mesh

    def to_warp(self, **kwargs) -> wp.Mesh:
        mesh: wp.Mesh = wp.Mesh(
            wp.from_jax(self.positions, dtype=wp.vec3),
            wp.from_jax(self.geometry.cells.ravel(), dtype=wp.int32),
            **kwargs,
        )
        return mesh

    # endregion Exchange
