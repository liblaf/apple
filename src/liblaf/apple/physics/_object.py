from typing import Self

import flax.struct
import jax
import jax.numpy as jnp
import pyvista as pv
import warp as wp
from jaxtyping import ArrayLike, Float, Integer

from liblaf.apple import math


class Object(flax.struct.PyTreeNode):
    # data fields
    displacement: Float[jax.Array, "points dim"] = flax.struct.field(default=None)
    force: Float[jax.Array, "points dim"] = flax.struct.field(default=None)
    velocity: Float[jax.Array, "points dim"] = flax.struct.field(default=None)
    cells: Integer[jax.Array, "cells 4"] = flax.struct.field(default=None)

    # computed properties
    cell_size: Float[jax.Array, " cells"] = flax.struct.field(default=None)
    density: Float[jax.Array, " cells"] = flax.struct.field(default=None)
    point_mass: Float[jax.Array, " points"] = flax.struct.field(default=None)
    surface_point_id: Float[jax.Array, " surface_points"] = flax.struct.field(
        default=None
    )

    # dirichlet boundary condition
    dirichlet_index: Integer[jax.Array, " dirichlet"] = flax.struct.field(default=None)
    dirichlet_values: Float[jax.Array, " dirichlet"] = flax.struct.field(default=None)
    free_index: Integer[jax.Array, " free"] = flax.struct.field(default=None)

    # meta fields
    dim: int = flax.struct.field(pytree_node=False, default=3)
    mesh: pv.UnstructuredGrid = flax.struct.field(pytree_node=False, default=None)
    surface_pv: pv.PolyData = flax.struct.field(pytree_node=False, default=None)
    surface_wp: wp.Mesh = flax.struct.field(pytree_node=False, default=None)

    @classmethod
    def from_pyvista(cls, mesh: pv.UnstructuredGrid) -> Self:
        mesh = mesh.compute_cell_sizes()
        cells: Integer[jax.Array, "cells 4"] = jnp.asarray(
            mesh.cells_dict[pv.CellType.TETRA], int
        )
        cell_size: Float[jax.Array, " cells"] = jnp.asarray(
            mesh.cell_data["Volume"], float
        )
        self: Self = cls(cells=cells, cell_size=cell_size, mesh=mesh)
        if (density := mesh.cell_data.get("Density")) is not None:
            self = self.replace(density=jnp.asarray(density, float))
        return self

    @property
    def n_cells(self) -> int:
        return self.mesh.n_cells

    @property
    def n_dirichlet(self) -> int:
        return self.dirichlet_index.size

    @property
    def n_dof(self) -> int:
        return self.mesh.n_points * self.dim

    @property
    def n_free(self) -> int:
        return self.free_index.size

    @property
    def n_points(self) -> int:
        return self.mesh.n_points

    def with_displacement(
        self, displacement: Float[ArrayLike, "points dim"] | None = None
    ) -> Self:
        if displacement is None:
            return self
        displacement = jnp.asarray(displacement, float)
        displacement = math.broadcast_to(displacement, (self.n_points, self.dim))
        return self.replace(displacement=displacement)

    def with_force(self, force: Float[ArrayLike, "points dim"] | None = None) -> Self:
        if force is None:
            return self
        force = jnp.asarray(force, float)
        force = math.broadcast_to(force, (self.n_points, self.dim))
        return self.replace(force=force)

    def with_velocity(
        self, velocity: Float[ArrayLike, "points dim"] | None = None
    ) -> Self:
        if velocity is None:
            return self
        velocity = jnp.asarray(velocity, float)
        velocity = math.broadcast_to(velocity, (self.n_points, self.dim))
        return self.replace(velocity=velocity)

    def with_density(self, density: Float[ArrayLike, " cells"] | None = None) -> Self:
        if density is None:
            return self
        density = jnp.asarray(density, float)
        density = math.broadcast_to(density, (self.n_cells,))
        return self.replace(density=density)
