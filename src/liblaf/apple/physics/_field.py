from typing import Self

import flax.struct
import jax
import jax.numpy as jnp
from jaxtyping import Bool, Float, Integer
from numpy.typing import ArrayLike

from liblaf.apple import elem, utils

from ._domain import Domain
from ._geometry import Geometry


class Field(flax.struct.PyTreeNode):
    domain: Domain
    dim: int = flax.struct.field(pytree_node=False, default=3, kw_only=True)
    id: str = flax.struct.field(pytree_node=False, default="displacement", kw_only=True)

    values: Float[jax.Array, "points dim"] = flax.struct.field(
        default=None, kw_only=True
    )
    forces: Float[jax.Array, "points dim"] = flax.struct.field(
        default=None, kw_only=True
    )
    velocities: Float[jax.Array, "points dim"] = flax.struct.field(
        default=None, kw_only=True
    )

    dirichlet_index: Float[jax.Array, " dirichlet"] = flax.struct.field(
        default=None, kw_only=True
    )
    dirichlet_values: Float[jax.Array, " dirichlet"] = flax.struct.field(
        default=None, kw_only=True
    )
    free_index: Float[jax.Array, " free"] = flax.struct.field(
        default=None, kw_only=True
    )

    @classmethod
    def from_domain(
        cls,
        domain: Domain,
        dim: int = 3,
        id: str = "displacement",  # noqa: A002
    ) -> Self:
        return cls(id=id, domain=domain, dim=dim)

    @property
    def cells(self) -> Integer[jax.Array, "cells 4"]:
        return self.domain.cells

    @property
    def free_values(self) -> Float[jax.Array, " free"]:
        values: Float[jax.Array, " DoF"] = self.values.ravel()
        if self.free_index is None:
            return values
        return values[self.free_index]

    @property
    def geometry(self) -> Geometry:
        return self.domain.geometry

    @property
    def n_cells(self) -> int:
        return self.domain.n_cells

    @property
    def n_dirichlet(self) -> int:
        if self.dirichlet_index is None:
            return 0
        return self.dirichlet_index.size

    @property
    def n_dof(self) -> int:
        return self.dim * self.n_points

    @property
    def n_free(self) -> int:
        return self.n_dof - self.n_dirichlet

    @property
    def n_points(self) -> int:
        return self.domain.n_points

    @property
    def points(self) -> Float[jax.Array, "points 3"]:
        return self.domain.points

    # region FEM

    @property
    def deformation_gradient(self) -> Float[jax.Array, "cells 3 3"]:
        return elem.tetra.deformation_gradient(
            self.values[self.cells], self.domain.dh_dX
        )

    @property
    def dh_dX(self) -> Float[jax.Array, "cells 4 3"]:
        return self.domain.dh_dX

    @property
    def dV(self) -> Float[jax.Array, " cells"]:
        return self.domain.cell_sizes

    # endregion FEM

    def with_dirichlet(
        self,
        dirichlet_index: Integer[ArrayLike, " dirichlet"] | None = None,
        dirichlet_mask: Bool[ArrayLike, " DoF"] | None = None,
        dirichlet_values: Float[ArrayLike, " dirichlet"] | None = None,
        free_index: Float[ArrayLike, " free"] | None = None,
        free_mask: Bool[ArrayLike, " DoF"] | None = None,
    ) -> Self:
        if dirichlet_index is not None:
            dirichlet_index = jnp.asarray(dirichlet_index, dtype=int)
            dirichlet_mask = jnp.zeros((self.n_dof,), dtype=bool)
            dirichlet_mask = dirichlet_mask.at[dirichlet_index].set(True)
            free_mask = ~dirichlet_mask
            (free_index,) = jnp.nonzero(free_mask)
        elif dirichlet_mask is not None:
            dirichlet_mask = jnp.asarray(dirichlet_mask, dtype=bool)
            (dirichlet_index,) = jnp.nonzero(dirichlet_mask)
            free_mask = ~dirichlet_mask
            (free_index,) = jnp.nonzero(free_mask)
        elif free_index is not None:
            free_index = jnp.asarray(free_index, dtype=int)
            free_mask = jnp.zeros((self.n_dof,), dtype=bool)
            free_mask = free_mask.at[free_index].set(True)
            dirichlet_mask = ~free_mask
            (dirichlet_index,) = jnp.nonzero(dirichlet_mask)
        elif free_mask is not None:
            free_mask = jnp.asarray(free_mask, dtype=bool)
            (free_index,) = jnp.nonzero(free_mask)
            dirichlet_mask = ~free_mask
            (dirichlet_index,) = jnp.nonzero(dirichlet_mask)
        else:
            dirichlet_index = jnp.empty((0,), dtype=int)
            free_index = jnp.arange(self.n_dof)

        if dirichlet_values is None:
            dirichlet_values = jnp.zeros(dirichlet_index.shape)
        dirichlet_values = jnp.asarray(dirichlet_values, dtype=float)
        dirichlet_values = jnp.broadcast_to(dirichlet_values, dirichlet_index.shape)

        return self.replace(
            dirichlet_index=dirichlet_index,
            dirichlet_values=dirichlet_values,
            free_index=free_index,
        )

    @utils.jit
    def with_forces(self, forces: Float[ArrayLike, "points dim"] | None = None) -> Self:
        forces: Float[jax.Array, "points dim"] = make_values(
            values=forces, n_points=self.n_points, dim=self.dim
        )
        return self.replace(forces=forces)

    @utils.jit(static_argnames=("dirichlet",))
    def with_free_values(
        self,
        free_values: Float[ArrayLike, " free"] | None = None,
        *,
        dirichlet: bool = True,
    ) -> Self:
        if free_values is None:
            free_values = jnp.zeros((self.n_free,), dtype=float)
        free_values = jnp.asarray(free_values, dtype=float)
        free_values = jnp.broadcast_to(free_values, (self.n_free,))
        values: Float[jax.Array, " DoF"]
        if free_values.size == self.n_dof:
            values = free_values
        else:
            values = jnp.zeros((self.n_dof,), dtype=float)
            if dirichlet:
                values = values.at[self.dirichlet_index].set(self.dirichlet_values)
            values = values.at[self.free_index].set(free_values)
        return self.replace(values=values.reshape(self.n_points, self.dim))

    @utils.jit
    def with_velocities(
        self, velocities: Float[ArrayLike, "points dim"] | None = None
    ) -> Self:
        velocities: Float[jax.Array, "points dim"] = make_values(
            values=velocities, n_points=self.n_points, dim=self.dim
        )
        return self.replace(velocities=velocities)

    @utils.jit
    def with_values(self, values: Float[ArrayLike, "points dim"] | None = None) -> Self:
        values: Float[jax.Array, "points dim"] = make_values(
            values=values, n_points=self.n_points, dim=self.dim
        )
        return self.replace(values=values)


def make_values(
    values: Float[ArrayLike, "points dim"] | None = None, *, n_points: int, dim: int = 3
) -> Float[jax.Array, "points dim"]:
    if values is None:
        return jnp.zeros((n_points, dim), dtype=float)
    values = jnp.asarray(values, dtype=float)
    if values.size < n_points * dim:
        return jnp.broadcast_to(values, (n_points, dim))
    return values.reshape(n_points, dim)
