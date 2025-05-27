from typing import Self

import flax.struct
import jax
import jax.numpy as jnp
from jaxtyping import Float, Integer
from numpy.typing import ArrayLike

from liblaf.apple import elem, testing, utils

from ._domain import Domain
from ._geometry import Geometry


class Field(flax.struct.PyTreeNode):
    domain: Domain
    values: Float[jax.Array, "points dim"]
    dim: int = flax.struct.field(pytree_node=False, default=3)
    id: str = flax.struct.field(pytree_node=False, default="field")

    def __post_init__(self) -> None:
        testing.assert_shape(self.values, (self.n_points, self.dim))

    @property
    def cells(self) -> Integer[jax.Array, "cells 4"]:
        return self.domain.cells

    @property
    def n_cells(self) -> int:
        return self.domain.n_cells

    @property
    def n_dof(self) -> int:
        return self.dim * self.n_points

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
        return self.domain.dV

    # endregion FEM


class FieldSpec(flax.struct.PyTreeNode):
    id: str = flax.struct.field(pytree_node=False)
    domain: Domain
    dim: int = flax.struct.field(pytree_node=False)

    dirichlet_index: Float[jax.Array, " dirichlet"]
    dirichlet_values: Float[jax.Array, " dirichlet"]
    free_index: Float[jax.Array, " free"]

    def __post_init__(self) -> None:
        testing.assert_shape(self.dirichlet_values, self.dirichlet_index.shape)

    @classmethod
    def from_domain(
        cls,
        domain: Domain,
        dim: int = 3,
        dirichlet_index: Integer[ArrayLike, " dirichlet"] | None = None,
        dirichlet_mask: Float[ArrayLike, " DoF"] | None = None,
        dirichlet_values: Float[ArrayLike, " dirichlet"] | None = None,
        free_index: Float[ArrayLike, " free"] | None = None,
        free_mask: Float[ArrayLike, " DoF"] | None = None,
        id: str = "field",  # noqa: A002
    ) -> Self:
        n_dof: int = domain.n_points * dim
        if dirichlet_index is not None:
            dirichlet_index = jnp.asarray(dirichlet_index, dtype=int)
            dirichlet_mask = jnp.zeros((n_dof,), dtype=bool)
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
            free_mask = jnp.zeros((n_dof,), dtype=bool)
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
            free_index = jnp.arange(n_dof)

        if dirichlet_values is None:
            dirichlet_values = jnp.zeros(dirichlet_index.shape)
        dirichlet_values = jnp.asarray(dirichlet_values, dtype=float)
        dirichlet_values = jnp.broadcast_to(dirichlet_values, dirichlet_index.shape)

        return cls(
            id=id,
            domain=domain,
            dim=dim,
            dirichlet_index=dirichlet_index,
            dirichlet_values=dirichlet_values,
            free_index=free_index,
        )

    @property
    def geometry(self) -> Geometry:
        return self.domain.geometry

    @property
    def n_dirichlet(self) -> int:
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

    @utils.jit
    def make_field(self, free_values: Float[jax.Array, " free"] | None = None) -> Field:
        if free_values is None:
            free_values = jnp.zeros((self.n_free,))
        free_values = jnp.asarray(free_values)
        free_values = jnp.broadcast_to(free_values, (self.n_free,))
        values: Float[jax.Array, " DoF"] = jax.numpy.zeros(
            (self.n_dof,), dtype=free_values.dtype
        )
        values = values.at[self.dirichlet_index].set(self.dirichlet_values)
        values = values.at[self.free_index].set(free_values)
        return Field(
            domain=self.domain,
            values=values.reshape(self.n_points, self.dim),
            dim=self.dim,
            id=self.id,
        )

    @utils.jit
    def make_field_no_dirichlet(
        self, free_values: Float[jax.Array, " free"] | None = None
    ) -> Field:
        if free_values is None:
            free_values = jnp.zeros((self.n_free,))
        free_values = jnp.asarray(free_values)
        free_values = jnp.broadcast_to(free_values, (self.n_free,))
        values: Float[jax.Array, " DoF"] = jax.numpy.zeros(
            (self.n_dof,), dtype=free_values.dtype
        )
        values = values.at[self.free_index].set(free_values)
        return Field(
            domain=self.domain,
            values=values.reshape(self.n_points, self.dim),
            dim=self.dim,
            id=self.id,
        )
