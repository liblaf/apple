from __future__ import annotations

import attrs
import jax.numpy as jnp
import pyvista as pv
from jaxtyping import Array, ArrayLike, Bool, DTypeLike, Float, Integer
from liblaf.peach import tree

from liblaf import grapes
from liblaf.apple.constants import DIRICHLET_MASK, DIRICHLET_VALUE, DOF_ID
from liblaf.apple.jax import math

from ._dirichlet import Dirichlet


def _default_mask(self: DirichletBuilder) -> Bool[Array, "p J"]:
    return jnp.empty((0, self.dim), dtype=bool)


def _default_values(self: DirichletBuilder) -> Float[Array, "p J"]:
    return jnp.empty((0, self.dim), dtype=float)


@tree.define
class DirichletBuilder:
    dim: int = tree.field(default=3)
    mask: Bool[Array, "p J"] = tree.field(
        default=attrs.Factory(_default_mask, takes_self=True)
    )
    values: Float[Array, "p J"] = tree.field(
        default=attrs.Factory(_default_values, takes_self=True)
    )

    def add(self, mesh: pv.DataSet) -> None:
        dof_id: Integer[Array, " p"] = math.asarray(
            grapes.getitem(mesh.point_data, DOF_ID), dtype=int
        )
        dirichlet_mask: Bool[Array, "p J"] = _broadcast_to(
            grapes.getitem(mesh.point_data, DIRICHLET_MASK),
            dtype=bool,
            shape=mesh.points.shape,
        )
        dirichlet_value: Float[Array, "p J"] = _broadcast_to(
            grapes.getitem(mesh.point_data, DIRICHLET_VALUE),
            dtype=float,
            shape=mesh.points.shape,
        )
        self.mask = self.mask.at[dof_id].set(dirichlet_mask)
        self.values = self.values.at[dof_id].set(dirichlet_value)

    def finish(self) -> Dirichlet:
        mask_flat: Bool[Array, " N"] = self.mask.flatten()
        index: Integer[Array, " dirichlet"]
        (index,) = jnp.nonzero(mask_flat)
        index_free: Integer[Array, " free"]
        (index_free,) = jnp.nonzero(~mask_flat)
        return Dirichlet(
            index=index,
            free_index=index_free,
            n_dofs=self.mask.size,
            values=self.values.flatten()[index],
        )

    def resize(self, n_points: int) -> None:
        if n_points <= self.mask.shape[0]:
            return
        pad_width: tuple[tuple[int, int], tuple[int, int]] = (
            (0, n_points - self.mask.shape[0]),
            (0, 0),
        )
        self.mask = jnp.pad(self.mask, pad_width)
        self.values = jnp.pad(self.values, pad_width)


def _broadcast_to(a: ArrayLike, *, dtype: DTypeLike, shape: tuple[int, int]) -> Array:
    a = math.asarray(a, dtype=dtype)
    if a.ndim == 1:
        return jnp.broadcast_to(a[:, jnp.newaxis], shape)
    if a.ndim == 2:
        return jnp.broadcast_to(a, shape)
    raise NotImplementedError
