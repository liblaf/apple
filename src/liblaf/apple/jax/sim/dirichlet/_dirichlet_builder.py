from collections.abc import Sequence

import attrs
import jax.numpy as jnp
import pyvista as pv
from jaxtyping import Array, ArrayLike, Bool, DTypeLike, Float, Integer

from liblaf.apple.jax import math, tree

from ._dirichlet import Dirichlet


def _default_mask(self: "DirichletBuilder") -> Bool[Array, "p J"]:
    return jnp.empty((0, self.dim), dtype=bool)


def _default_values(self: "DirichletBuilder") -> Float[Array, "p J"]:
    return jnp.empty((0, self.dim), dtype=float)


@tree.pytree
class DirichletBuilder:
    dim: int = tree.field(default=3)
    mask: Bool[Array, "p J"] = tree.field(
        default=attrs.Factory(_default_mask, takes_self=True)
    )
    values: Float[Array, "p J"] = tree.field(
        default=attrs.Factory(_default_values, takes_self=True)
    )

    def add(self, mesh: pv.DataSet) -> None:
        point_id: Integer[Array, " p"] = math.asarray(
            mesh.point_data["point-id"], dtype=int
        )
        dirichlet_mask: Bool[Array, "p J"] = _broadcast_to(
            mesh.point_data["dirichlet-mask"], dtype=bool, shape=mesh.points.shape
        )
        dirichlet_values: Float[Array, "p J"] = _broadcast_to(
            mesh.point_data["dirichlet-values"], dtype=float, shape=mesh.points.shape
        )
        self.mask = self.mask.at[point_id].set(dirichlet_mask)
        self.values = self.values.at[point_id].set(dirichlet_values)

    def finish(self) -> Dirichlet:
        index: Sequence[Integer[Array, " dirichlet"]] = jnp.nonzero(self.mask)
        return Dirichlet(index=index, values=self.values[index])

    def resize(self, n_points: int) -> None:
        if n_points <= self.mask.shape[0]:
            return
        new_mask: Bool[Array, "p J"] = jnp.zeros((n_points, self.dim), dtype=bool)
        new_mask = new_mask.at[: self.mask.shape[0]].set(self.mask)
        self.mask = new_mask
        new_values: Float[Array, "p J"] = jnp.zeros((n_points, self.dim), dtype=float)
        new_values = new_values.at[: self.values.shape[0]].set(self.values)
        self.values = new_values


def _broadcast_to(a: ArrayLike, *, dtype: DTypeLike, shape: tuple[int, int]) -> Array:
    a = math.asarray(a, dtype=dtype)
    if a.ndim == 1:
        return jnp.broadcast_to(a[:, jnp.newaxis], shape)
    if a.ndim == 2:
        return jnp.broadcast_to(a, shape)
    raise NotImplementedError
