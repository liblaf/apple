from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np
import pyvista as pv
from jaxtyping import Array, ArrayLike, Bool, Float, Integer
from liblaf.peach import tree

from liblaf.apple.constants import DIRICHLET_MASK, DIRICHLET_VALUE, POINT_ID

from ._dirichlet import Dirichlet


@tree.define
class DirichletBuilder:
    mask: Bool[np.ndarray, "points dim"]
    value: Float[np.ndarray, "points dim"]

    def __init__(self, dim: int = 3) -> None:
        mask: Bool[np.ndarray, "points dim"] = np.empty((0, dim), bool)
        value: Float[np.ndarray, "points dim"] = np.empty((0, dim))
        self.__attrs_init__(mask=mask, value=value)  # pyright: ignore[reportAttributeAccessIssue]

    @property
    def dim(self) -> int:
        return self.mask.shape[1]

    @property
    def n_points(self) -> int:
        return self.mask.shape[0]

    def add_pyvista(self, obj: pv.DataSet) -> None:
        point_id = obj.point_data[POINT_ID]
        self.resize(point_id.max() + 1)
        dirichlet_mask: Bool[Array, "points dim"] = self._left_broadcast_to(
            obj.point_data[DIRICHLET_MASK], obj.n_points
        )
        dirichlet_value: Float[Array, "points dim"] = self._left_broadcast_to(
            obj.point_data[DIRICHLET_VALUE], obj.n_points
        )
        self.mask[point_id] = dirichlet_mask
        self.value[point_id] = dirichlet_value

    def finalize(self) -> Dirichlet:
        mask: Bool[Array, "points dim"] = jnp.asarray(self.mask)
        dirichlet_index: Integer[Array, " dirichlet"] = jnp.flatnonzero(mask)
        return Dirichlet(
            dim=self.dim,
            dirichlet_index=dirichlet_index,
            dirichlet_value=jnp.asarray(self.value.flat[dirichlet_index]),
            free_index=jnp.flatnonzero(~mask),
            n_points=self.n_points,
        )

    def resize(self, n_points: int) -> None:
        pad_after: int = n_points - self.n_points
        if pad_after <= 0:
            return
        self.mask = np.pad(self.mask, ((0, pad_after), (0, 0)), constant_values=False)
        self.value = np.pad(self.value, ((0, pad_after), (0, 0)), constant_values=0.0)

    def _left_broadcast_to(self, arr: ArrayLike, n_points: int) -> Array:
        shape: Sequence[int] = (n_points, self.dim)
        arr = jnp.asarray(arr)
        arr = jnp.reshape(arr, arr.shape + (1,) * (len(shape) - arr.ndim))
        return jnp.broadcast_to(arr, shape)
