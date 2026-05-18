import attrs
import jax.numpy as jnp
import numpy as np
import pyvista as pv
from jaxtyping import Array, Float, Integer

from liblaf import jarp
from liblaf.apple.common import FIXED_MASK, FIXED_VALUE, GLOBAL_POINT_ID

from ._dof_map import DofMap


@jarp.define
class DofMapBuilder:
    dim: int = jarp.static(default=3)

    def _default_fixed_mask(self) -> Integer[np.ndarray, "points dim"]:
        return np.empty((0, self.dim), np.bool)

    def _default_full_values(self) -> Float[np.ndarray, "points dim"]:
        return np.empty((0, self.dim))

    fixed_mask: Integer[np.ndarray, "points dim"] = jarp.field(
        default=attrs.Factory(_default_fixed_mask, takes_self=True)
    )
    full_values: Float[np.ndarray, "points dim"] = jarp.field(
        default=attrs.Factory(_default_full_values, takes_self=True)
    )

    @property
    def n_points(self) -> int:
        return self.fixed_mask.shape[0]

    def add_fixed(self, obj: pv.DataSet) -> None:
        global_point_id: Integer[np.ndarray, " points"] = obj.point_data[
            GLOBAL_POINT_ID.vtk
        ]
        self.fixed_mask[global_point_id] = obj.point_data[FIXED_MASK.vtk]
        self.full_values[global_point_id] = obj.point_data[FIXED_VALUE.vtk]

    def add_vertices(self, obj: pv.DataSet) -> None:
        obj.point_data[GLOBAL_POINT_ID.vtk] = np.arange(
            self.n_points, self.n_points + obj.n_points
        )
        self.fixed_mask = np.pad(
            self.fixed_mask, ((0, obj.n_points), (0, 0)), constant_values=False
        )
        self.full_values = np.pad(
            self.full_values, ((0, obj.n_points), (0, 0)), constant_values=0.0
        )

    def finalize(self) -> DofMap:
        fixed_indices: Integer[Array, " fixed"] = jnp.flatnonzero(self.fixed_mask)
        fixed_values: Float[Array, " fixed"] = jnp.asarray(
            self.full_values.flatten()[fixed_indices]
        )
        free_indices: Integer[Array, " free"] = jnp.flatnonzero(~self.fixed_mask)
        return DofMap(
            dim=self.dim,
            n_points=self.n_points,
            fixed_indices=fixed_indices,
            fixed_values=fixed_values,
            free_indices=free_indices,
        )
