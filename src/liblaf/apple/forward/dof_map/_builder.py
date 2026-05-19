import attrs
import numpy as np
import pyvista as pv
import torch
from jaxtyping import Bool, Float, Integer
from torch import Tensor

from liblaf.apple.common import FIXED_MASK, FIXED_VALUE, GLOBAL_POINT_ID

from ._dof_map import DofMap


@attrs.define
class DofMapBuilder:
    dim: int = attrs.field(default=3)

    def _default_fixed_mask(self) -> Integer[np.ndarray, "points dim"]:
        return np.empty((0, self.dim), np.bool)

    def _default_full_values(self) -> Float[np.ndarray, "points dim"]:
        return np.empty((0, self.dim))

    fixed_mask: Integer[np.ndarray, "points dim"] = attrs.field(
        default=attrs.Factory(_default_fixed_mask, takes_self=True)
    )
    full_values: Float[np.ndarray, "points dim"] = attrs.field(
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
        fixed_mask: Bool[Tensor, "points dim"] = torch.as_tensor(self.fixed_mask)
        full_values: Float[Tensor, "points dim"] = torch.as_tensor(self.full_values)
        fixed_indices: Integer[Tensor, " fixed"] = torch.nonzero(
            fixed_mask.flatten()
        ).squeeze(dim=-1)
        fixed_values: Float[Tensor, " fixed"] = torch.as_tensor(
            full_values.flatten()[fixed_indices]
        )
        free_indices: Integer[Tensor, " free"] = torch.nonzero(
            ~fixed_mask.flatten()
        ).squeeze(dim=-1)
        return DofMap(
            dim=self.dim,
            n_points=self.n_points,
            fixed_indices=fixed_indices,
            fixed_values=fixed_values,
            free_indices=free_indices,
        )
