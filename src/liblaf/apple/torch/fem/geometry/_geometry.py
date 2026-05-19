import functools
from typing import Self, cast

import attrs
import pyvista as pv
import torch
from jaxtyping import Float, Integer
from torch import Tensor

from liblaf.apple.common import GLOBAL_POINT_ID
from liblaf.apple.torch.fem.element import Element


@attrs.define
class Geometry:
    mesh: pv.DataSet

    @classmethod
    def from_pyvista(cls, mesh: pv.DataObject) -> Self:
        return cast("Self", _geometry_from_pyvista(mesh))

    @property
    def element(self) -> Element:
        raise NotImplementedError

    @property
    def n_cells(self) -> int:
        return self.cells_local.shape[0]

    @property
    def cell_data(self) -> pv.DataSetAttributes:
        return self.mesh.cell_data

    @property
    def cells_global(self) -> Integer[Tensor, "c a"]:
        return self.global_point_id[self.cells_local]

    @property
    def cells_local(self) -> Integer[Tensor, "c a"]:
        raise NotImplementedError

    @property
    def global_point_id(self) -> Integer[Tensor, "p J"]:
        return torch.tensor(self.point_data[GLOBAL_POINT_ID.vtk], dtype=torch.int32)

    @property
    def point_data(self) -> pv.DataSetAttributes:
        return self.mesh.point_data

    @property
    def points(self) -> Float[Tensor, "p J"]:
        dtype: torch.dtype = torch.get_default_dtype()
        return torch.tensor(self.mesh.points, dtype=dtype)


@functools.singledispatch
def _geometry_from_pyvista(mesh: pv.DataObject) -> Geometry:
    raise TypeError(mesh)
