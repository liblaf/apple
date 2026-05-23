from typing import override

import attrs
import pyvista as pv
import torch
from jaxtyping import Integer
from torch import Tensor

from liblaf.apple.torch.fem.element import ElementTetra

from ._geometry import Geometry, _geometry_from_pyvista


@attrs.define
class GeometryTetra(Geometry):
    mesh: pv.UnstructuredGrid

    @property
    @override
    def element(self) -> ElementTetra:
        return ElementTetra()

    @property
    @override
    def cells_local(self) -> Integer[Tensor, "c a"]:
        return torch.tensor(self.mesh.cells_dict[pv.CellType.TETRA], dtype=torch.int32)  # ty:ignore[invalid-argument-type]


@_geometry_from_pyvista.register(pv.UnstructuredGrid)
def _(mesh: pv.UnstructuredGrid) -> GeometryTetra:
    return GeometryTetra(mesh=mesh)
