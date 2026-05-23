from typing import override

import attrs
import pyvista as pv
import torch
from jaxtyping import Integer
from torch import Tensor

from ._geometry import Geometry, _geometry_from_pyvista


@attrs.define
class GeometryTriangle(Geometry):
    mesh: pv.PolyData

    @property
    @override
    def cells_local(self) -> Integer[Tensor, "c a"]:
        return torch.tensor(self.mesh.regular_faces)


@_geometry_from_pyvista.register(pv.PolyData)
def _(mesh: pv.PolyData) -> GeometryTriangle:
    mesh: pv.PolyData = mesh.triangulate()
    return GeometryTriangle(mesh=mesh)
