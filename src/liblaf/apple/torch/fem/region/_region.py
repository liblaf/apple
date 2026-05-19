import contextlib
import functools
import logging
from typing import Self

import attrs
import einops
import pyvista as pv
import torch
from jaxtyping import Float, Integer
from torch import Tensor

from liblaf.apple.torch.fem.element import Element
from liblaf.apple.torch.fem.geometry import Geometry
from liblaf.apple.torch.fem.quadrature import Scheme

logger: logging.Logger = logging.getLogger(__name__)


@attrs.define
class Region:
    geometry: Geometry
    quadrature: Scheme = attrs.field(default=None)

    @classmethod
    def from_geometry(
        cls, geometry: Geometry, *, quadrature: Scheme | None = None
    ) -> Self:
        if quadrature is None:
            with contextlib.suppress(NotImplementedError):
                quadrature: Scheme = geometry.element.quadrature
        self: Self = cls(geometry=geometry, quadrature=quadrature)
        return self

    @classmethod
    def from_pyvista(
        cls, mesh: pv.DataObject, *, quadrature: Scheme | None = None
    ) -> Self:
        geometry: Geometry = Geometry.from_pyvista(mesh)
        self: Self = cls.from_geometry(geometry, quadrature=quadrature)
        return self

    @property
    def n_cells(self) -> int:
        return self.geometry.n_cells

    @property
    def cells_global(self) -> Integer[Tensor, "c a"]:
        return self.geometry.cells_global

    @property
    def cells_local(self) -> Integer[Tensor, "c a"]:
        return self.geometry.cells_local

    @property
    def element(self) -> Element:
        return self.geometry.element

    @property
    def mesh(self) -> pv.DataSet:
        return self.geometry.mesh

    @property
    def point_data(self) -> pv.DataSetAttributes:
        return self.geometry.point_data

    @property
    def cell_data(self) -> pv.DataSetAttributes:
        return self.geometry.cell_data

    @property
    def points(self) -> Float[Tensor, "p J"]:
        return self.geometry.points

    @functools.cached_property
    def h(self) -> Float[Tensor, "q a"]:
        return torch.stack([self.element.function(q) for q in self.quadrature.points])

    @functools.cached_property
    def dhdr(self) -> Float[Tensor, "q a J"]:
        return torch.stack([self.element.gradient(q) for q in self.quadrature.points])

    @functools.cached_property
    def dXdr(self) -> Float[Tensor, "c q J J"]:
        return einops.einsum(
            self.points[self.cells_local], self.dhdr, "c a I, q a J -> c q I J"
        )

    @functools.cached_property
    def drdX(self) -> Float[Tensor, "c q J J"]:
        return torch.linalg.inv(self.dXdr)

    @functools.cached_property
    def dV(self) -> Float[Tensor, "c q"]:
        dV: Float[Tensor, "c q"] = (
            torch.linalg.det(self.dXdr) * self.quadrature.weights[torch.newaxis, :]
        )
        if torch.any(dV <= 0):
            logger.warning("dV <= 0")
        return dV

    @functools.cached_property
    def dhdX(self) -> Float[Tensor, "c q a J"]:
        return einops.einsum(self.dhdr, self.drdX, "q a I, c q I J -> c q a J")

    def deformation_gradient(self, u: Float[Tensor, "p J"]) -> Float[Tensor, "c q J J"]:
        grad: Float[Tensor, "c q J J"] = self.gradient(u)
        F: Float[Tensor, "c q J J"] = (
            grad + torch.eye(3)[torch.newaxis, torch.newaxis, ...]
        )
        return F

    def gradient(
        self, u: Float[Tensor, " points *shape"]
    ) -> Float[Tensor, "c q *shape J"]:
        result: Float[Tensor, "c q *shape J"] = einops.einsum(
            self.scatter(u), self.dhdX, "c a ..., c q a J -> c q ... J"
        )
        return result

    def integrate(self, a: Float[Tensor, "c q *shape"]) -> Float[Tensor, " c *shape"]:
        return einops.einsum(a, self.dV, "c q ..., c q -> c ...")

    def scatter(
        self, u: Float[Tensor, " points *shape"]
    ) -> Float[Tensor, "c a *shape"]:
        return u[self.cells_global]
