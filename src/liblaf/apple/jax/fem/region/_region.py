import logging
from typing import Self

import einops
import jarp
import jax.numpy as jnp
import pyvista as pv
from jaxtyping import Array, Float, Integer

from liblaf.apple.jax.fem.element import Element
from liblaf.apple.jax.fem.geometry import Geometry
from liblaf.apple.jax.fem.quadrature import Scheme

logger = logging.getLogger(__name__)


@jarp.define
class Region:
    geometry: Geometry = jarp.field()
    quadrature: Scheme = jarp.field()

    h: Float[Array, "q a"] = jarp.array(default=None)
    dhdr: Float[Array, "q a J"] = jarp.array(default=None)
    dXdr: Float[Array, "c q J J"] = jarp.array(default=None)
    drdX: Float[Array, "c q J J"] = jarp.array(default=None)
    dV: Float[Array, "c q"] = jarp.array(default=None)
    dhdX: Float[Array, "c q a J"] = jarp.array(default=None)

    @classmethod
    def from_geometry(
        cls, geometry: Geometry, *, grad: bool = False, quadrature: Scheme | None = None
    ) -> Self:
        if quadrature is None:
            quadrature = geometry.element.quadrature
        self: Self = cls(geometry=geometry, quadrature=quadrature)
        if grad:
            self.compute_grad()
        return self

    @classmethod
    def from_pyvista(
        cls,
        mesh: pv.DataObject,
        *,
        grad: bool = False,
        quadrature: Scheme | None = None,
    ) -> Self:
        geometry: Geometry = Geometry.from_pyvista(mesh)
        self: Self = cls.from_geometry(geometry, grad=grad, quadrature=quadrature)
        return self

    @property
    def n_cells(self) -> int:
        return self.geometry.n_cells

    @property
    def cells_global(self) -> Integer[Array, "c a"]:
        return self.geometry.cells_global

    @property
    def cells_local(self) -> Integer[Array, "c a"]:
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
    def points(self) -> Float[Array, "p J"]:
        return self.geometry.points

    def compute_grad(self) -> None:
        h: Float[Array, "q a"] = jnp.stack(
            [self.element.function(q) for q in self.quadrature.points]
        )
        dhdr: Float[Array, "q a J"] = jnp.stack(
            [self.element.gradient(q) for q in self.quadrature.points]
        )
        dXdr: Float[Array, "c q J J"] = einops.einsum(
            self.points[self.cells_local], dhdr, "c a I, q a J -> c q I J"
        )
        drdX: Float[Array, "c q J J"] = jnp.linalg.inv(dXdr)
        dV: Float[Array, "c q"] = (
            jnp.linalg.det(dXdr) * self.quadrature.weights[jnp.newaxis, :]
        )
        if jnp.any(dV <= 0):
            logger.warning("dV <= 0")
        dhdX: Float[Array, "c q a J"] = einops.einsum(
            dhdr, drdX, "q a I, c q I J -> c q a J"
        )
        self.h = h
        self.dhdr = dhdr
        self.dXdr = dXdr
        self.drdX = drdX
        self.dV = dV
        self.dhdX = dhdX

    def deformation_gradient(self, u: Float[Array, "p J"]) -> Float[Array, "c q J J"]:
        grad: Float[Array, "c q J J"] = self.gradient(u)
        F: Float[Array, "c q J J"] = (
            grad + jnp.identity(3)[jnp.newaxis, jnp.newaxis, ...]
        )
        return F

    def gradient(
        self, u: Float[Array, " points *shape"]
    ) -> Float[Array, "c q *shape J"]:
        result: Float[Array, "c q *shape J"] = einops.einsum(
            self.scatter(u), self.dhdX, "c a ..., c q a J -> c q ... J"
        )
        return result

    def integrate(self, a: Float[Array, "c q *shape"]) -> Float[Array, " c *shape"]:
        return einops.einsum(a, self.dV, "c q ..., c q -> c ...")

    def scatter(self, u: Float[Array, " points *shape"]) -> Float[Array, "c a *shape"]:
        return u[self.cells_global]
