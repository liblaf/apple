from typing import Self

import einops
import jax.numpy as jnp
import pyvista as pv
from jaxtyping import Array, Float, Integer
from loguru import logger

from liblaf.apple import struct
from liblaf.apple.jax import math
from liblaf.apple.jax.sim.element import Element, ElementTetra
from liblaf.apple.jax.sim.quadrature import Scheme


@struct.pytree
class Region:
    element: Element = struct.field()
    mesh: pv.UnstructuredGrid = struct.field()
    quadrature: Scheme = struct.field()

    points: Float[Array, "p J"] = struct.array()
    cells: Integer[Array, "c a"] = struct.array()

    h: Float[Array, "q a"] = struct.array(default=None)
    dhdr: Float[Array, "q a J"] = struct.array(default=None)
    dXdr: Float[Array, "c q J J"] = struct.array(default=None)
    drdX: Float[Array, "c q J J"] = struct.array(default=None)
    dV: Float[Array, "c q"] = struct.array(default=None)
    dhdX: Float[Array, "c q a J"] = struct.array(default=None)

    @classmethod
    def from_pyvista(cls, mesh: pv.UnstructuredGrid) -> Self:
        element = ElementTetra()
        self: Self = cls(
            element=element,
            mesh=mesh,
            quadrature=element.quadrature,
            points=math.asarray(mesh.points),
            cells=math.asarray(mesh.cells_dict[pv.CellType.TETRA]),
        )
        self.compute_grad()
        return self

    def compute_grad(self) -> None:
        h: Float[Array, "q a"] = jnp.stack(
            [self.element.function(q) for q in self.quadrature.points]
        )
        dhdr: Float[Array, "q a J"] = jnp.stack(
            [self.element.gradient(q) for q in self.quadrature.points]
        )
        dXdr: Float[Array, "c q J J"] = einops.einsum(
            self.scatter(self.points), dhdr, "c a I, q a J -> c q I J"
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

    def deformation_gradient(self, x: Float[Array, "p J"]) -> Float[Array, "c q J J"]:
        grad: Float[Array, "c q J J"] = self.gradient(x)
        F: Float[Array, "c q J J"] = (
            grad + jnp.identity(3)[jnp.newaxis, jnp.newaxis, ...]
        )
        return F

    def gradient(
        self, x: Float[Array, " points *shape"]
    ) -> Float[Array, "c q *shape J"]:
        result: Float[Array, "c q *shape J"] = einops.einsum(
            self.scatter(x), self.dhdX, "c a ..., c q a J -> c q ... J"
        )
        return result

    def integrate(self, a: Float[Array, "c q *shape"]) -> Float[Array, " c *shape"]:
        return einops.einsum(a, self.dV, "c q ..., c q -> c ...")

    def scatter(self, x: Float[Array, " points *shape"]) -> Float[Array, "c a *shape"]:
        return x[self.cells]
