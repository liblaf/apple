from typing import Self

import einops
import jax
import jax.numpy as jnp
import pyvista as pv
from jaxtyping import Float, Integer

from liblaf.apple import struct, utils
from liblaf.apple.sim.element import Element
from liblaf.apple.sim.geometry import Geometry
from liblaf.apple.sim.quadrature import Scheme


@struct.pytree
class Region(struct.PyTreeMixin):
    geometry: Geometry = struct.data(default=None)
    quadrature: Scheme = struct.data(default=None)

    h: Float[jax.Array, "q a"] = struct.array(default=None)
    dhdr: Float[jax.Array, "q a J"] = struct.array(default=None)
    dXdr: Float[jax.Array, "c q J J"] = struct.array(default=None)
    drdX: Float[jax.Array, "c q J J"] = struct.array(default=None)
    dV: Float[jax.Array, "c q"] = struct.array(default=None)
    dhdX: Float[jax.Array, "c q a J"] = struct.array(default=None)

    @classmethod
    def from_pyvista(
        cls, mesh: pv.DataSet, quadrature: Scheme | None = None, *, grad: bool = True
    ) -> Self:
        geometry: Geometry = Geometry.from_pyvista(mesh)
        return cls.from_geometry(geometry, quadrature=quadrature, grad=grad)

    @classmethod
    def from_geometry(
        cls, geometry: Geometry, quadrature: Scheme | None = None, *, grad: bool = True
    ) -> Self:
        if quadrature is None:
            quadrature = geometry.quadrature
        self: Self = cls(geometry=geometry, quadrature=quadrature)
        if grad:
            self = self.with_grad()
        return self

    # region Structure

    @property
    def element(self) -> Element:
        return self.geometry.element

    # endregion Structure

    # region Numbers

    @property
    def dim(self) -> int:
        return self.geometry.dim

    @property
    def n_cells(self) -> int:
        return self.geometry.n_cells

    @property
    def n_points(self) -> int:
        return self.geometry.n_points

    # endregion Numbers

    # region Arrays

    @property
    @utils.validate
    def cells(self) -> Integer[jax.Array, "{self.n_cells} {self.element.n_points}"]:
        return self.geometry.cells

    @property
    @utils.validate
    def points(self) -> Float[jax.Array, "{self.n_points} {self.dim}"]:
        return self.geometry.points

    # endregion Arrays

    # region Operator

    @utils.jit_method(inline=True)
    def deformation_gradient(
        self, x: Float[jax.Array, "points J"]
    ) -> Float[
        jax.Array, "{self.n_cells} {self.quadrature.n_points} {self.dim} {self.dim}"
    ]:
        result: Float[jax.Array, "c q J J"] = self.gradient(x)
        result += jnp.identity(self.dim, dtype=result.dtype)[
            jnp.newaxis, jnp.newaxis, ...
        ]
        return result

    @utils.jit_method(inline=True)
    def gradient(
        self, x: Float[jax.Array, "points *dim"]
    ) -> Float[jax.Array, "{self.n_cells} {self.quadrature.n_points} *dim {self.dim}"]:
        result: Float[jax.Array, "c q *dim J"] = einops.einsum(
            self.scatter(x), self.dhdX, "c a ..., c q a J -> c q ... J"
        )
        return result

    @utils.jit_method(inline=True)
    def scatter(
        self, x: Float[jax.Array, "points *dim"]
    ) -> Float[jax.Array, "{self.n_cells} {self.element.n_points} *dim"]:
        return x[self.cells]

    # endregion Operator

    # region Gradient

    @utils.jit_method()
    def with_grad(self) -> Self:
        h: Float[jax.Array, "q a"] = jnp.asarray(
            [self.element.function(q) for q in self.quadrature.points]
        )
        dhdr: Float[jax.Array, "q a J"] = jnp.asarray(
            [self.element.gradient(q) for q in self.quadrature.points]
        )
        dXdr: Float[jax.Array, "c q J J"] = einops.einsum(
            self.scatter(self.points), dhdr, "c a I, q a J -> c q I J"
        )
        drdX: Float[jax.Array, "c q J J"] = jnp.linalg.inv(dXdr)
        dV: Float[jax.Array, "c q"] = (
            jnp.linalg.det(dXdr) * self.quadrature.weights[jnp.newaxis, :]
        )
        dhdX: Float[jax.Array, "c q a J"] = einops.einsum(
            dhdr, drdX, "q a I, c q I J -> c q a J"
        )
        return self.evolve(h=h, dhdr=dhdr, dXdr=dXdr, drdX=drdX, dV=dV, dhdX=dhdX)

    # endregion Gradient
