from typing import Self

import einops
import jax
import jax.numpy as jnp
import pyvista as pv
from jaxtyping import Float
from numpy.typing import ArrayLike

from liblaf.apple import struct
from liblaf.apple.sim import element as _e
from liblaf.apple.sim import geometry as _g
from liblaf.apple.sim import quadrature as _q


class Region(struct.Node):
    geometry: _g.Geometry = struct.static(default=None)
    element: _e.Element = struct.static(default=None)
    quadrature: _q.Scheme = struct.static(default=None)

    # grad
    h: Float[jax.Array, "q a"] = struct.array(default=None)
    dhdr: Float[jax.Array, "q a J"] = struct.array(default=None)
    dXdr: Float[jax.Array, "c q I J"] = struct.array(default=None)
    drdX: Float[jax.Array, "c q J I"] = struct.array(default=None)
    dV: Float[jax.Array, "c q"] = struct.array(default=None)
    dhdX: Float[jax.Array, "c q a I"] = struct.array(default=None)

    @classmethod
    def from_geometry(
        cls,
        geometry: _g.Geometry,
        element: _e.Element | None = None,
        quadrature: _q.Scheme | None = None,
        *,
        grad: bool = True,
        hess: bool = False,
    ) -> Self:
        self: Self = cls(geometry=geometry, element=element, quadrature=quadrature)  # pyright: ignore[reportArgumentType]
        if grad:
            self = self.with_grad()
        if hess:
            raise NotImplementedError
        return self

    @property
    def area(self) -> Float[jax.Array, " cells"]:
        return self.geometry.area

    @property
    def boundary(self) -> "RegionBoundary":
        return RegionBoundary.from_region(self)

    @property
    def cells(self) -> Float[jax.Array, "cells a"]:
        return self.geometry.cells

    @property
    def grad(self) -> "RegionGrad":
        return RegionGrad.from_region(self)

    @property
    def mesh(self) -> pv.DataSet:
        return self.geometry.mesh

    @property
    def n_cells(self) -> int:
        return self.geometry.n_cells

    @property
    def n_points(self) -> int:
        return self.geometry.n_points

    @property
    def points(self) -> Float[jax.Array, "points J"]:
        return self.geometry.points

    @property
    def volume(self) -> Float[jax.Array, " cells"]:
        return self.geometry.volume

    @property
    def weights(self) -> Float[jax.Array, " q"]:
        return self.quadrature.weights

    def gather(
        self, values: Float[ArrayLike, "cells a *dim"], /
    ) -> Float[jax.Array, " points *dim"]:
        values = jnp.asarray(values)
        return jax.ops.segment_sum(
            einops.rearrange(values, "c a ... -> (c a) ..."),
            einops.rearrange(self.cells, "c a -> (c a)"),
            num_segments=self.n_points,
        )

    def gradient(
        self, values: Float[ArrayLike, " points *dim"], /
    ) -> Float[jax.Array, "cells q *dim J"]:
        values = jnp.asarray(values)
        return einops.einsum(
            self.scatter(values), self.dhdX, "c a ..., c q a J -> c q ... J"
        )

    def integrate(
        self, values: Float[ArrayLike, " cells q *dim"], /
    ) -> Float[jax.Array, "*dim"]:
        values = jnp.asarray(values)
        return einops.einsum(values, self.dV, "c q ..., c q -> ...")

    def scatter(
        self, values: Float[ArrayLike, " points *dim"], /
    ) -> Float[jax.Array, "cells a *dim"]:
        values = jnp.asarray(values)
        return values[self.cells]

    def with_grad(self) -> Self:
        if self.dhdX is not None:
            return self
        new: Self = self.evolve(h=self._compute_h())
        new = new.evolve(dhdr=new._compute_dhdr())  # noqa: SLF001
        new = new.evolve(dXdr=new._compute_dXdr())  # noqa: SLF001
        new = new.evolve(drdX=new._compute_drdX())  # noqa: SLF001
        new = new.evolve(dV=new._compute_dV())  # noqa: SLF001
        new = new.evolve(dhdX=new._compute_dhdX())  # noqa: SLF001
        return new

    def _compute_h(self) -> Float[jax.Array, "q a"]:
        """Element shape function array `h_qa` of shape function `a` evaluated at quadrature point `q`."""
        return jnp.asarray([self.element.function(q) for q in self.quadrature.points])

    def _compute_dhdr(self) -> Float[jax.Array, "q a J"]:
        """Partial derivative of element shape function array `dhdr_qaJ` with shape function `a` w.r.t. natural element coordinate `J` evaluated at quadrature point `q` for every cell `c` (geometric gradient or **Jacobian** transformation between `X` and `r`)."""
        return jnp.asarray([self.element.gradient(q) for q in self.quadrature.points])

    def _compute_dXdr(self) -> Float[jax.Array, "c q I J"]:
        """Geometric gradient `dXdr_cqIJ` as partial derivative of undeformed coordinate `I` w.r.t. natural element coordinate `J` evaluated at quadrature point `q` for every cell `c` (geometric gradient or **Jacobian** transformation between `X` and `r`)."""
        return einops.einsum(
            self.points[self.cells], self.dhdr, "c a I, q a J -> c q I J"
        )

    def _compute_drdX(self) -> Float[jax.Array, "c q J I"]:
        """Inverse of `dXdr`."""
        return jnp.linalg.inv(self._compute_dXdr())

    def _compute_dV(self) -> Float[jax.Array, "c q"]:
        """Numeric differential volume element as product of determinant of geometric gradient `dV_qc = det(dXdr)_qc w_q` and quadrature weight `w_q`, evaluated at quadrature point `q` for every cell `c`."""
        return jnp.linalg.det(self.dXdr) * self.quadrature.weights

    def _compute_dhdX(self) -> Float[jax.Array, "c q a J"]:
        """Partial derivative of element shape functions `dhdX_cqaJ` of shape function `a` w.r.t. undeformed coordinate `J` evaluated at quadrature point `q` for every cell `c`."""
        return einops.einsum(self.dhdr, self.dXdr, "q a I, c q I J -> c q a J")


class RegionBoundary(Region):
    is_view: bool = struct.class_var(default=True, init=False)

    @classmethod
    def from_region(cls, region: Region) -> Self:
        return cls(refs=(region,), geometry=region.geometry.boundary)


class RegionGrad(Region):
    is_view: bool = struct.class_var(default=True, init=False)

    @classmethod
    def from_region(cls, region: Region) -> Self:
        return cls(
            refs=(region,),
            geometry=region.geometry,
            element=region.element,
            quadrature=region.quadrature,
            h=region.h,
            dhdr=region.dhdr,
            dXdr=region.dXdr,
            drdX=region.drdX,
            dV=region.dV,
            dhdX=region.dhdX,
        )
