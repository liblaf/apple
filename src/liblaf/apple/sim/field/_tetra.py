from typing import Self, no_type_check, override

import jax
import warp as wp
from jaxtyping import Float

from liblaf.apple import func, utils
from liblaf.apple.typed.warp import mat33, mat43

from ._abc import Field, FieldGrad


class FieldTetra(Field):
    @property
    def dhdX_squeeze(self) -> Float[jax.Array, "cells a J"]:
        return self.dhdX.reshape(
            self.n_cells, self.element.n_points, self.quadrature.dim
        )

    @property
    @override
    @utils.jit
    def grad(self) -> FieldGrad:
        values: Float[jax.Array, " cells *dim"] = _gradient_warp(
            self.values_scatter, self.dhdX_squeeze
        )[0]
        return FieldGrad.from_region(
            self.region.grad, values=values, dim=(*self.dim, self.quadrature.dim)
        )

    @property
    @override
    @utils.jit
    def deformation_gradient(self) -> FieldGrad:
        values: Float[jax.Array, "cells 3 3"] = _deformation_gradient_warp(
            self.values_scatter, self.dhdX_squeeze
        )[0]
        return FieldGrad.from_region(
            self.region.grad, values=values, dim=(*self.dim, self.quadrature.dim)
        )

    @override
    @utils.jit
    def deformation_gradient_jvp(self, p: Self) -> FieldGrad:  # pyright: ignore[reportIncompatibleMethodOverride]
        values: Float[jax.Array, "cells 3 3"] = _deformation_gradient_jvp_warp(
            self.dhdX_squeeze, p.values_scatter
        )[0]
        return FieldGrad.from_region(
            self.region.grad, values=values, dim=(*self.dim, self.quadrature.dim)
        )

    @override
    @utils.jit
    def deformation_gradient_vjp(self, p: FieldGrad) -> Self:  # pyright: ignore[reportIncompatibleMethodOverride]
        values: Float[jax.Array, "cells a dim"] = _deformation_gradient_vjp_warp(
            self.dhdX_squeeze, p.values.squeeze()
        )[0]
        return self.with_values(self.region.gather(values))


@no_type_check
@utils.jax_kernel
def _gradient_warp(
    u: wp.array(dtype=mat43),
    dhdX: wp.array(dtype=mat43),
    grad: wp.array(dtype=mat33),
) -> None:
    tid = wp.tid()
    grad[tid] = func.gradient(u[tid], dhdX[tid])


@no_type_check
@utils.jax_kernel
def _deformation_gradient_warp(
    u: wp.array(dtype=mat43),
    dhdX: wp.array(dtype=mat43),
    F: wp.array(dtype=mat33),
) -> None:
    tid = wp.tid()
    F[tid] = func.deformation_gradient(u[tid], dhdX[tid])


@no_type_check
@utils.jax_kernel
def _deformation_gradient_jvp_warp(
    dhdX: wp.array(dtype=mat43),
    p: wp.array(dtype=mat43),
    results: wp.array(dtype=mat33),
) -> None:
    tid = wp.tid()
    results[tid] = func.deformation_gradient_jvp(dhdX[tid], p[tid])


@no_type_check
@utils.jax_kernel
def _deformation_gradient_vjp_warp(
    dhdX: wp.array(dtype=mat43),
    p: wp.array(dtype=mat33),
    results: wp.array(dtype=mat43),
) -> None:
    tid = wp.tid()
    results[tid] = func.deformation_gradient_vjp(dhdX[tid], p[tid])
