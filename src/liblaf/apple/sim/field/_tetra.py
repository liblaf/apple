from typing import Self, no_type_check, override

import jax
import warp as wp
from jaxtyping import Float

from liblaf.apple import func, utils
from liblaf.apple.typed.warp import mat33, mat43

from ._abc import Field


class FieldTetraCell(Field): ...


class FieldTetraPoint(Field):
    @property
    @override
    @utils.jit
    def grad(self) -> FieldTetraCell:
        values: Float[jax.Array, " cells *dim"] = _gradient_warp(
            self.values_scatter, self.dh_dX
        )[0]
        return FieldTetraCell.from_space(
            self.space.grad, values=values, dim=(*self.dim, 3)
        )

    @property
    @override
    @utils.jit
    def deformation_gradient(self) -> FieldTetraCell:
        values: Float[jax.Array, "cells 3 3"] = _deformation_gradient_warp(
            self.values_scatter, self.dh_dX
        )[0]
        return FieldTetraCell.from_space(
            self.space.grad, values=values, dim=(*self.dim, 3)
        )

    @override
    @utils.jit
    def deformation_gradient_jvp(self, p: Self) -> FieldTetraCell:  # pyright: ignore[reportIncompatibleMethodOverride]
        values: Float[jax.Array, "cells 3 3"] = _deformation_gradient_jvp_warp(
            self.dh_dX, p.values_scatter
        )[0]
        return FieldTetraCell.from_space(
            self.space.grad, values=values, dim=(*self.dim, 3)
        )

    @override
    @utils.jit
    def deformation_gradient_vjp(self, p: FieldTetraCell) -> Self:  # pyright: ignore[reportIncompatibleMethodOverride]
        values: Float[jax.Array, "cells a dim"] = _deformation_gradient_vjp_warp(
            self.dh_dX, p.values_scatter
        )[0]
        return self.evolve(values=self.space.gather(values=values))


@no_type_check
@utils.jax_kernel
def _gradient_warp(
    u: wp.array(dtype=mat43),
    dh_dX: wp.array(dtype=mat43),
    grad: wp.array(dtype=mat33),
) -> None:
    tid = wp.tid()
    grad[tid] = func.gradient(u[tid], dh_dX[tid])


@no_type_check
@utils.jax_kernel
def _deformation_gradient_warp(
    u: wp.array(dtype=mat43),
    dh_dX: wp.array(dtype=mat43),
    F: wp.array(dtype=mat33),
) -> None:
    tid = wp.tid()
    F[tid] = func.deformation_gradient(u[tid], dh_dX[tid])


@no_type_check
@utils.jax_kernel
def _deformation_gradient_jvp_warp(
    dh_dX: wp.array(dtype=mat43),
    p: wp.array(dtype=mat43),
    results: wp.array(dtype=mat33),
) -> None:
    tid = wp.tid()
    results[tid] = func.deformation_gradient_jvp(dh_dX[tid], p[tid])


@no_type_check
@utils.jax_kernel
def _deformation_gradient_vjp_warp(
    dh_dX: wp.array(dtype=mat43),
    p: wp.array(dtype=mat33),
    results: wp.array(dtype=mat43),
) -> None:
    tid = wp.tid()
    results[tid] = func.deformation_gradient_vjp(dh_dX[tid], p[tid])
