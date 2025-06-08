from typing import Self, no_type_check, override

import jax
import jax.numpy as jnp
import warp as wp
from jaxtyping import Float

from liblaf.apple import func, utils
from liblaf.apple.typed.warp import mat33, mat43

from ._abc import Field


class FieldTetra(Field):
    @property
    @override
    @utils.jit
    def grad(self) -> Float[jax.Array, "cells dim J=3"]:
        return _gradient_warp(self.values[self.cells], self.dh_dX)[0]

    @property
    @override
    @utils.jit
    def deformation_gradient(self) -> Float[jax.Array, "cells 3 3"]:
        return _deformation_gradient_warp(self.values[self.cells], self.dh_dX)[0]

    @override
    @utils.jit
    def deformation_gradient_jvp(self, p: Self) -> Float[jax.Array, "cells 3 3"]:
        return _deformation_gradient_jvp_warp(self.dh_dX, p.values[self.cells])[0]

    @override
    @utils.jit
    def deformation_gradient_vjp(
        self, p: Float[jax.Array, "cells 3 3"]
    ) -> Float[jax.Array, "cells a dim"]:
        p = jnp.asarray(p)
        return _deformation_gradient_vjp_warp(self.dh_dX, p)[0]


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
