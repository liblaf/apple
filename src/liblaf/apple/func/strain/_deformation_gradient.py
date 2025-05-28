from typing import no_type_check

import warp as wp

from liblaf.apple.typed.warp import mat33, mat43


@no_type_check
@wp.func
def gradient(u: mat43, dh_dX: mat43) -> mat33:
    r"""$\frac{\partial u}{\partial x}$."""
    return wp.transpose(u) @ dh_dX


@no_type_check
@wp.func
def deformation_gradient(u: mat43, dh_dX: mat43) -> mat33:
    r"""$F = \frac{\partial u}{\partial x} + I$."""
    return gradient(u, dh_dX) + wp.identity(3, dtype=float)


@no_type_check
@wp.func
def deformation_gradient_jvp(dh_dX: mat43, p: mat43) -> mat33:
    r"""$\frac{\partial F}{\partial x} p$."""
    return wp.transpose(p) @ dh_dX


@no_type_check
@wp.func
def deformation_gradient_vjp(dh_dX: mat43, p: mat33) -> mat43:
    r"""$\frac{\partial F}{\partial x}^T p$."""
    return dh_dX @ wp.transpose(p)
