from typing import no_type_check

import warp as wp

from liblaf.apple.warp.typing import mat33, mat43


@wp.func
@no_type_check
def gradient(u: mat43, dhdX: mat43) -> mat33:
    r"""$\frac{\partial u}{\partial x}$."""
    return wp.transpose(u) @ dhdX


@wp.func
@no_type_check
def deformation_gradient(u: mat43, dhdX: mat43) -> mat33:
    r"""$F = \frac{\partial u}{\partial x} + I$."""
    return gradient(u, dhdX) + wp.identity(3, dtype=float)


@wp.func
@no_type_check
def deformation_gradient_jvp(dhdX: mat43, p: mat43) -> mat33:
    r"""$\frac{\partial F}{\partial x} p$."""
    return wp.transpose(p) @ dhdX


@wp.func
@no_type_check
def deformation_gradient_vjp(dhdX: mat43, p: mat33) -> mat43:
    r"""$\frac{\partial F}{\partial x}^T p$."""
    return dhdX @ wp.transpose(p)
