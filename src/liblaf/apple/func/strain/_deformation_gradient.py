from typing import no_type_check

import warp as wp

from liblaf.apple.typed.warp import mat43


@no_type_check
@wp.func
def deformation_gradient_jvp(dh_dX: mat43, p: mat43) -> wp.mat33:
    r"""$\frac{\partial F}{\partial x} p$."""
    return wp.transpose(p) @ dh_dX


@no_type_check
@wp.func
def deformation_gradient_vjp(dh_dX: mat43, p: wp.mat33) -> mat43:
    r"""$\frac{\partial F}{\partial x}^T p$."""
    return dh_dX @ wp.transpose(p)
