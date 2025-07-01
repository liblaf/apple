from typing import no_type_check

import warp as wp

from liblaf.apple.func import utils
from liblaf.apple.typed.warp import mat33, mat43

from ._deformation_gradient import deformation_gradient_jvp


@no_type_check
@wp.func
def h1_quad(p: mat43, dh_dX: mat43, g1: mat33) -> float:
    return utils.square(wp.ddot(deformation_gradient_jvp(dh_dX, p), g1))


@no_type_check
@wp.func
def h2_quad(p: mat43, dh_dX: mat43, g2: mat33) -> float:
    return utils.square(wp.ddot(deformation_gradient_jvp(dh_dX, p), g2))


@no_type_check
@wp.func
def h3_quad(p: mat43, dh_dX: mat43, g3: mat33) -> float:
    return utils.square(wp.ddot(deformation_gradient_jvp(dh_dX, p), g3))


@no_type_check
@wp.func
def h4_quad(
    p: mat43, dh_dX: mat43, *, lambdas: wp.vec3, Q0: mat33, Q1: mat33, Q2: mat33
) -> float:
    dFdx_p = deformation_gradient_jvp(dh_dX, p)  # mat33
    return (
        lambdas[0] * utils.square(wp.ddot(Q0, dFdx_p))
        + lambdas[1] * utils.square(wp.ddot(Q1, dFdx_p))
        + lambdas[2] * utils.square(wp.ddot(Q2, dFdx_p))
    )


@no_type_check
@wp.func
def h5_quad(p: mat43, dh_dX: mat43) -> float:
    dFdx_p = deformation_gradient_jvp(dh_dX, p)  # mat33
    return 2.0 * utils.frobenius_norm_square(dFdx_p)


@no_type_check
@wp.func
def h6_quad(p: mat43, F: mat33, dh_dX: mat43) -> float:
    dFdx_p = deformation_gradient_jvp(dh_dX, p)  # mat33
    f0 = utils.col(F, 0)
    f1 = utils.col(F, 1)
    f2 = utils.col(F, 2)
    p0 = utils.col(dFdx_p, 0)
    p1 = utils.col(dFdx_p, 1)
    p2 = utils.col(dFdx_p, 2)
    return (
        wp.dot(p0, wp.cross(f1, p2) - wp.cross(f2, p1))
        + wp.dot(p1, wp.cross(f2, p0) - wp.cross(f0, p2))
        + wp.dot(p2, wp.cross(f0, p1) - wp.cross(f1, p0))
    )
