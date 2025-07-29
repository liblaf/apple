from typing import no_type_check

import warp as wp

from liblaf.apple.func import utils
from liblaf.apple.typed.warp import mat33, mat43, vec3

from ._deformation_gradient import deformation_gradient_vjp


@no_type_check
@wp.func
def h1_diag(dh_dX: mat43, g1: mat33) -> mat43:
    return utils.cw_square(deformation_gradient_vjp(dh_dX, g1))


@no_type_check
@wp.func
def h2_diag(dh_dX: mat43, g2: mat33) -> mat43:
    return utils.cw_square(deformation_gradient_vjp(dh_dX, g2))


@no_type_check
@wp.func
def h3_diag(dh_dX: mat43, g3: mat33) -> mat43:
    return utils.cw_square(deformation_gradient_vjp(dh_dX, g3))


@no_type_check
@wp.func
def h4_diag(dh_dX: mat43, *, lambdas: vec3, Q0: mat33, Q1: mat33, Q2: mat33) -> mat43:
    W0 = deformation_gradient_vjp(dh_dX, Q0)  # mat43
    W1 = deformation_gradient_vjp(dh_dX, Q1)  # mat43
    W2 = deformation_gradient_vjp(dh_dX, Q2)  # mat43
    return (
        lambdas[0] * utils.cw_square(W0)
        + lambdas[1] * utils.cw_square(W1)
        + lambdas[2] * utils.cw_square(W2)
    )


@no_type_check
@wp.func
def h5_diag(dh_dX: mat43) -> mat43:
    t0 = wp.length_sq(dh_dX[0])
    t1 = wp.length_sq(dh_dX[1])
    t2 = wp.length_sq(dh_dX[2])
    t3 = wp.length_sq(dh_dX[3])
    return 2.0 * wp.matrix_from_rows(
        vec3(t0, t0, t0), vec3(t1, t1, t1), vec3(t2, t2, t2), vec3(t3, t3, t3)
    )


@no_type_check
@wp.func
def h6_diag() -> mat43:
    return mat43()
