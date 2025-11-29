from typing import Any, no_type_check

import warp as wp

from liblaf.apple.warp import math

from ._deformation import deformation_gradient_vjp

mat43 = Any
mat33 = Any
vec3 = Any


@wp.func
@no_type_check
def h1_diag(dhdX: mat43, g1: mat33) -> mat43:
    """$diag(h_1)$."""
    return math.cw_square(deformation_gradient_vjp(dhdX, g1))


@wp.func
@no_type_check
def h2_diag(dhdX: mat43, g2: mat33) -> mat43:
    """$diag(h_2)$."""
    return math.cw_square(deformation_gradient_vjp(dhdX, g2))


@wp.func
@no_type_check
def h3_diag(dhdX: mat43, g3: mat33) -> mat43:
    """$diag(h_3)$."""
    return math.cw_square(deformation_gradient_vjp(dhdX, g3))


@wp.func
@no_type_check
def h4_diag(dhdX: mat43, lambdas: vec3, Q0: mat33, Q1: mat33, Q2: mat33) -> mat43:
    """$diag(h_4)$."""
    W0 = deformation_gradient_vjp(dhdX, Q0)  # mat43
    W1 = deformation_gradient_vjp(dhdX, Q1)  # mat43
    W2 = deformation_gradient_vjp(dhdX, Q2)  # mat43
    return (
        lambdas[0] * math.cw_square(W0)
        + lambdas[1] * math.cw_square(W1)
        + lambdas[2] * math.cw_square(W2)
    )


@wp.func
@no_type_check
def h5_diag(dhdX: mat43) -> mat43:
    """$diag(h_5)$."""
    t0 = wp.length_sq(dhdX[0])
    t1 = wp.length_sq(dhdX[1])
    t2 = wp.length_sq(dhdX[2])
    t3 = wp.length_sq(dhdX[3])
    return dhdX.dtype(2.0) * wp.matrix_from_rows(
        wp.vector(t0, t0, t0),
        wp.vector(t1, t1, t1),
        wp.vector(t2, t2, t2),
        wp.vector(t3, t3, t3),
    )


@wp.func
@no_type_check
def h6_diag(dhdX: mat43, F: mat33) -> mat43:  # noqa: ARG001
    """$diag(h_6)$."""
    return wp.matrix(shape=(4, 3), dtype=dhdX.dtype)
