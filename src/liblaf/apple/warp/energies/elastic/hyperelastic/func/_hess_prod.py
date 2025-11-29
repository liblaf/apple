from typing import Any, no_type_check

import warp as wp

from ._deformation import deformation_gradient_jvp, deformation_gradient_vjp

mat33 = Any
mat43 = Any
vec3 = Any


@wp.func
@no_type_check
def h1_prod(p: mat43, dhdX: mat43, g1: mat33) -> mat43:
    """$h_1 p$."""
    dFdxT_g = deformation_gradient_vjp(dhdX, g1)  # mat33
    return wp.ddot(dFdxT_g, p) * dFdxT_g


@wp.func
@no_type_check
def h2_prod(p: mat43, dhdX: mat43, g2: mat33) -> mat43:
    """$h_2 p$."""
    dFdxT_g = deformation_gradient_vjp(dhdX, g2)  # mat33
    return wp.ddot(dFdxT_g, p) * dFdxT_g


@wp.func
@no_type_check
def h3_prod(p: mat43, dhdX: mat43, g3: mat33) -> mat43:
    """$h_3 p$."""
    dFdxT_g = deformation_gradient_vjp(dhdX, g3)  # mat33
    return wp.ddot(dFdxT_g, p) * dFdxT_g


@wp.func
@no_type_check
def h4_prod(
    p: mat43, dhdX: mat43, lambdas: vec3, Q0: mat33, Q1: mat33, Q2: mat33
) -> mat43:
    """$h_4 p$."""
    dFdxT_q0 = deformation_gradient_vjp(dhdX, Q0)  # mat33
    dFdxT_q1 = deformation_gradient_vjp(dhdX, Q1)  # mat33
    dFdxT_q2 = deformation_gradient_vjp(dhdX, Q2)  # mat33
    return (
        lambdas[0] * wp.ddot(dFdxT_q0, p) * dFdxT_q0
        + lambdas[1] * wp.ddot(dFdxT_q1, p) * dFdxT_q1
        + lambdas[2] * wp.ddot(dFdxT_q2, p) * dFdxT_q2
    )


@wp.func
@no_type_check
def h5_prod(p: mat43, dhdX: mat43) -> mat43:
    """$h_5 p$."""
    dFdx_p = deformation_gradient_jvp(dhdX, p)  # mat33
    return type(dhdX[0, 0])(2.0) * deformation_gradient_vjp(dhdX, dFdx_p)


@wp.func
@no_type_check
def h6_prod(p: mat43, dhdX: mat43, F: mat33) -> mat43:
    """$h_6 p$."""
    dFdx_p = deformation_gradient_jvp(dhdX, p)  # mat33
    f0 = F[:, 0]  # vec3
    f1 = F[:, 1]  # vec3
    f2 = F[:, 2]  # vec3
    p0 = dFdx_p[:, 0]  # vec3
    p1 = dFdx_p[:, 1]  # vec3
    p2 = dFdx_p[:, 2]  # vec3
    Hp = wp.matrix_from_cols(
        wp.cross(f1, p2) - wp.cross(f2, p1),
        wp.cross(f2, p0) - wp.cross(f0, p2),
        wp.cross(f0, p1) - wp.cross(f1, p0),
    )  # mat33
    return deformation_gradient_vjp(dhdX, Hp)
