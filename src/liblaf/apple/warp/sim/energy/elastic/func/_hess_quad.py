from typing import no_type_check

import warp as wp

from liblaf.apple.warp import math
from liblaf.apple.warp.typing.generics import mat33, mat43, vec3

from ._deformation_gradient import deformation_gradient_jvp


@wp.func
@no_type_check
def h1_quad(p: mat43, dhdX: mat43, g1: mat33):
    """$p^T h_1 p$.

    Args:
        p (mat43): ...
        dhdX (mat43): ...
        g1 (mat33): ...

    Returns:
        (float): ...
    """
    return math.square(wp.ddot(deformation_gradient_jvp(dhdX, p), g1))


@wp.func
@no_type_check
def h2_quad(p: mat43, dhdX: mat43, g2: mat33):
    """$p^T h_2 p$.

    Args:
        p (mat43): ...
        dhdX (mat43): ...
        g2 (mat33): ...

    Returns:
        (float): ...
    """
    return math.square(wp.ddot(deformation_gradient_jvp(dhdX, p), g2))


@wp.func
@no_type_check
def h3_quad(p: mat43, dhdX: mat43, g3: mat33):
    """$p^T h_3 p$.

    Args:
        p (mat43): ...
        dhdX (mat43): ...
        g3 (mat33): ...

    Returns:
        (float): ...
    """
    return math.square(wp.ddot(deformation_gradient_jvp(dhdX, p), g3))


@wp.func
@no_type_check
def h4_quad(p: mat43, dhdX: mat43, lambdas: vec3, Q0: mat33, Q1: mat33, Q2: mat33):
    """$p^T h_4 p$.

    Args:
        p (mat43): ...
        dhdX (mat43): ...
        lambdas (vec3): ...
        p (mat43): ...
        Q0 (mat33): ...
        Q1 (mat33): ...
        Q2 (mat33): ...

    Returns:
        (float): ...
    """
    dFdx_p = deformation_gradient_jvp(dhdX, p)  # mat33
    return (
        lambdas[0] * math.square(wp.ddot(Q0, dFdx_p))
        + lambdas[1] * math.square(wp.ddot(Q1, dFdx_p))
        + lambdas[2] * math.square(wp.ddot(Q2, dFdx_p))
    )


@wp.func
@no_type_check
def h5_quad(p: mat43, dhdX: mat43):
    """$p^T h_5 p$.

    Args:
        p (mat43): ...
        dhdX (mat43): ...

    Returns:
        (float): ...
    """
    dFdx_p = deformation_gradient_jvp(dhdX, p)  # mat33
    return type(dhdX[0, 0])(2.0) * math.frobenius_norm_square(dFdx_p)


@wp.func
@no_type_check
def h6_quad(p: mat43, dhdX: mat43, F: mat33):
    """$p^T h_6 p$.

    Args:
        p (mat43): ...
        dhdX (mat43): ...
        F (mat33): ...

    Returns:
        (float): ...
    """
    dFdx_p = deformation_gradient_jvp(dhdX, p)  # mat33
    f0 = F[:, 0]
    f1 = F[:, 1]
    f2 = F[:, 2]
    p0 = dFdx_p[:, 0]
    p1 = dFdx_p[:, 1]
    p2 = dFdx_p[:, 2]
    return (
        wp.dot(p0, wp.cross(f1, p2) - wp.cross(f2, p1))
        + wp.dot(p1, wp.cross(f2, p0) - wp.cross(f0, p2))
        + wp.dot(p2, wp.cross(f0, p1) - wp.cross(f1, p0))
    )
