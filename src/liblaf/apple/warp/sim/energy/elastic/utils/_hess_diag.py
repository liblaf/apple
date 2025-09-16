from typing import no_type_check

import warp as wp

from liblaf.apple.warp import math
from liblaf.apple.warp.typing import mat33, mat43, vec3

from ._deformation_gradient import deformation_gradient_vjp


@wp.func
@no_type_check
def h1_diag(*, dhdX: mat43, g1: mat33):
    """$diag(h_1)$.

    Args:
        dhdX (mat43): ...
        g1 (mat33): ...

    Returns:
        (mat43): ...
    """
    return math.cw_square(deformation_gradient_vjp(dhdX, g1))


@wp.func
@no_type_check
def h2_diag(*, dhdX: mat43, g2: mat33):
    """$diag(h_2)$.

    Args:
        dhdX (mat43): ...
        g2 (mat33): ...

    Returns:
        (mat43): ...
    """
    return math.cw_square(deformation_gradient_vjp(dhdX, g2))


@wp.func
@no_type_check
def h3_diag(*, dhdX: mat43, g3: mat33):
    """$diag(h_3)$.

    Args:
        dhdX (mat43): ...
        g3 (mat33): ...

    Returns:
        (mat43): ...
    """
    return math.cw_square(deformation_gradient_vjp(dhdX, g3))


@wp.func
@no_type_check
def h4_diag(*, dhdX: mat43, lambdas: vec3, Q0: mat33, Q1: mat33, Q2: mat33):
    """$diag(h_4)$.

    Args:
        dhdX (mat43): ...
        lambdas (vec3): ...
        Q0 (mat33): ...
        Q1 (mat33): ...
        Q2 (mat33): ...

    Returns:
        (mat43): ...
    """
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
def h5_diag(*, dhdX: mat43):
    """$diag(h_5)$.

    Args:
        dhdX (mat43): ...

    Returns:
        (mat43): ...
    """
    t0 = wp.length_sq(dhdX[0])
    t1 = wp.length_sq(dhdX[1])
    t2 = wp.length_sq(dhdX[2])
    t3 = wp.length_sq(dhdX[3])
    return type(dhdX[0, 0])(2.0) * wp.matrix_from_rows(
        wp.vector(t0, t0, t0),
        wp.vector(t1, t1, t1),
        wp.vector(t2, t2, t2),
        wp.vector(t3, t3, t3),
    )


@wp.func
@no_type_check
def h6_diag():
    """$diag(h_6)$."""
    raise NotImplementedError
