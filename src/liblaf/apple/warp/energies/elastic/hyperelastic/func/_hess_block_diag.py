from typing import Any, no_type_check

import warp as wp

from . import _misc
from ._deformation import deformation_gradient_vjp

mat43 = Any
mat33 = Any
vec3 = Any


@wp.func
@no_type_check
def outer_product_block_diag(v: mat43) -> tuple[mat33, mat33, mat33, mat33]:
    v0 = v[0]
    v1 = v[1]
    v2 = v[2]
    v3 = v[3]
    return (
        wp.outer(v0, v0),
        wp.outer(v1, v1),
        wp.outer(v2, v2),
        wp.outer(v3, v3),
    )


@wp.func
@no_type_check
def h1_block_diag(dhdX: mat43, g1: mat33) -> tuple[mat33, mat33, mat33, mat33]:
    """Block diagonal of $h_1$."""
    v = deformation_gradient_vjp(dhdX, g1)
    return outer_product_block_diag(v)


@wp.func
@no_type_check
def h2_block_diag(dhdX: mat43, g2: mat33) -> tuple[mat33, mat33, mat33, mat33]:
    """Block diagonal of $h_2$."""
    v = deformation_gradient_vjp(dhdX, g2)
    return outer_product_block_diag(v)


@wp.func
@no_type_check
def h3_block_diag(dhdX: mat43, g3: mat33) -> tuple[mat33, mat33, mat33, mat33]:
    """Block diagonal of $h_3$."""
    v = deformation_gradient_vjp(dhdX, g3)
    return outer_product_block_diag(v)


@wp.func
@no_type_check
def h4_block_diag(
    dhdX: mat43, U: mat33, sigma: vec3, V: mat33, *, clamp: bool = True
) -> tuple[mat33, mat33, mat33, mat33]:
    """Block diagonal of $h_4$."""
    lambdas = _misc.lambdas(sigma, clamp=clamp)
    Q0, Q1, Q2 = _misc.Qs(U, V)
    W0 = deformation_gradient_vjp(dhdX, Q0)
    W1 = deformation_gradient_vjp(dhdX, Q1)
    W2 = deformation_gradient_vjp(dhdX, Q2)

    H0_0 = wp.outer(W0[0], W0[0])
    H0_1 = wp.outer(W0[1], W0[1])
    H0_2 = wp.outer(W0[2], W0[2])
    H0_3 = wp.outer(W0[3], W0[3])

    H1_0 = wp.outer(W1[0], W1[0])
    H1_1 = wp.outer(W1[1], W1[1])
    H1_2 = wp.outer(W1[2], W1[2])
    H1_3 = wp.outer(W1[3], W1[3])

    H2_0 = wp.outer(W2[0], W2[0])
    H2_1 = wp.outer(W2[1], W2[1])
    H2_2 = wp.outer(W2[2], W2[2])
    H2_3 = wp.outer(W2[3], W2[3])

    return (
        lambdas[0] * H0_0 + lambdas[1] * H1_0 + lambdas[2] * H2_0,
        lambdas[0] * H0_1 + lambdas[1] * H1_1 + lambdas[2] * H2_1,
        lambdas[0] * H0_2 + lambdas[1] * H1_2 + lambdas[2] * H2_2,
        lambdas[0] * H0_3 + lambdas[1] * H1_3 + lambdas[2] * H2_3,
    )


@wp.func
@no_type_check
def h5_block_diag(dhdX: mat43) -> tuple[mat33, mat33, mat33, mat33]:
    """Block diagonal of $h_5$."""
    t0 = wp.length_sq(dhdX[0])
    t1 = wp.length_sq(dhdX[1])
    t2 = wp.length_sq(dhdX[2])
    t3 = wp.length_sq(dhdX[3])

    Id = wp.identity(3, dtype=dhdX.dtype)

    return (
        dhdX.dtype(2.0) * t0 * Id,
        dhdX.dtype(2.0) * t1 * Id,
        dhdX.dtype(2.0) * t2 * Id,
        dhdX.dtype(2.0) * t3 * Id,
    )


@wp.func
@no_type_check
def h6_block_diag(dhdX: mat43, F: mat33) -> tuple[mat33, mat33, mat33, mat33]:  # noqa: ARG001
    """Block diagonal of $h_6$."""
    Z = wp.matrix(shape=(3, 3), dtype=dhdX.dtype)
    return (Z, Z, Z, Z)
