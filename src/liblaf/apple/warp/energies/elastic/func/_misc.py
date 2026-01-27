from typing import Any, no_type_check

import warp as wp

int_ = Any
vec3 = Any
vec4i = Any
vec6 = Any
mat33 = Any
mat43 = Any


@wp.func
def dRdF_vjp(M: mat33, lambdas: vec3, Q0: mat33, Q1: mat33, Q2: mat33) -> mat33:
    return (
        lambdas[0] * wp.ddot(M, Q0) * Q0
        + lambdas[1] * wp.ddot(M, Q1) * Q1
        + lambdas[2] * wp.ddot(M, Q2) * Q2
    )


@wp.func
def get_cell_displacements(
    u: wp.array1d(dtype=vec3),  # (points,)
    cell: vec4i,
) -> mat43:
    return wp.matrix_from_rows(u[cell[0]], u[cell[1]], u[cell[2]], u[cell[3]])  # mat43


@wp.func
@no_type_check
def lambdas(sigma: vec3, *, clamp: bool = True) -> vec3:
    _2 = sigma.dtype(2.0)
    if clamp:
        lambda0 = _2 / wp.max(sigma[0] + sigma[1], _2)
        lambda1 = _2 / wp.max(sigma[1] + sigma[2], _2)
        lambda2 = _2 / wp.max(sigma[2] + sigma[0], _2)
    else:
        lambda0 = _2 / (sigma[0] + sigma[1])
        lambda1 = _2 / (sigma[1] + sigma[2])
        lambda2 = _2 / (sigma[2] + sigma[0])
    return wp.vector(lambda0, lambda1, lambda2)


@wp.func
@no_type_check
def make_activation_mat33(activation: vec6) -> mat33:
    return wp.identity(3, activation.dtype) + wp.matrix_from_rows(
        wp.vector(activation[0], activation[3], activation[4]),
        wp.vector(activation[3], activation[1], activation[5]),
        wp.vector(activation[4], activation[5], activation[2]),
    )


@wp.func
@no_type_check
def Qs(U: mat33, V: mat33) -> tuple[mat33, mat33, mat33]:
    _2 = U.dtype(2.0)
    _sqrt2 = wp.sqrt(_2)
    U0 = U[:, 0]
    U1 = U[:, 1]
    U2 = U[:, 2]
    V0 = V[:, 0]
    V1 = V[:, 1]
    V2 = V[:, 2]
    Q0 = (wp.outer(U1, V0) - wp.outer(U0, V1)) / _sqrt2
    Q1 = (wp.outer(U1, V2) - wp.outer(U2, V1)) / _sqrt2
    Q2 = (wp.outer(U0, V2) - wp.outer(U2, V0)) / _sqrt2
    return Q0, Q1, Q2
