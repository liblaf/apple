from typing import Any, no_type_check

import warp as wp


@wp.func
@no_type_check
def lambdas(sigma: Any, *, clamp: bool = True):
    """...

    Args:
        sigma (vec3): ...
        clamp: ...

    Returns:
        lambdas (vec3): ...
    """
    _2 = type(sigma[0])(2.0)
    lambda0 = _2 / (sigma[0] + sigma[1])
    lambda1 = _2 / (sigma[1] + sigma[2])
    lambda2 = _2 / (sigma[2] + sigma[0])
    if clamp:
        _0 = type(sigma[0])(0.0)
        _1 = type(sigma[0])(1.0)
        lambda0 = wp.clamp(lambda0, _0, _1)
        lambda1 = wp.clamp(lambda1, _0, _1)
        lambda2 = wp.clamp(lambda2, _0, _1)
    return wp.vector(lambda0, lambda1, lambda2)


@wp.func
@no_type_check
def make_activation_mat33(activation: Any):
    """...

    Args:
        activation (vec6): ...

    Returns:
        mat33: ...
    """
    return wp.identity(3, type(activation[0])) + wp.matrix_from_rows(
        wp.vector(activation[0], activation[3], activation[4]),
        wp.vector(activation[3], activation[1], activation[5]),
        wp.vector(activation[4], activation[5], activation[2]),
    )


@wp.func
@no_type_check
def Qs(U: Any, V: Any):
    """...

    Args:
        U (mat33): ...
        V (mat33): ...

    Returns:
        Q0 (mat33): ...
        Q1 (mat33): ...
        Q2 (mat33): ...
    """
    _2 = type(U[0, 0])(2.0)
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
