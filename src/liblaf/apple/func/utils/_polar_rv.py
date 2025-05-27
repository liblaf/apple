from typing import no_type_check

import warp as wp


@no_type_check
@wp.func
def svd_rv(A: wp.mat33):  # noqa: ANN202
    U = wp.mat33()
    sigma = wp.vec3()
    V = wp.mat33()
    wp.svd3(A, U, sigma, V)
    return U, sigma, V


@no_type_check
@wp.func
def polar_rv(A: wp.mat33):  # noqa: ANN202
    U, sigma, V = svd_rv(A)
    R = U @ wp.transpose(V)
    S = V @ wp.diag(sigma) @ wp.transpose(V)
    return R, S
