from typing import Any, no_type_check

import warp as wp

vec3 = Any
mat33 = Any


@wp.func
@no_type_check
def svd_rv(F: mat33) -> tuple[mat33, vec3, mat33]:
    U, sigma, V = wp.svd3(F)
    return U, sigma, V


@wp.func
@no_type_check
def polar_rv(F: mat33) -> tuple[mat33, mat33]:
    U, sigma, V = svd_rv(F)
    R = U @ wp.transpose(V)
    S = V @ wp.diag(sigma) @ wp.transpose(V)
    return R, S
