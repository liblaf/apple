from typing import Any, no_type_check

import warp as wp

mat33 = Any
vec3 = Any


@wp.func
@no_type_check
def svd_rv(A: mat33) -> tuple[mat33, vec3, mat33]:
    U, s, V = wp.svd3(A)
    return U, s, V


@wp.func
@no_type_check
def polar_rv(A: mat33) -> tuple[mat33, mat33]:
    U, s, V = svd_rv(A)
    R = U @ wp.transpose(V)
    S = V @ wp.diag(s) @ wp.transpose(V)
    return R, S
