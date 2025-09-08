# ruff: noqa: F811

from typing import no_type_check

import warp as wp
import warp.types as wpt


@wp.func
@no_type_check
def svd_rv(A: wpt.mat33f) -> tuple[wpt.mat33f, wpt.vec3f, wpt.mat33f]:
    U, s, V = wp.svd3(A)
    return U, s, V


@wp.func
@no_type_check
def svd_rv(A: wpt.mat33d) -> tuple[wpt.mat33d, wpt.vec3d, wpt.mat33d]:
    U, s, V = wp.svd3(A)
    return U, s, V


@wp.func
@no_type_check
def polar_rv(A: wpt.mat33f) -> tuple[wpt.mat33f, wpt.mat33f]:
    U, s, V = svd_rv(A)
    R = U @ wp.transpose(V)
    S = V @ wp.diag(s) @ wp.transpose(V)
    return R, S


@wp.func
@no_type_check
def polar_rv(A: wpt.mat33d) -> tuple[wpt.mat33d, wpt.mat33d]:
    U, s, V = svd_rv(A)
    R = U @ wp.transpose(V)
    S = V @ wp.diag(s) @ wp.transpose(V)
    return R, S
