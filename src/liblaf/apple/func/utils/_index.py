from typing import Any, no_type_check

import warp as wp

from liblaf.apple.typed.warp import mat33, vec9


@no_type_check
@wp.func
def col(A: Any, i: Any):  # noqa: ANN202
    return wp.transpose(A)[i]


@no_type_check
@wp.func
def vec(A: mat33) -> vec9:
    return vec9(
        A[0, 0], A[0, 1], A[0, 2], A[1, 0], A[1, 1], A[1, 2], A[2, 0], A[2, 1], A[2, 2]
    )
