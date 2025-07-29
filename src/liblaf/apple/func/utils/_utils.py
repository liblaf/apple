from typing import Any

import warp as wp


@wp.func
def cw_square(A: Any):  # noqa: ANN202
    return wp.cw_mul(A, A)


@wp.func
def frobenius_norm_square(a: Any) -> float:
    return wp.ddot(a, a)


@wp.func
def square(a: Any):  # noqa: ANN202
    return a * a
