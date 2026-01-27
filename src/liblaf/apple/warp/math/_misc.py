from typing import Any

import warp as wp


@wp.func
def cw_max_4x(a: Any, b: Any) -> Any:
    return wp.matrix_from_rows(
        wp.max(a[0], b[0]), wp.max(a[1], b[1]), wp.max(a[2], b[2]), wp.max(a[3], b[3])
    )


@wp.func
def cw_square(a: Any) -> Any:
    return wp.cw_mul(a, a)


@wp.func
def fro_norm_square(M: Any) -> Any:
    r"""$\norm{M}_F^2$."""
    return wp.ddot(M, M)


@wp.func
def square(a: Any) -> Any:
    return a * a
