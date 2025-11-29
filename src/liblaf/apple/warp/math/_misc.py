from typing import Any

import warp as wp


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
