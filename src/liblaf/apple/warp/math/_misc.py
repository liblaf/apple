from typing import Any

import warp as wp


@wp.func
def cw_square(a: Any):
    return a * a


@wp.func
def frobenius_norm_square(M: Any):
    r"""$\norm{M}_F^2$.

    Args:
        M (matrix): ...

    Returns:
        (float): ...
    """
    return wp.ddot(M, M)


@wp.func
def square(a: Any):
    return a * a
