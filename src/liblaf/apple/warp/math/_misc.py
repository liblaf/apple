import warp as wp

from liblaf.apple.warp.typing import float_, mat33


@wp.func
def cw_square(a: mat33) -> mat33:
    return a * a


@wp.func
def frobenius_norm_square(a: mat33) -> float_:
    return wp.ddot(a, a)


@wp.func
def square(a: float_) -> float_:
    return a * a
