import warp as wp


@wp.func
def frobenius_norm_square(a: wp.mat33) -> float:
    return wp.ddot(a, a)
