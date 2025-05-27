import warp as wp

from liblaf.apple.typed.warp import mat33


def I1(S: mat33) -> float:
    r"""$I_1$.

    $$
    I_1 = \operatorname{tr}(R^T F) = \operatorname{tr}(S)
    $$
    """
    return wp.trace(S)


def I2(F: mat33) -> float:
    r"""$I_2$.

    $$
    I_2 = I_C = \|F\|_F^2
    $$
    """
    return wp.ddot(F, F)


def I3(F: mat33) -> float:
    r"""$I_3$.

    $$
    I_3 = J = \det(F)
    $$
    """
    return wp.determinant(F)
