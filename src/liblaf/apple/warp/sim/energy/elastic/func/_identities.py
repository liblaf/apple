import warp as wp

from liblaf.apple.warp.typing.generics import mat33


@wp.func
def I1(S: mat33):
    r"""$I_1$.

    $$
    I_1 = \operatorname{tr}(R^T F) = \operatorname{tr}(S)
    $$

    Args:
        S (mat33): ...

    Returns:
        I1 (float): ...
    """
    return wp.trace(S)


@wp.func
def I2(F: mat33):
    r"""$I_2$.

    $$
    I_2 = I_C = \|F\|_F^2
    $$

    Args:
        F (mat33): ...

    Returns:
        I2 (float): ...
    """
    return wp.ddot(F, F)


@wp.func
def I3(F: mat33):
    r"""$I_3$.

    $$
    I_3 = J = \det(F)
    $$

    Args:
        F (mat33): ...

    Returns:
        I3 (float): ...
    """
    return wp.determinant(F)
