from typing import Any

import warp as wp

mat33 = Any
scalar = Any


@wp.func
def I1(S: mat33) -> scalar:
    r"""$I_1$.

    $$
    I_1 = \operatorname{tr}(R^T F) = \operatorname{tr}(S)
    $$
    """
    return wp.trace(S)


@wp.func
def I2(F: mat33) -> scalar:
    r"""$I_2$.

    $$
    I_2 = I_C = \|F\|_F^2
    $$
    """
    return wp.ddot(F, F)


@wp.func
def I3(F: mat33) -> scalar:
    r"""$I_3$.

    $$
    I_3 = J = \det(F)
    $$
    """
    return wp.determinant(F)
