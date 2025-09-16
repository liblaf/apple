from typing import Any, no_type_check

import warp as wp


@wp.func
@no_type_check
def g1(R: Any):
    r"""Gradient of $I_1$ w.r.t. $F$.

    $$
    g_1 = vec(R)
    $$

    Args:
        R (mat33): ...

    Returns:
        g1 (mat33): ...
    """
    return R


@wp.func
@no_type_check
def g2(F: Any):
    """Gradient of $I_2$ w.r.t. $F$.

    $$
    g_2 = 2 vec(F)
    $$

    Args:
        F (mat33): ...

    Returns:
        g2 (mat33): ...
    """
    return type(F[0, 0])(2.0) * F


@wp.func
@no_type_check
def g3(F: Any):
    r"""Gradient of $I_3$ w.r.t. $F$.

    $$
    g_3 = vec([f_1 \cp f_2, f_2 \cp f_0, f_0 \cp f_1])
    $$

    Args:
        F (mat33): ...

    Returns:
        g3 (mat33): ...
    """
    f0, f1, f2 = F[:, 0], F[:, 1], F[:, 2]
    return wp.matrix_from_cols(wp.cross(f1, f2), wp.cross(f2, f0), wp.cross(f0, f1))
