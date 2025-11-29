from typing import Any, no_type_check

import warp as wp

mat33 = Any


@wp.func
@no_type_check
def g1(R: mat33) -> mat33:
    r"""Gradient of $I_1$ w.r.t. $F$.

    $$
    g_1 = vec(R)
    $$
    """
    return R


@wp.func
@no_type_check
def g2(F: mat33) -> mat33:
    """Gradient of $I_2$ w.r.t. $F$.

    $$
    g_2 = 2 vec(F)
    $$
    """
    return F.dtype(2.0) * F


@wp.func
@no_type_check
def g3(F: mat33) -> mat33:
    r"""Gradient of $I_3$ w.r.t. $F$.

    $$
    g_3 = vec([f_1 \cp f_2, f_2 \cp f_0, f_0 \cp f_1])
    $$
    """
    f0, f1, f2 = F[:, 0], F[:, 1], F[:, 2]
    return wp.matrix_from_cols(wp.cross(f1, f2), wp.cross(f2, f0), wp.cross(f0, f1))
