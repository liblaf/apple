from typing import no_type_check

import warp as wp

from liblaf.apple.warp.typing import mat33


@wp.func
@no_type_check
def g1(*, R: mat33) -> mat33:
    return R


@wp.func
@no_type_check
def g2(*, F: mat33) -> mat33:
    return type(F[0, 0])(2.0) * F


@wp.func
@no_type_check
def g3(*, F: mat33) -> mat33:
    f0, f1, f2 = F[:, 0], F[:, 1], F[:, 2]
    return wp.matrix_from_cols(wp.cross(f1, f2), wp.cross(f2, f0), wp.cross(f0, f1))
