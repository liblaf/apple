from typing import no_type_check

import warp as wp

from liblaf.apple.func import utils
from liblaf.apple.typed.warp import mat33


@no_type_check
@wp.func
def g1(R: mat33) -> mat33:
    return R


@no_type_check
@wp.func
def g2(F: mat33) -> mat33:
    return 2.0 * F


@no_type_check
@wp.func
def g3(F: mat33) -> mat33:
    f0, f1, f2 = utils.col(F, 0), utils.col(F, 1), utils.col(F, 2)
    return wp.matrix_from_cols(wp.cross(f1, f2), wp.cross(f2, f0), wp.cross(f0, f1))
