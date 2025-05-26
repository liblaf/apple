from typing import Any, no_type_check

import warp as wp


@no_type_check
@wp.func
def col(A: Any, i: Any):  # noqa: ANN202
    return wp.transpose(A)[i]
