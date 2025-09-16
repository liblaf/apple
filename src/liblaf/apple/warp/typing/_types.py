from typing import Any

import warp.types as wpt

type Struct = Any

float_ = wpt.float64
int_ = wpt.int32
mat33 = wpt.matrix((3, 3), float_)
mat43 = wpt.matrix((4, 3), float_)
vec3 = wpt.vector(3, float_)
vec4i = wpt.vector(4, int_)
vec6 = wpt.vector(6, float_)


def floating_dtypes() -> list[type[wpt.float_base]]:
    return [wpt.float32, wpt.float64]
