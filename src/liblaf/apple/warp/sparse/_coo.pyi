import warp as wp

from liblaf.apple.warp.typing import float_

class Coo2d:
    data: wp.array(dtype=float_)
    coords: wp.array(dtype=wp.vec2i)
