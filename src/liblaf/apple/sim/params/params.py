from jaxtyping import Array, Float

from liblaf.apple import struct


class GlobalParams(struct.PyTree):
    time_step: Float[Array, ""] = struct.array(default=1 / 30)
