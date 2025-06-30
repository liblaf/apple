from jaxtyping import Array, Float

from liblaf.apple import struct


@struct.pytree
class GlobalParams(struct.PyTreeMixin):
    time_step: Float[Array, ""] = struct.array(default=1 / 30)
