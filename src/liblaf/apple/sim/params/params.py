import jax
from jaxtyping import Float

from liblaf.apple import struct


@struct.pytree
class GlobalParams(struct.PyTreeMixin):
    time_step: Float[jax.Array, ""] = struct.array(default=1 / 30)
