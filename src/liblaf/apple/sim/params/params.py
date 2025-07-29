import jax.numpy as jnp
from jaxtyping import Array, Float

from liblaf.apple import struct


class GlobalParams(struct.PyTree):
    time_step: Float[Array, ""] = struct.array(factory=lambda: jnp.asarray(1 / 30))
