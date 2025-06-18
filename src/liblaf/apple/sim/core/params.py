import jax
import jax.numpy as jnp
from jaxtyping import Float

from liblaf.apple import struct


class GlobalParams(struct.PyTree):
    time_step: Float[jax.Array, ""] = struct.array(factory=lambda: jnp.asarray(1 / 30))
