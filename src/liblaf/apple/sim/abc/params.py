import jax
import jax.numpy as jnp
from jaxtyping import Float

from liblaf.apple import struct


def _default_time_step() -> Float[jax.Array, ""]:
    with jax.ensure_compile_time_eval():
        return jnp.asarray(1 / 30)


class GlobalParams(struct.PyTree):
    time_step: Float[jax.Array, ""] = struct.array(factory=_default_time_step)
