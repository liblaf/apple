import flax.struct
import jax
import jax.numpy as jnp
from jaxtyping import Float

from ._abc import FunctionSpace


class FunctionSpaceTetra(FunctionSpace):
    w: Float[jax.Array, ""] = flax.struct.field(
        default_factory=lambda: jnp.asarray(1.0 / 6.0)
    )
    h: Float[jax.Array, "a=4"] = flax.struct.field(
        default_factory=lambda: jnp.asarray([0.25, 0.25, 0.25, 0.25])
    )
    dh_dr: Float[jax.Array, "a=4 J=3"] = flax.struct.field(
        default_factory=lambda: jnp.asarray(
            [
                [-1, -1, -1],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
            ],
            dtype=float,
        )
    )
