import jax
import jax.numpy as jnp
from jaxtyping import Float


@jax.jit
def det_jac(F: Float[jax.Array, "3 3"]) -> Float[jax.Array, "3 3"]:
    F00: Float[jax.Array, ""] = F[0, 0]
    F01: Float[jax.Array, ""] = F[0, 1]
    F02: Float[jax.Array, ""] = F[0, 2]
    F10: Float[jax.Array, ""] = F[1, 0]
    F11: Float[jax.Array, ""] = F[1, 1]
    F12: Float[jax.Array, ""] = F[1, 2]
    F20: Float[jax.Array, ""] = F[2, 0]
    F21: Float[jax.Array, ""] = F[2, 1]
    F22: Float[jax.Array, ""] = F[2, 2]
    return jnp.asarray(
        [
            [F11 * F22 - F12 * F21, -F10 * F22 + F12 * F20, F10 * F21 - F11 * F20],
            [-F01 * F22 + F02 * F21, F00 * F22 - F02 * F20, -F00 * F21 + F01 * F20],
            [F01 * F12 - F02 * F11, -F00 * F12 + F02 * F10, F00 * F11 - F01 * F10],
        ]
    )
