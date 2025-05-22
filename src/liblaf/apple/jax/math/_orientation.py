import jax
import jax.numpy as jnp
from jaxtyping import Float
from numpy.typing import ArrayLike

from liblaf.apple import utils


@utils.jit
def orientation_matrix(
    a: Float[ArrayLike, "*N 3"], b: Float[ArrayLike, "*N 3"]
) -> Float[ArrayLike, "*N 3 3"]:
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    norm_square: Float[ArrayLike, "*N 1"] = jnp.sum(a**2, axis=-1, keepdims=True)
    orientation: Float[ArrayLike, "*N 3 3"] = (
        a[..., None, :] * b[..., :, None] / norm_square
    )
    return jax.vmap(jax.vmap(jnp.cross))(a, b)
