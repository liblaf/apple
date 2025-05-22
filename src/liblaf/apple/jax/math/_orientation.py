import einops
import jax
import jax.numpy as jnp
from jaxtyping import Float
from numpy.typing import ArrayLike

from liblaf.apple import utils


@utils.jit
def orientation_matrix(
    a: Float[ArrayLike, "*N 3"], b: Float[ArrayLike, "*N 3"]
) -> Float[jax.Array, "*N 3 3"]:
    r""".

    ```math
    result = \frac{b a^T}{a^T a}
    ```
    """
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    norm_square: Float[ArrayLike, "*N 1"] = jnp.sum(a**2, axis=-1)
    orientation: Float[ArrayLike, "*N 3 3"] = (
        einops.einsum(b, a, "... i, ... j -> ... i j") / norm_square[..., None, None]
    )
    return orientation
