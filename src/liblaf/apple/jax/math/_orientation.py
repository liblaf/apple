import beartype
import einops
import jax
import jax.numpy as jnp
import jaxtyping
from jaxtyping import Float

from liblaf.apple import utils


@utils.jit
@jaxtyping.jaxtyped(typechecker=beartype.beartype)
def orientation_matrix(
    a: Float[jax.Array, "*N 3"], b: Float[jax.Array, "*N 3"]
) -> Float[jax.Array, "*N 3 3"]:
    r""".

    ```math
    result = \frac{b a^T}{a^T a}
    ```
    """
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    norm_square: Float[jax.Array, "*N 1"] = jnp.sum(a**2, axis=-1)
    orientation: Float[jax.Array, "*N 3 3"] = (
        einops.einsum(b, a, "... i, ... j -> ... i j") / norm_square[..., None, None]
    )
    return orientation
