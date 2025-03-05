import beartype
import einops
import jax
import jaxtyping
from jaxtyping import Float

from liblaf import apple


@jaxtyping.jaxtyped(typechecker=beartype.beartype)
@apple.jit()
def first_fundamental_form(
    points: Float[jax.Array, "*C 3 3"],
) -> Float[jax.Array, "*C 2 2"]:
    dr: Float[jax.Array, "*C 2 3"] = points[..., 1:, :] - points[..., :1, :]
    I: Float[jax.Array, "*C 2 2"] = einops.einsum(dr, dr, "... i j, ... k j -> ... i k")  # noqa: E741
    return I
