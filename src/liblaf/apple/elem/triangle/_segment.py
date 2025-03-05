import beartype
import einops
import jax
import jaxtyping
from jaxtyping import Float, Integer

from liblaf import apple


@jaxtyping.jaxtyped(typechecker=beartype.beartype)
@apple.jit(static_argnames=["n_points"])
def segment_sum(
    data: Float[jax.Array, "C 3 *D"], cells: Integer[jax.Array, "C 3"], n_points: int
) -> Float[jax.Array, " P *D"]:
    return jax.ops.segment_sum(
        einops.rearrange(data, "C points_per_cell ... -> (C points_per_cell) ..."),
        # data.reshape((data.shape[0] * data.shape[1], *data.shape[2:])),
        einops.rearrange(cells, "C points_per_cell -> (C points_per_cell)"),
        num_segments=n_points,
    )
