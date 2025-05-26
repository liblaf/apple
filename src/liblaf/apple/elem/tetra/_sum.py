import beartype
import einops
import jax
import jaxtyping
from jaxtyping import Float, Integer

from liblaf.apple import utils


@jaxtyping.jaxtyped(typechecker=beartype.beartype)
@utils.jit(static_argnames=["n_points"])
def segment_sum(
    data: Float[jax.Array, "C 4 *D"], cells: Integer[jax.Array, "C 4"], n_points: int
) -> Float[jax.Array, " P *D"]:
    return jax.ops.segment_sum(
        einops.rearrange(data, "C points_per_cell ... -> (C points_per_cell) ..."),
        einops.rearrange(cells, "C points_per_cell -> (C points_per_cell)"),
        num_segments=n_points,
    )
