import beartype
import einops
import jax
import jaxtyping
from jaxtyping import Float, Integer

from liblaf.apple import utils


@jaxtyping.jaxtyped(typechecker=beartype.beartype)
@utils.jit(static_argnames=["n_points"])
def segment_sum(
    data: Float[jax.Array, "cells 4 *dim"],
    cells: Integer[jax.Array, "cells 4"],
    n_points: int,
) -> Float[jax.Array, " points *dim"]:
    return jax.ops.segment_sum(
        einops.rearrange(
            data, "cells points_per_cell ... -> (cells points_per_cell) ..."
        ),
        einops.rearrange(cells, "cells points_per_cell -> (cells points_per_cell)"),
        num_segments=n_points,
    )
