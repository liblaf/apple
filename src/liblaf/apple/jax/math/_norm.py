from collections.abc import Sequence

import jax.numpy as jnp
from jaxtyping import Array, Float

type Axis = int | Sequence[int] | None


def fro_norm_square(x: Array, axis: Axis = None) -> Float[Array, "*batch"]:
    return jnp.sum(jnp.square(x), axis=axis)
