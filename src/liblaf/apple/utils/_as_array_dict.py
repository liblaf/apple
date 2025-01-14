from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy.typing as npt


def as_array_dict(
    data: Mapping[str, npt.ArrayLike] | None = None,
) -> dict[str, jax.Array]:
    if not data:
        return {}
    return {k: jnp.asarray(v) for k, v in data.items()}
