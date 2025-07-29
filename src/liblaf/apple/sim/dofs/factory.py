from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np

from .dofs import DOFs


def make_dofs(shape: Sequence[int], *, offset: int = 0) -> DOFs:
    return DOFs(
        index=jnp.arange(offset, offset + np.prod(shape), dtype=int), shape=shape
    )
