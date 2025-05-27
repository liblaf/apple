from typing import no_type_check

import einops
import jax
import jax.numpy as jnp
import numpy as np
import warp as wp
from jaxtyping import Float

from liblaf.apple import elem, func, naive, utils
from liblaf.apple.typed.jax import Mat9x12, Mat12x12, Mat43, Mat99
from liblaf.apple.typed.warp import mat43


def test_h5_diag(n_cells: int = 7) -> None:
    random = utils.Random()
    points: Float[jax.Array, "cells 4 3"] = random.uniform((n_cells, 4, 3))
    expected: Float[jax.Array, "cells 4 3"] = h5_diag_naive(points)
    actual: Float[jax.Array, "cells 4 3"] = h5_diag_actual(points)
    np.testing.assert_allclose(actual, expected, rtol=1e-2)


def h5_diag_naive(
    points: Float[jax.Array, "cells 4 3"],
) -> Float[jax.Array, "cells 4 3"]:
    def h5_diag_elem(points: Mat43) -> Mat43:
        dFdx: Mat9x12 = naive.dFdx(points)
        H2: Mat99 = naive.H2()
        h5: Mat12x12 = dFdx.T @ H2 @ dFdx
        return einops.rearrange(jnp.diagonal(h5), "(i j) -> i j", i=4, j=3)

    return jax.vmap(h5_diag_elem)(points)


def h5_diag_actual(
    points: Float[jax.Array, "cells 4 3"],
) -> Float[jax.Array, "cells 4 3"]:
    dh_dX: Float[jax.Array, "cells 4 3"] = elem.tetra.dh_dX(points)

    @no_type_check
    @utils.jax_kernel
    def h5_diag_kernel(
        dh_dX: wp.array(dtype=mat43), h5_diag: wp.array(dtype=mat43)
    ) -> None:
        tid = wp.tid()
        h5_diag[tid] = func.h5_diag(dh_dX=dh_dX[tid])

    h5_diag: Float[jax.Array, "cells 4 3"]
    (h5_diag,) = h5_diag_kernel(dh_dX)
    return h5_diag
