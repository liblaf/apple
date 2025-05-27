from typing import no_type_check

import einops
import jax
import jax.numpy as jnp
import numpy as np
import warp as wp
from jaxtyping import Float

from liblaf.apple import elem, func, naive, utils
from liblaf.apple.typed.jax import Mat9x12, Mat33, Mat43, Vec9
from liblaf.apple.typed.warp import mat43


def test_h3_diag(n_cells: int = 7) -> None:
    random = utils.Random()
    points: Float[jax.Array, "cells 4 3"] = random.uniform((n_cells, 4, 3))
    g3: Float[jax.Array, "cells 3 3"] = random.uniform((n_cells, 3, 3))

    def h3_diag_naive(
        points: Float[jax.Array, "cells 4 3"], g3: Float[jax.Array, "cells 3 3"]
    ) -> Float[jax.Array, "cells 4 3"]:
        def h3_diag_elem(points: Mat43, g3: Mat33) -> Mat43:
            dFdx: Mat9x12 = naive.dFdx(points)
            g3: Vec9 = einops.rearrange(g3, "i j -> (j i)")
            h3: Float[jax.Array, "12 12"] = jnp.outer(dFdx.T @ g3, dFdx.T @ g3)
            return einops.rearrange(jnp.diagonal(h3), "(i j) -> i j", i=4, j=3)

        return jax.vmap(h3_diag_elem)(points, g3)

    expected: Float[jax.Array, "cells 4 3"] = h3_diag_naive(points, g3)

    def h3_diag_actual(
        points: Float[jax.Array, "cells 4 3"], g3: Float[jax.Array, "cells 3 3"]
    ) -> Float[jax.Array, "cells 4 3"]:
        dh_dX: Float[jax.Array, "cells 4 3"] = elem.tetra.dh_dX(points)

        @no_type_check
        @utils.jax_kernel
        def h3_diag_kernel(
            dh_dX: wp.array(dtype=mat43),
            g3: wp.array(dtype=wp.mat33),
            h3_diag: wp.array(dtype=mat43),
        ) -> None:
            tid = wp.tid()
            h3_diag[tid] = func.h3_diag(dh_dX[tid], g3[tid])

        h3_diag: Float[jax.Array, "cells 4 3"]
        (h3_diag,) = h3_diag_kernel(dh_dX, g3)
        return h3_diag

    actual: Float[jax.Array, "cells 4 3"] = h3_diag_actual(points, g3)
    np.testing.assert_allclose(actual, expected, rtol=1e-4)
