from typing import no_type_check

import einops
import jax
import jax.numpy as jnp
import numpy as np
import warp as wp
from jaxtyping import Float

from liblaf.apple import elem, func, naive, utils
from liblaf.apple.typed.jax import Mat9x12, Mat12x12, Mat33, Mat43, Mat99, Vec3
from liblaf.apple.typed.warp import mat43


def test_h4_diag(n_cells: int = 7) -> None:
    random = utils.Random()
    points: Float[jax.Array, "cells 4 3"] = random.uniform((n_cells, 4, 3))
    lambdas: Float[jax.Array, "cells 3"] = random.uniform(
        (n_cells, 3), minval=1.0, maxval=2.0
    )
    Q0: Float[jax.Array, "cells 3 3"] = random.uniform((n_cells, 3, 3))
    Q1: Float[jax.Array, "cells 3 3"] = random.uniform((n_cells, 3, 3))
    Q2: Float[jax.Array, "cells 3 3"] = random.uniform((n_cells, 3, 3))
    expected: Float[jax.Array, "cells 4 3"] = h4_diag_naive(points, lambdas, Q0, Q1, Q2)
    actual: Float[jax.Array, "cells 4 3"] = h4_diag_actual(points, lambdas, Q0, Q1, Q2)
    np.testing.assert_allclose(actual, expected, rtol=1e-1)


def h4_diag_naive(
    points: Float[jax.Array, "cells 4 3"],
    lambdas: Float[jax.Array, "cells 3"],
    Q0: Float[jax.Array, "cells 3 3"],
    Q1: Float[jax.Array, "cells 3 3"],
    Q2: Float[jax.Array, "cells 3 3"],
) -> Float[jax.Array, "cells 4 3"]:
    def h4_diag_elem(
        points: Mat43, lambdas: Vec3, Q0: Mat33, Q1: Mat33, Q2: Mat33
    ) -> Mat43:
        dFdx: Mat9x12 = naive.dFdx(points)
        H1: Mat99 = naive.H1(lambdas, Q0, Q1, Q2)
        h4: Mat12x12 = dFdx.T @ H1 @ dFdx
        return einops.rearrange(jnp.diagonal(h4), "(i j) -> i j", i=4, j=3)

    return jax.vmap(h4_diag_elem)(points, lambdas, Q0, Q1, Q2)


def h4_diag_actual(
    points: Float[jax.Array, "cells 4 3"],
    lambdas: Float[jax.Array, "cells 3"],
    Q0: Float[jax.Array, "cells 3 3"],
    Q1: Float[jax.Array, "cells 3 3"],
    Q2: Float[jax.Array, "cells 3 3"],
) -> Float[jax.Array, "cells 4 3"]:
    dh_dX: Float[jax.Array, "cells 4 3"] = elem.tetra.dh_dX(points)

    @no_type_check
    @utils.jax_kernel
    def h4_diag_kernel(
        dh_dX: wp.array(dtype=mat43),
        lambdas: wp.array(dtype=wp.vec3),
        Q0: wp.array(dtype=wp.mat33),
        Q1: wp.array(dtype=wp.mat33),
        Q2: wp.array(dtype=wp.mat33),
        h4_diag: wp.array(dtype=mat43),
    ) -> None:
        tid = wp.tid()
        h4_diag[tid] = func.h4_diag(
            dh_dX=dh_dX[tid],
            lambdas=lambdas[tid],
            Q0=Q0[tid],
            Q1=Q1[tid],
            Q2=Q2[tid],
        )

    h4_diag: Float[jax.Array, "cells 4 3"]
    (h4_diag,) = h4_diag_kernel(dh_dX, lambdas, Q0, Q1, Q2)
    return h4_diag
