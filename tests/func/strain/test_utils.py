from typing import no_type_check

import jax
import numpy as np
import warp as wp
from jaxtyping import Float

from liblaf.apple import func, naive, utils


def test_Qs(n_cells: int = 7) -> None:
    random = utils.Random()
    U: Float[jax.Array, "cells 3 3"] = random.uniform((n_cells, 3, 3))
    V: Float[jax.Array, "cells 3 3"] = random.uniform((n_cells, 3, 3))

    @utils.jit
    def Qs_naive(
        U: Float[jax.Array, "cells 3 3"], V: Float[jax.Array, "cells 3 3"]
    ) -> tuple[
        Float[jax.Array, "cells 3 3"],
        Float[jax.Array, "cells 3 3"],
        Float[jax.Array, "cells 3 3"],
    ]:
        return jax.vmap(naive.strain.Qs)(U, V)

    expected: tuple[
        Float[jax.Array, "cells 3 3"],
        Float[jax.Array, "cells 3 3"],
        Float[jax.Array, "cells 3 3"],
    ] = Qs_naive(U, V)

    @no_type_check
    @utils.jax_kernel(num_outputs=3)
    def Qs_actual(
        U: wp.array(dtype=wp.mat33),
        V: wp.array(dtype=wp.mat33),
        Q0: wp.array(dtype=wp.mat33),
        Q1: wp.array(dtype=wp.mat33),
        Q2: wp.array(dtype=wp.mat33),
    ) -> None:
        tid = wp.tid()
        Q0_, Q1_, Q2_ = func.strain.Qs(U[tid], V[tid])
        Q0[tid] = Q0_
        Q1[tid] = Q1_
        Q2[tid] = Q2_

    actual: tuple[
        Float[jax.Array, "cells 3 3"],
        Float[jax.Array, "cells 3 3"],
        Float[jax.Array, "cells 3 3"],
    ] = Qs_actual(U, V)  # pyright: ignore[reportCallIssue]

    np.testing.assert_allclose(actual, expected, rtol=1e-6)
