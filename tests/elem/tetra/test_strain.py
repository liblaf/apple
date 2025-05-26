import einops
import jax
import numpy as np
from jaxtyping import Float

from liblaf.apple import elem, naive, utils
from liblaf.apple.typed.jax import Mat9x12, Mat43, Vec9, Vec12


def test_deformation_gradient(n_cells: int = 7) -> None:
    rng = utils.Random()
    points: Float[jax.Array, "cells 4 3"] = rng.uniform((n_cells, 4, 3))
    u: Float[jax.Array, "cells 4 3"] = rng.uniform((n_cells, 4, 3))

    @utils.jit
    def deformation_gradient_naive(
        u: Float[jax.Array, "cells 4 3"], points: Float[jax.Array, "cells 4 3"]
    ) -> Float[jax.Array, "cells 3 3"]:
        return jax.vmap(naive.elem.deformation_gradient)(u, points)

    expected: Float[jax.Array, "cells 3 3"] = deformation_gradient_naive(u, points)

    dh_dX: Float[jax.Array, "cells 4 3"] = elem.tetra.dh_dX(points)
    actual: Float[jax.Array, "cells 3 3"] = elem.tetra.deformation_gradient(u, dh_dX)

    np.testing.assert_allclose(actual, expected, rtol=1e-4)


def test_dFdx(n_cells: int = 7) -> None:
    rng = utils.Random()
    points: Float[jax.Array, "cells 4 3"] = rng.uniform((n_cells, 4, 3))

    @utils.jit
    def dFdx_naive(
        points: Float[jax.Array, "cells 4 3"],
    ) -> Float[jax.Array, "cells 9 12"]:
        return jax.vmap(naive.elem.dFdx)(points)

    expected: Float[jax.Array, "cells 9 12"] = dFdx_naive(points)

    dh_dX: Float[jax.Array, "cells 4 3"] = elem.tetra.dh_dX(points)
    actual: Float[jax.Array, "cells 3 3 4 3"] = elem.tetra.deformation_gradient_jac(
        dh_dX
    )
    actual: Float[jax.Array, "cells 9 12"] = einops.rearrange(
        actual, "cells i j k l -> cells (j i) (k l)"
    )

    np.testing.assert_allclose(actual, expected)


def test_dFdx_p(n_cells: int = 7) -> None:
    rng = utils.Random()
    points: Float[jax.Array, "cells 4 3"] = rng.uniform((n_cells, 4, 3))
    p: Float[jax.Array, "cells 4 3"] = rng.uniform((n_cells, 4, 3))

    @utils.jit
    def dFdx_p_naive(
        points: Float[jax.Array, "cells 4 3"], p: Float[jax.Array, "cells 4 3"]
    ) -> Float[jax.Array, "cells 9"]:
        def dFdx_p_naive_elem(points: Mat43, p: Mat43) -> Vec9:
            p: Vec12 = einops.rearrange(p, "i j -> (i j)")
            dFdx: Mat9x12 = naive.elem.dFdx(points)
            return dFdx @ p

        return jax.vmap(dFdx_p_naive_elem)(points, p)

    expected: Float[jax.Array, "cells 9"] = dFdx_p_naive(points, p)

    dh_dX: Float[jax.Array, "cells 4 3"] = elem.tetra.dh_dX(points)
    actual: Float[jax.Array, "cells 3 3"] = elem.tetra.deformation_gradient_jvp(
        dh_dX, p
    )
    actual: Float[jax.Array, "cells 9"] = einops.rearrange(
        actual, "cells i j -> cells (j i)"
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-6)
