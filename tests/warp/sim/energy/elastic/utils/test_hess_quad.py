from collections.abc import Sequence

import einops
import hypothesis
import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import jax.numpy as jnp
import numpy as np
import warp as wp
from jaxtyping import Array, Float

import liblaf.apple.warp.utils as wp_utils
from liblaf.apple.warp.sim.energy.elastic import utils


def random_matrices(
    shape: Sequence[int],
) -> st.SearchStrategy[Float[Array, " batch *shape"]]:
    shapes: st.SearchStrategy[tuple[int, ...]] = hnp.array_shapes(
        min_dims=1, max_dims=1
    )
    return hnp.arrays(
        dtype=st.shared(
            hnp.floating_dtypes(endianness="=", sizes=[32, 64]), key="dtype"
        ),
        shape=st.shared(shapes, key="shape").map(lambda s: (*s, *shape)),
        elements=hnp.from_dtype(np.dtype(np.float16), min_value=-1.0, max_value=1.0),
    ).map(lambda a: jnp.asarray(a))


@hypothesis.given(
    dhdX=random_matrices((4, 3)), g1=random_matrices((3, 3)), p=random_matrices((4, 3))
)
def test_h1_quad(
    dhdX: Float[Array, "batch 4 3"],
    g1: Float[Array, "batch 3 3"],
    p: Float[Array, "batch 4 3"],
) -> None:
    dhdX_wp: wp.array = wp_utils.from_jax(dhdX, (4, 3))
    g1_wp: wp.array = wp_utils.from_jax(g1, (3, 3))
    p_wp: wp.array = wp_utils.from_jax(p, (4, 3))
    h1_prod_wp: wp.array = wp.map(utils.h1_prod, dhdX_wp, g1_wp, p_wp)  # pyright: ignore[reportAssignmentType]
    h1_quad_wp: wp.array = wp.map(utils.h1_quad, dhdX_wp, g1_wp, p_wp)  # pyright: ignore[reportAssignmentType]
    h1_prod: Float[Array, "batch 4 3"] = wp.to_jax(h1_prod_wp)
    h1_quad: Float[Array, " batch"] = wp.to_jax(h1_quad_wp)
    h1_quad_expected: Float[Array, " batch"] = einops.einsum(
        p, h1_prod, "batch i j, batch i j -> batch"
    )
    if np.isdtype(dhdX.dtype, np.float64):
        np.testing.assert_allclose(h1_quad, h1_quad_expected)
    elif np.isdtype(dhdX.dtype, np.float32):
        np.testing.assert_allclose(h1_quad, h1_quad_expected, atol=1e-6, rtol=1e-6)
    else:
        np.testing.assert_allclose(h1_quad, h1_quad_expected)


@hypothesis.given(
    dhdX=random_matrices((4, 3)), g2=random_matrices((3, 3)), p=random_matrices((4, 3))
)
def test_h2_quad(
    dhdX: Float[Array, "batch 4 3"],
    g2: Float[Array, "batch 3 3"],
    p: Float[Array, "batch 4 3"],
) -> None:
    dhdX_wp: wp.array = wp_utils.from_jax(dhdX, (4, 3))
    g2_wp: wp.array = wp_utils.from_jax(g2, (3, 3))
    p_wp: wp.array = wp_utils.from_jax(p, (4, 3))
    h2_prod_wp: wp.array = wp.map(utils.h2_prod, dhdX_wp, g2_wp, p_wp)  # pyright: ignore[reportAssignmentType]
    h2_quad_wp: wp.array = wp.map(utils.h2_quad, dhdX_wp, g2_wp, p_wp)  # pyright: ignore[reportAssignmentType]
    h2_prod: Float[Array, "batch 4 3"] = wp.to_jax(h2_prod_wp)
    h2_quad: Float[Array, " batch"] = wp.to_jax(h2_quad_wp)
    h2_quad_expected: Float[Array, " batch"] = einops.einsum(
        p, h2_prod, "batch i j, batch i j -> batch"
    )
    if np.isdtype(dhdX.dtype, np.float64):
        np.testing.assert_allclose(h2_quad, h2_quad_expected)
    elif np.isdtype(dhdX.dtype, np.float32):
        np.testing.assert_allclose(h2_quad, h2_quad_expected, atol=1e-6, rtol=1e-6)
    else:
        np.testing.assert_allclose(h2_quad, h2_quad_expected)


@hypothesis.given(
    dhdX=random_matrices((4, 3)), g3=random_matrices((3, 3)), p=random_matrices((4, 3))
)
def test_h3_quad(
    dhdX: Float[Array, "batch 4 3"],
    g3: Float[Array, "batch 3 3"],
    p: Float[Array, "batch 4 3"],
) -> None:
    dhdX_wp: wp.array = wp_utils.from_jax(dhdX, (4, 3))
    g3_wp: wp.array = wp_utils.from_jax(g3, (3, 3))
    p_wp: wp.array = wp_utils.from_jax(p, (4, 3))
    h3_prod_wp: wp.array = wp.map(utils.h3_prod, dhdX_wp, g3_wp, p_wp)  # pyright: ignore[reportAssignmentType]
    h3_quad_wp: wp.array = wp.map(utils.h3_quad, dhdX_wp, g3_wp, p_wp)  # pyright: ignore[reportAssignmentType]
    h3_prod: Float[Array, "batch 4 3"] = wp.to_jax(h3_prod_wp)
    h3_quad: Float[Array, " batch"] = wp.to_jax(h3_quad_wp)
    h3_quad_expected: Float[Array, " batch"] = einops.einsum(
        p, h3_prod, "batch i j, batch i j -> batch"
    )
    if np.isdtype(dhdX.dtype, np.float64):
        np.testing.assert_allclose(h3_quad, h3_quad_expected)
    elif np.isdtype(dhdX.dtype, np.float32):
        np.testing.assert_allclose(h3_quad, h3_quad_expected, atol=1e-6, rtol=1e-6)
    else:
        np.testing.assert_allclose(h3_quad, h3_quad_expected)
