from collections.abc import Callable, Sequence

import einops
import hypothesis
import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import jax.numpy as jnp
import numpy as np
import warp as wp
from jaxtyping import Array, Float

import liblaf.apple.warp.utils as wp_utils
from liblaf.apple.warp.sim.energy.elastic import func


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


def check_quad(
    prod: Callable | wp.Function,
    quad: Callable | wp.Function,
    p: Float[Array, "batch 4 3"],
    args: Sequence[wp.array],
) -> None:
    p_wp: wp.array = wp_utils.to_warp(p, wp_utils.MatrixLike(4, 3))
    prod_wp: wp.array = wp.map(prod, p_wp, *args)  # pyright: ignore[reportAssignmentType]
    prod_jax: Float[Array, "batch 4 3"] = wp.to_jax(prod_wp)
    quad_wp: wp.array = wp.map(quad, p_wp, *args)  # pyright: ignore[reportAssignmentType]
    quad_jax: Float[Array, " batch"] = wp.to_jax(quad_wp)
    quad_expected: Float[Array, " batch"] = einops.einsum(
        p, prod_jax, "batch i j, batch i j -> batch"
    )
    if jnp.isdtype(p.dtype, jnp.float32):
        np.testing.assert_allclose(quad_jax, quad_expected, atol=1e-4, rtol=1e-4)
    elif jnp.isdtype(p.dtype, jnp.float64):
        np.testing.assert_allclose(quad_jax, quad_expected, atol=1e-15)
    else:
        np.testing.assert_allclose(quad_jax, quad_expected)


@hypothesis.given(
    p=random_matrices((4, 3)), dhdX=random_matrices((4, 3)), g1=random_matrices((3, 3))
)
def test_h1_quad(
    p: Float[Array, "batch 4 3"],
    dhdX: Float[Array, "batch 4 3"],
    g1: Float[Array, "batch 3 3"],
) -> None:
    dhdX_wp: wp.array = wp_utils.to_warp(dhdX, wp_utils.MatrixLike(4, 3))
    g1_wp: wp.array = wp_utils.to_warp(g1, wp_utils.MatrixLike(3, 3))
    args: Sequence[wp.array] = (dhdX_wp, g1_wp)
    check_quad(func.h1_prod, func.h1_quad, p, args)


@hypothesis.given(
    p=random_matrices((4, 3)), dhdX=random_matrices((4, 3)), g2=random_matrices((3, 3))
)
def test_h2_quad(
    p: Float[Array, "batch 4 3"],
    dhdX: Float[Array, "batch 4 3"],
    g2: Float[Array, "batch 3 3"],
) -> None:
    dhdX_wp: wp.array = wp_utils.to_warp(dhdX, wp_utils.MatrixLike(4, 3))
    g2_wp: wp.array = wp_utils.to_warp(g2, wp_utils.MatrixLike(3, 3))
    args: Sequence[wp.array] = (dhdX_wp, g2_wp)
    check_quad(func.h2_prod, func.h2_quad, p, args)


@hypothesis.given(
    p=random_matrices((4, 3)), dhdX=random_matrices((4, 3)), g3=random_matrices((3, 3))
)
def test_h3_quad(
    p: Float[Array, "batch 4 3"],
    dhdX: Float[Array, "batch 4 3"],
    g3: Float[Array, "batch 3 3"],
) -> None:
    dhdX_wp: wp.array = wp_utils.to_warp(dhdX, wp_utils.MatrixLike(4, 3))
    g3_wp: wp.array = wp_utils.to_warp(g3, wp_utils.MatrixLike(3, 3))
    args: Sequence[wp.array] = (dhdX_wp, g3_wp)
    check_quad(func.h3_prod, func.h3_quad, p, args)


@hypothesis.given(
    p=random_matrices((4, 3)),
    dhdX=random_matrices((4, 3)),
    lambdas=random_matrices((3,)),
    Q0=random_matrices((3, 3)),
    Q1=random_matrices((3, 3)),
    Q2=random_matrices((3, 3)),
)
def test_h4_quad(
    p: Float[Array, "batch 4 3"],
    dhdX: Float[Array, "batch 4 3"],
    lambdas: Float[Array, "batch 3"],
    Q0: Float[Array, "batch 3 3"],
    Q1: Float[Array, "batch 3 3"],
    Q2: Float[Array, "batch 3 3"],
) -> None:
    dhdX_wp: wp.array = wp_utils.to_warp(dhdX, wp_utils.MatrixLike(4, 3))
    lambdas_wp: wp.array = wp_utils.to_warp(lambdas, wp_utils.VectorLike(3))
    Q0_wp: wp.array = wp_utils.to_warp(Q0, wp_utils.MatrixLike(3, 3))
    Q1_wp: wp.array = wp_utils.to_warp(Q1, wp_utils.MatrixLike(3, 3))
    Q2_wp: wp.array = wp_utils.to_warp(Q2, wp_utils.MatrixLike(3, 3))
    args: Sequence[wp.array] = (dhdX_wp, lambdas_wp, Q0_wp, Q1_wp, Q2_wp)
    check_quad(func.h4_prod, func.h4_quad, p, args)


@hypothesis.given(p=random_matrices((4, 3)), dhdX=random_matrices((4, 3)))
def test_h5_quad(p: Float[Array, "batch 4 3"], dhdX: Float[Array, "batch 4 3"]) -> None:
    dhdX_wp: wp.array = wp_utils.to_warp(dhdX, wp_utils.MatrixLike(4, 3))
    args: Sequence[wp.array] = (dhdX_wp,)
    check_quad(func.h5_prod, func.h5_quad, p, args)


@hypothesis.given(
    p=random_matrices((4, 3)), dhdX=random_matrices((4, 3)), F=random_matrices((3, 3))
)
def test_h6_quad(
    p: Float[Array, "batch 4 3"],
    dhdX: Float[Array, "batch 4 3"],
    F: Float[Array, "batch 3 3"],
) -> None:
    dhdX_wp: wp.array = wp_utils.to_warp(dhdX, wp_utils.MatrixLike(4, 3))
    F_wp: wp.array = wp_utils.to_warp(F, wp_utils.MatrixLike(3, 3))
    args: Sequence[wp.array] = (dhdX_wp, F_wp)
    check_quad(func.h6_prod, func.h6_quad, p, args)
