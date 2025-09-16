from collections.abc import Callable, Sequence

import hypothesis
import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import jax.numpy as jnp
import numpy as np
import warp as wp
from jaxtyping import Array, DTypeLike, Float

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


def diag_from_quad(
    quad: Callable | wp.Function, args: Sequence[wp.array], dtype: DTypeLike
) -> Float[Array, "batch 4 3"]:
    diag: Float[Array, "batch 4 3"] = jnp.zeros((*args[0].shape, 4, 3), dtype)
    for i, j in np.ndindex(4, 3):
        p: Float[Array, "batch 4 3"] = jnp.zeros_like(diag)
        p = p.at[..., i, j].set(1.0)
        p_wp: wp.array = wp_utils.to_warp(p, wp_utils.MatrixLike(4, 3))
        quad_wp: wp.array = wp.map(quad, p_wp, *args)  # pyright: ignore[reportAssignmentType]
        quad_jax: Float[Array, " batch"] = wp.to_jax(quad_wp)
        diag = diag.at[..., i, j].set(quad_jax)
    return diag


def check_diag(
    diag: Callable | wp.Function,
    quad: Callable | wp.Function,
    args: Sequence[wp.array],
    dtype: DTypeLike,
) -> None:
    diag_wp: wp.array = wp.map(diag, *args)  # pyright: ignore[reportAssignmentType]
    diag_jax: Float[Array, " batch 4 3"] = wp.to_jax(diag_wp)
    diag_expected: Float[Array, "batch 4 3"] = diag_from_quad(quad, args, dtype)
    np.testing.assert_allclose(diag_jax, diag_expected)


@hypothesis.given(dhdX=random_matrices((4, 3)), g1=random_matrices((3, 3)))
def test_h1_diag(
    dhdX: Float[Array, "batch 4 3"], g1: Float[Array, "batch 3 3"]
) -> None:
    dhdX_wp: wp.array = wp_utils.to_warp(dhdX, wp_utils.MatrixLike(4, 3))
    g1_wp: wp.array = wp_utils.to_warp(g1, wp_utils.MatrixLike(3, 3))
    args: Sequence[wp.array] = (dhdX_wp, g1_wp)
    check_diag(func.h1_diag, func.h1_quad, args, dhdX.dtype)


@hypothesis.given(dhdX=random_matrices((4, 3)), g2=random_matrices((3, 3)))
def test_h2_diag(
    dhdX: Float[Array, "batch 4 3"], g2: Float[Array, "batch 3 3"]
) -> None:
    dhdX_wp: wp.array = wp_utils.to_warp(dhdX, wp_utils.MatrixLike(4, 3))
    g2_wp: wp.array = wp_utils.to_warp(g2, wp_utils.MatrixLike(3, 3))
    args: Sequence[wp.array] = (dhdX_wp, g2_wp)
    check_diag(func.h2_diag, func.h2_quad, args, dhdX.dtype)


@hypothesis.given(dhdX=random_matrices((4, 3)), g3=random_matrices((3, 3)))
def test_h3_diag(
    dhdX: Float[Array, "batch 4 3"], g3: Float[Array, "batch 3 3"]
) -> None:
    dhdX_wp: wp.array = wp_utils.to_warp(dhdX, wp_utils.MatrixLike(4, 3))
    g3_wp: wp.array = wp_utils.to_warp(g3, wp_utils.MatrixLike(3, 3))
    args: Sequence[wp.array] = (dhdX_wp, g3_wp)
    check_diag(func.h3_diag, func.h3_quad, args, dhdX.dtype)


@hypothesis.given(
    dhdX=random_matrices((4, 3)),
    lambdas=random_matrices((3,)),
    Q0=random_matrices((3, 3)),
    Q1=random_matrices((3, 3)),
    Q2=random_matrices((3, 3)),
)
def test_h4_diag(
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
    check_diag(func.h4_diag, func.h4_quad, args, dhdX.dtype)


@hypothesis.given(dhdX=random_matrices((4, 3)))
def test_h5_diag(dhdX: Float[Array, "batch 4 3"]) -> None:
    dhdX_wp: wp.array = wp_utils.to_warp(dhdX, wp_utils.MatrixLike(4, 3))
    args: Sequence[wp.array] = (dhdX_wp,)
    check_diag(func.h5_diag, func.h5_quad, args, dhdX.dtype)


@hypothesis.given(dhdX=random_matrices((4, 3)), F=random_matrices((3, 3)))
def test_h6_diag(dhdX: Float[Array, "batch 4 3"], F: Float[Array, "batch 3 3"]) -> None:
    dhdX_wp: wp.array = wp_utils.to_warp(dhdX, wp_utils.MatrixLike(4, 3))
    F_wp: wp.array = wp_utils.to_warp(F, wp_utils.MatrixLike(3, 3))
    args: Sequence[wp.array] = (dhdX_wp, F_wp)
    check_diag(func.h6_diag, func.h6_quad, args, dhdX.dtype)
