from collections.abc import Callable, Sequence

import hypothesis
import jax.numpy as jnp
import numpy as np
import warp as wp
from jaxtyping import Array, DTypeLike, Float

import liblaf.apple.warp.utils as wpu
from liblaf.apple.jax import testing
from liblaf.apple.warp.energies.elastic.hyperelastic import func


def diag_from_quad(
    quad: Callable, args: Sequence[wp.array], dtype: DTypeLike
) -> Float[Array, "batch 4 3"]:
    diag: Float[Array, "batch 4 3"] = jnp.zeros((*args[0].shape, 4, 3), dtype)
    for i, j in np.ndindex(4, 3):
        p: Float[Array, "batch 4 3"] = jnp.zeros_like(diag)
        p = p.at[..., i, j].set(1.0)
        p_wp: wp.array = wpu.to_warp(p, (4, 3))
        quad_wp: wp.array = wp.map(quad, p_wp, *args)  # pyright: ignore[reportAssignmentType]
        quad_jax: Float[Array, " batch"] = wp.to_jax(quad_wp)
        diag = diag.at[..., i, j].set(quad_jax)
    return diag


def check_diag(
    diag: Callable, quad: Callable, args: Sequence[wp.array], dtype: DTypeLike
) -> None:
    diag_wp: wp.array = wp.map(diag, *args)  # pyright: ignore[reportAssignmentType]
    diag_jax: Float[Array, " batch 4 3"] = wp.to_jax(diag_wp)
    diag_expected: Float[Array, "batch 4 3"] = diag_from_quad(quad, args, dtype)
    np.testing.assert_allclose(diag_jax, diag_expected)


@hypothesis.given(dhdX=testing.matrices((4, 3)), g1=testing.matrices((3, 3)))
def test_h1_diag(
    dhdX: Float[Array, "batch 4 3"], g1: Float[Array, "batch 3 3"]
) -> None:
    dhdX_wp: wp.array = wpu.to_warp(dhdX, (4, 3))
    g1_wp: wp.array = wpu.to_warp(g1, (3, 3))
    args: Sequence[wp.array] = (dhdX_wp, g1_wp)
    check_diag(func.h1_diag, func.h1_quad, args, dhdX.dtype)


@hypothesis.given(dhdX=testing.matrices((4, 3)), g2=testing.matrices((3, 3)))
def test_h2_diag(
    dhdX: Float[Array, "batch 4 3"], g2: Float[Array, "batch 3 3"]
) -> None:
    dhdX_wp: wp.array = wpu.to_warp(dhdX, (4, 3))
    g2_wp: wp.array = wpu.to_warp(g2, (3, 3))
    args: Sequence[wp.array] = (dhdX_wp, g2_wp)
    check_diag(func.h2_diag, func.h2_quad, args, dhdX.dtype)


@hypothesis.given(dhdX=testing.matrices((4, 3)), g3=testing.matrices((3, 3)))
def test_h3_diag(
    dhdX: Float[Array, "batch 4 3"], g3: Float[Array, "batch 3 3"]
) -> None:
    dhdX_wp: wp.array = wpu.to_warp(dhdX, (4, 3))
    g3_wp: wp.array = wpu.to_warp(g3, (3, 3))
    args: Sequence[wp.array] = (dhdX_wp, g3_wp)
    check_diag(func.h3_diag, func.h3_quad, args, dhdX.dtype)


@hypothesis.given(
    dhdX=testing.matrices((4, 3)),
    U=testing.matrices((3, 3)),
    s=testing.matrices((3,)),
    V=testing.matrices((3, 3)),
)
def test_h4_diag(
    dhdX: Float[Array, "batch 4 3"],
    U: Float[Array, "batch 3 3"],
    s: Float[Array, "batch 3"],
    V: Float[Array, "batch 3 3"],
) -> None:
    dhdX_wp: wp.array = wpu.to_warp(dhdX, (4, 3))
    U_wp: wp.array = wpu.to_warp(U, (3, 3))
    s_wp: wp.array = wpu.to_warp(s, 3)
    V_wp: wp.array = wpu.to_warp(V, (3, 3))
    clamp: wp.array = wp.ones(dhdX_wp.shape, wp.bool)
    args: Sequence[wp.array] = (dhdX_wp, U_wp, s_wp, V_wp, clamp)
    check_diag(func.h4_diag, func.h4_quad, args, dhdX.dtype)


@hypothesis.given(
    dhdX=testing.matrices((4, 3)),
    U=testing.matrices((3, 3)),
    s=testing.matrices((3,)),
    V=testing.matrices((3, 3)),
)
def test_h4_diag_no_clamp(
    dhdX: Float[Array, "batch 4 3"],
    U: Float[Array, "batch 3 3"],
    s: Float[Array, "batch 3"],
    V: Float[Array, "batch 3 3"],
) -> None:
    dhdX_wp: wp.array = wpu.to_warp(dhdX, (4, 3))
    U_wp: wp.array = wpu.to_warp(U, (3, 3))
    s_wp: wp.array = wpu.to_warp(s, 3)
    V_wp: wp.array = wpu.to_warp(V, (3, 3))
    clamp: wp.array = wp.zeros(dhdX_wp.shape, wp.bool)
    args: Sequence[wp.array] = (dhdX_wp, U_wp, s_wp, V_wp, clamp)
    check_diag(func.h4_diag, func.h4_quad, args, dhdX.dtype)


@hypothesis.given(dhdX=testing.matrices((4, 3)))
def test_h5_diag(dhdX: Float[Array, "batch 4 3"]) -> None:
    dhdX_wp: wp.array = wpu.to_warp(dhdX, (4, 3))
    args: Sequence[wp.array] = (dhdX_wp,)
    check_diag(func.h5_diag, func.h5_quad, args, dhdX.dtype)


@hypothesis.given(dhdX=testing.matrices((4, 3)), F=testing.matrices((3, 3)))
def test_h6_diag(dhdX: Float[Array, "batch 4 3"], F: Float[Array, "batch 3 3"]) -> None:
    dhdX_wp: wp.array = wpu.to_warp(dhdX, (4, 3))
    F_wp: wp.array = wpu.to_warp(F, (3, 3))
    args: Sequence[wp.array] = (dhdX_wp, F_wp)
    check_diag(func.h6_diag, func.h6_quad, args, dhdX.dtype)
