from collections.abc import Callable, Sequence

import einops
import hypothesis
import numpy as np
import warp as wp
from jaxtyping import Array, Float

import liblaf.apple.warp.utils as wpu
from liblaf.apple.jax import testing
from liblaf.apple.warp.energies.elastic.hyperelastic import func


def check_prod(
    prod: Callable,
    quad: Callable,
    p: Float[Array, "batch 4 3"],
    args: Sequence[wp.array],
) -> None:
    p_wp: wp.array = wpu.to_warp(p, (4, 3))
    prod_wp: wp.array = wp.map(prod, p_wp, *args)  # pyright: ignore[reportAssignmentType]
    prod_jax: Float[Array, "batch 4 3"] = wp.to_jax(prod_wp)
    quad_wp: wp.array = wp.map(quad, p_wp, *args)  # pyright: ignore[reportAssignmentType]
    quad_jax: Float[Array, " batch"] = wp.to_jax(quad_wp)
    quad_expected: Float[Array, " batch"] = einops.einsum(
        p, prod_jax, "batch i j, batch i j -> batch"
    )
    np.testing.assert_allclose(quad_jax, quad_expected, atol=1e-15)


@hypothesis.given(
    p=testing.matrices((4, 3)),
    dhdX=testing.matrices((4, 3)),
    g1=testing.matrices((3, 3)),
)
def test_h1_prod(
    p: Float[Array, "batch 4 3"],
    dhdX: Float[Array, "batch 4 3"],
    g1: Float[Array, "batch 3 3"],
) -> None:
    dhdX_wp: wp.array = wpu.to_warp(dhdX, (4, 3))
    g1_wp: wp.array = wpu.to_warp(g1, (3, 3))
    args: Sequence[wp.array] = (dhdX_wp, g1_wp)
    check_prod(func.h1_prod, func.h1_quad, p, args)


@hypothesis.given(
    p=testing.matrices((4, 3)),
    dhdX=testing.matrices((4, 3)),
    g2=testing.matrices((3, 3)),
)
def test_h2_prod(
    p: Float[Array, "batch 4 3"],
    dhdX: Float[Array, "batch 4 3"],
    g2: Float[Array, "batch 3 3"],
) -> None:
    dhdX_wp: wp.array = wpu.to_warp(dhdX, (4, 3))
    g2_wp: wp.array = wpu.to_warp(g2, (3, 3))
    args: Sequence[wp.array] = (dhdX_wp, g2_wp)
    check_prod(func.h2_prod, func.h2_quad, p, args)


@hypothesis.given(
    p=testing.matrices((4, 3)),
    dhdX=testing.matrices((4, 3)),
    g3=testing.matrices((3, 3)),
)
def test_h3_prod(
    p: Float[Array, "batch 4 3"],
    dhdX: Float[Array, "batch 4 3"],
    g3: Float[Array, "batch 3 3"],
) -> None:
    dhdX_wp: wp.array = wpu.to_warp(dhdX, (4, 3))
    g3_wp: wp.array = wpu.to_warp(g3, (3, 3))
    args: Sequence[wp.array] = (dhdX_wp, g3_wp)
    check_prod(func.h3_prod, func.h3_quad, p, args)


@hypothesis.given(
    p=testing.matrices((4, 3)),
    dhdX=testing.matrices((4, 3)),
    U=testing.matrices((3, 3)),
    s=testing.matrices((3,)),
    V=testing.matrices((3, 3)),
)
def test_h4_prod(
    p: Float[Array, "batch 4 3"],
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
    check_prod(func.h4_prod, func.h4_quad, p, args)


@hypothesis.given(
    p=testing.matrices((4, 3)),
    dhdX=testing.matrices((4, 3)),
    U=testing.matrices((3, 3)),
    s=testing.matrices((3,)),
    V=testing.matrices((3, 3)),
)
def test_h4_prod_no_clamp(
    p: Float[Array, "batch 4 3"],
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
    check_prod(func.h4_prod, func.h4_quad, p, args)


@hypothesis.given(p=testing.matrices((4, 3)), dhdX=testing.matrices((4, 3)))
def test_h5_prod(p: Float[Array, "batch 4 3"], dhdX: Float[Array, "batch 4 3"]) -> None:
    dhdX_wp: wp.array = wpu.to_warp(dhdX, (4, 3))
    args: Sequence[wp.array] = (dhdX_wp,)
    check_prod(func.h5_prod, func.h5_quad, p, args)


@hypothesis.given(
    p=testing.matrices((4, 3)),
    dhdX=testing.matrices((4, 3)),
    F=testing.matrices((3, 3)),
)
def test_h6_prod(
    p: Float[Array, "batch 4 3"],
    dhdX: Float[Array, "batch 4 3"],
    F: Float[Array, "batch 3 3"],
) -> None:
    dhdX_wp: wp.array = wpu.to_warp(dhdX, (4, 3))
    F_wp: wp.array = wpu.to_warp(F, (3, 3))
    args: Sequence[wp.array] = (dhdX_wp, F_wp)
    check_prod(func.h6_prod, func.h6_quad, p, args)
