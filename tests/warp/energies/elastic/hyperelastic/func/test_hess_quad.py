from collections.abc import Callable
from typing import Any, no_type_check

import hypothesis
import numpy as np
import warp as wp
from jaxtyping import Array, Float

import liblaf.apple.warp.utils as wpu
from liblaf.apple.jax import testing
from liblaf.apple.warp import math
from liblaf.apple.warp.energies.elastic.hyperelastic import func

type Mat33 = Float[Array, "batch 3 3"]
type Mat43 = Float[Array, "batch 4 3"]
type Scalar = Float[Array, " batch"]


EPS: float = 1e-4


def numeric_quad(
    g: Callable, F: Mat33, p: Mat43, dhdX: Mat43, *, eps: float = EPS
) -> Scalar:
    @wp.kernel
    @no_type_check
    def kernel(
        F: wp.array(dtype=Any),
        p: wp.array(dtype=Any),
        dhdX: wp.array(dtype=Any),
        output: wp.array(dtype=Any),
    ) -> None:
        cid = wp.tid()
        dFdx_p = func.deformation_gradient_jvp(dhdX[cid], p[cid])  # mat33
        F0 = F[cid] - dFdx_p.dtype(0.5) * dFdx_p.dtype(eps) * dFdx_p  # mat33
        F1 = F[cid] + dFdx_p.dtype(0.5) * dFdx_p.dtype(eps) * dFdx_p  # mat33
        f0 = g(F0)  # mat33
        f1 = g(F1)  # mat33
        output[cid] = wp.ddot(dFdx_p, f1 - f0) / dFdx_p.dtype(eps)

    F_wp: wp.array = wpu.to_warp(F, (3, 3))
    p_wp: wp.array = wpu.to_warp(p, (4, 3))
    dhdX_wp: wp.array = wpu.to_warp(dhdX, (4, 3))
    output_wp: wp.array = wp.zeros(F_wp.shape, wp.dtype_from_jax(F.dtype))
    wp.launch(kernel, F_wp.shape, inputs=[F_wp, p_wp, dhdX_wp], outputs=[output_wp])
    output: Scalar = wp.to_jax(output_wp)
    return output


def h4_quad(F: Mat33, p: Mat43, dhdX: Mat43) -> Scalar:
    @wp.kernel
    @no_type_check
    def kernel(
        F: wp.array(dtype=Any),
        p: wp.array(dtype=Any),
        dhdX: wp.array(dtype=Any),
        output: wp.array(dtype=Any),
    ) -> None:
        cid = wp.tid()
        U, s, V = math.svd_rv(F[cid])  # mat33, vec3, mat33
        h4_quad = func.h4_quad(p[cid], dhdX[cid], U, s, V, clamp=False)  # mat43
        output[cid] = h4_quad

    F_wp: wp.array = wpu.to_warp(F, (3, 3))
    p_wp: wp.array = wpu.to_warp(p, (4, 3))
    dhdX_wp: wp.array = wpu.to_warp(dhdX, (4, 3))
    output_wp: wp.array = wp.zeros(F_wp.shape, wp.dtype_from_jax(F.dtype))
    wp.launch(kernel, F_wp.shape, inputs=[F_wp, p_wp, dhdX_wp], outputs=[output_wp])
    output: Scalar = wp.to_jax(output_wp)
    return output


def h5_quad(F: Mat33, p: Mat43, dhdX: Mat43) -> Scalar:
    @wp.kernel
    @no_type_check
    def kernel(
        _F: wp.array(dtype=Any),
        p: wp.array(dtype=Any),
        dhdX: wp.array(dtype=Any),
        output: wp.array(dtype=Any),
    ) -> None:
        cid = wp.tid()
        h5_quad = func.h5_quad(p[cid], dhdX[cid])  # mat43
        output[cid] = h5_quad

    F_wp: wp.array = wpu.to_warp(F, (3, 3))
    p_wp: wp.array = wpu.to_warp(p, (4, 3))
    dhdX_wp: wp.array = wpu.to_warp(dhdX, (4, 3))
    output_wp: wp.array = wp.zeros(F_wp.shape, wp.dtype_from_jax(F.dtype))
    wp.launch(kernel, F_wp.shape, inputs=[F_wp, p_wp, dhdX_wp], outputs=[output_wp])
    output: Scalar = wp.to_jax(output_wp)
    return output


def h6_quad(F: Mat33, p: Mat43, dhdX: Mat43) -> Scalar:
    @wp.kernel
    @no_type_check
    def kernel(
        F: wp.array(dtype=Any),
        p: wp.array(dtype=Any),
        dhdX: wp.array(dtype=Any),
        output: wp.array(dtype=Any),
    ) -> None:
        cid = wp.tid()
        h6_quad = func.h6_quad(p[cid], dhdX[cid], F[cid])  # mat43
        output[cid] = h6_quad

    F_wp: wp.array = wpu.to_warp(F, (3, 3))
    p_wp: wp.array = wpu.to_warp(p, (4, 3))
    dhdX_wp: wp.array = wpu.to_warp(dhdX, (4, 3))
    output_wp: wp.array = wp.zeros(F_wp.shape, wp.dtype_from_jax(F.dtype))
    wp.launch(kernel, F_wp.shape, inputs=[F_wp, p_wp, dhdX_wp], outputs=[output_wp])
    output: Scalar = wp.to_jax(output_wp)
    return output


@hypothesis.given(
    F=testing.spd_matrix(3),
    p=testing.matrices((4, 3)),
    dhdX=testing.matrices((4, 3)),
)
def test_h4_quad(F: Mat33, p: Mat43, dhdX: Mat43) -> None:
    @wp.func
    @no_type_check
    def g1(F: Any) -> Any:
        R, _ = math.polar_rv(F)
        return func.g1(R)

    actual: Scalar = h4_quad(F, p, dhdX)
    expected: Scalar = numeric_quad(g1, F, p, dhdX)
    np.testing.assert_allclose(actual, expected, atol=1e-10)


@hypothesis.given(
    F=testing.matrices((3, 3)),
    p=testing.matrices((4, 3)),
    dhdX=testing.matrices((4, 3)),
)
def test_h5_quad(F: Mat33, p: Mat43, dhdX: Mat43) -> None:
    actual: Scalar = h5_quad(F, p, dhdX)
    expected: Scalar = numeric_quad(func.g2, F, p, dhdX)
    np.testing.assert_allclose(actual, expected, atol=1e-10)


@hypothesis.given(
    F=testing.matrices((3, 3)),
    p=testing.matrices((4, 3)),
    dhdX=testing.matrices((4, 3)),
)
def test_h6_quad(F: Mat33, p: Mat43, dhdX: Mat43) -> None:
    actual: Scalar = h6_quad(F, p, dhdX)
    expected: Scalar = numeric_quad(func.g3, F, p, dhdX)
    np.testing.assert_allclose(actual, expected, atol=1e-10)
