from typing import Protocol

import hypothesis
import jax.numpy as jnp
import jax.test_util
import warp as wp
from jaxtyping import Array, Float
from liblaf.apple.warp.energies.elastic.hyperelastic import func

from liblaf.apple.jax import testing
from liblaf.apple.warp import math
from liblaf.apple.warp import utils as wpu


class IdentityFunction(Protocol):
    def __call__(self, F: Float[Array, "batch 3 3"]) -> Float[Array, " batch"]: ...


class IdentityGradient(Protocol):
    def __call__(self, F: Float[Array, "batch 3 3"]) -> Float[Array, "batch 3 3"]: ...


class IdentityJvp(Protocol):
    def __call__(
        self,
        primals: tuple[Float[Array, "batch 3 3"]],
        tangents: tuple[Float[Array, "batch 3 3"]],
    ) -> tuple[Float[Array, " batch"], Float[Array, " batch"]]: ...


def identity_jvp(fun: IdentityFunction, grad: IdentityGradient) -> IdentityJvp:
    def jvp(
        primals: tuple[Float[Array, "batch 3 3"]],
        tangents: tuple[Float[Array, "batch 3 3"]],
    ) -> tuple[Float[Array, " batch"], Float[Array, " batch"]]:
        F: Float[Array, "batch 3 3"]
        (F,) = primals
        dF: Float[Array, "batch 3 3"]
        (dF,) = tangents
        primals_out: Float[Array, " batch"] = fun(F)
        tangents_out: Float[Array, " batch"] = jnp.sum(grad(F) * dF, axis=(-2, -1))
        return primals_out, tangents_out

    return jvp


def I1(F: Float[Array, "batch 3 3"]) -> Float[Array, " batch"]:
    F_wp: wp.array = wpu.to_warp(F, (3, 3))
    S_wp: wp.array
    _, S_wp = wp.map(math.polar_rv, F_wp)  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    I1_wp: wp.array = wp.map(func.I1, S_wp)  # pyright: ignore[reportAssignmentType]
    I1: Float[Array, " batch"] = wp.to_jax(I1_wp)
    return I1


def I2(F: Float[Array, "batch 3 3"]) -> Float[Array, " batch"]:
    F_wp: wp.array = wpu.to_warp(F, (3, 3))
    I2_wp: wp.array = wp.map(func.I2, F_wp)  # pyright: ignore[reportAssignmentType]
    I2: Float[Array, " batch"] = wp.to_jax(I2_wp)
    return I2


def I3(F: Float[Array, "batch 3 3"]) -> Float[Array, " batch"]:
    F_wp: wp.array = wpu.to_warp(F, (3, 3))
    I3_wp: wp.array = wp.map(func.I3, F_wp)  # pyright: ignore[reportAssignmentType]
    I3: Float[Array, " batch"] = wp.to_jax(I3_wp)
    return I3


def g1(F: Float[Array, "batch 3 3"]) -> Float[Array, " batch 3 3"]:
    F_wp: wp.array = wpu.to_warp(F, (3, 3))
    R_wp: wp.array
    R_wp, _ = wp.map(math.polar_rv, F_wp)  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    g1_wp: wp.array = wp.map(func.g1, R_wp)  # pyright: ignore[reportAssignmentType]
    g1: Float[Array, "batch 3 3"] = wp.to_jax(g1_wp)
    return g1


def g2(F: Float[Array, "batch 3 3"]) -> Float[Array, " batch 3 3"]:
    F_wp: wp.array = wpu.to_warp(F, (3, 3))
    g2_wp: wp.array = wp.map(func.g2, F_wp)  # pyright: ignore[reportAssignmentType]
    g2: Float[Array, "batch 3 3"] = wp.to_jax(g2_wp)
    return g2


def g3(F: Float[Array, "batch 3 3"]) -> Float[Array, " batch 3 3"]:
    F_wp: wp.array = wpu.to_warp(F, (3, 3))
    g3_wp: wp.array = wp.map(func.g3, F_wp)  # pyright: ignore[reportAssignmentType]
    g3: Float[Array, "batch 3 3"] = wp.to_jax(g3_wp)
    return g3


def check_gradients(
    fun: IdentityFunction, grad: IdentityGradient, F: Float[Array, "batch 3 3"]
) -> None:
    I_jvp: IdentityJvp = identity_jvp(fun, grad)
    if jnp.isdtype(F.dtype, jnp.float32):
        jax.test_util.check_jvp(fun, I_jvp, (F,), atol=0.1, rtol=0.1)
    else:
        jax.test_util.check_jvp(fun, I_jvp, (F,))


@hypothesis.given(testing.spd_matrix(3))
def test_g1(F: Float[Array, "batch 3 3"]) -> None:
    check_gradients(I1, g1, F)


@hypothesis.given(testing.spd_matrix(3))
def test_g2(F: Float[Array, "batch 3 3"]) -> None:
    check_gradients(I2, g2, F)


@hypothesis.given(testing.spd_matrix(3))
def test_g3(F: Float[Array, "batch 3 3"]) -> None:
    check_gradients(I3, g3, F)
