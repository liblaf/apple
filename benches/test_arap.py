from typing import no_type_check

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import warp as wp
from jaxtyping import Float, PRNGKeyArray
from pytest_benchmark.fixture import BenchmarkFixture
from warp.jax_experimental import ffi

from liblaf import apple

pytestmark = pytest.mark.skipif(
    not wp.is_cuda_available(), reason="CUDA is not available"
)


def inputs(n: int = 10**5) -> Float[jax.Array, " N 3 3"]:
    key: PRNGKeyArray = jax.random.key(0)
    subkey: PRNGKeyArray
    key, subkey = jax.random.split(key)
    F: Float[jax.Array, " N"] = jax.random.uniform(subkey, shape=(n, 3, 3))
    return F


def svd_rv(
    F: Float[jax.Array, "3 3"],
) -> tuple[Float[jax.Array, "3 3"], Float[jax.Array, " 3"], Float[jax.Array, "3 3"]]:
    U: Float[jax.Array, "3 3"]
    sigma: Float[jax.Array, " 3"]
    VH: Float[jax.Array, "3 3"]
    U, sigma, VH = jnp.linalg.svd(F, full_matrices=False)
    detU: Float[jax.Array, ""] = jnp.linalg.det(U)
    detV: Float[jax.Array, ""] = jnp.linalg.det(VH)
    U = U.at[:, 2].set(U[:, 2] * detU)
    VH = VH.at[2, :].set(VH[2, :] * detV)
    sigma = sigma.at[2].set(sigma[2] * detU * detV)
    return U, sigma, VH


def polar_rv(
    F: Float[jax.Array, "3 3"],
) -> tuple[Float[jax.Array, "3 3"], Float[jax.Array, "3 3"]]:
    U: Float[jax.Array, "3 3"]
    sigma: Float[jax.Array, " 3"]
    VH: Float[jax.Array, "3 3"]
    U, sigma, VH = svd_rv(F)
    R: Float[jax.Array, "3 3"] = U @ VH
    Sigma: Float[jax.Array, "3 3"] = jnp.diagflat(sigma)
    S: Float[jax.Array, "3 3"] = VH.T @ Sigma @ VH
    return R, S


def arap(F: Float[jax.Array, "N 3 3"]) -> Float[jax.Array, ""]:
    R: Float[jax.Array, "N 3 3"]
    R, _S = polar_rv(F)
    return jnp.sum((F - R) ** 2)


@apple.block_until_ready_decorator
@jax.jit
def arap_jax(F: Float[jax.Array, "N 3 3"]) -> Float[jax.Array, " N"]:
    return jax.vmap(arap)(F)


@pytest.mark.benchmark(group="ARAP", warmup=True)
def test_arap_jax(benchmark: BenchmarkFixture) -> None:
    F: Float[jax.Array, "N 3 3"]
    F = inputs()

    benchmark(arap_jax, F)


@pytest.mark.benchmark(group="ARAP", warmup=True)
def test_arap_warp(benchmark: BenchmarkFixture) -> None:
    F: Float[jax.Array, "N 3 3"]
    F = inputs()

    @no_type_check
    @wp.func
    def svd_rv(F: wp.mat33):  # noqa: ANN202
        U = wp.mat33()
        sigma = wp.vec3()
        V = wp.mat33()
        wp.svd3(F, U, sigma, V)
        L = wp.identity(3, float)
        L[2, 2] = wp.determinant(U @ wp.transpose(V))
        detU = wp.determinant(U)
        detV = wp.determinant(V)
        if (detU < 0) and (detV > 0):
            U = U @ L
        elif (detU > 0) and (detV < 0):
            V = V @ L
        sigma[2] = sigma[2] * L[2, 2]
        return U, sigma, V

    @no_type_check
    @wp.func
    def polar_rv(F: wp.mat33):  # noqa: ANN202
        U = wp.mat33()
        sigma = wp.vec3()
        V = wp.mat33()
        U, sigma, V = svd_rv(F)
        R = U @ wp.transpose(V)
        S = V @ wp.diag(sigma) @ wp.transpose(V)
        return R, S

    @no_type_check
    @wp.func
    def arap(F: wp.mat33) -> float:
        R = wp.mat33()
        R, S = polar_rv(F)
        result = 0.0
        for i in range(3):
            for j in range(3):
                result += (F[i, j] - R[i, j]) * (F[i, j] - R[i, j])
        return result

    @no_type_check
    @wp.kernel
    def arap_warp_kernel(
        F: wp.array(dtype=wp.mat33), result: wp.array(dtype=float)
    ) -> None:
        tid = wp.tid()
        result[tid] = arap(F[tid])

    arap_jax_kernel = ffi.jax_kernel(arap_warp_kernel)

    @apple.block_until_ready_decorator
    @jax.jit
    def arap_warp(F: Float[jax.Array, "N 3 3"]) -> Float[jax.Array, " N"]:
        result: Float[jax.Array, " N"]
        (result,) = arap_jax_kernel(F)
        return result

    result: Float[jax.Array, " N"] = benchmark(arap_warp, F)
    np.testing.assert_allclose(result, arap_jax(F), rtol=3e-3)
