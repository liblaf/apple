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


def inputs(n: int = 10**6) -> tuple[Float[jax.Array, " N"], Float[jax.Array, " N"]]:
    key: PRNGKeyArray = jax.random.key(0)
    subkey: PRNGKeyArray
    key, subkey = jax.random.split(key)
    a: Float[jax.Array, " N"] = jax.random.uniform(subkey, shape=(n,))
    key, subkey = jax.random.split(key)
    b: Float[jax.Array, " N"] = jax.random.uniform(subkey, shape=(n,))
    return a, b


@pytest.mark.benchmark(group="dot", warmup=True)
def test_dot_jax(benchmark: BenchmarkFixture) -> None:
    a: Float[jax.Array, " N"]
    b: Float[jax.Array, " N"]
    a, b = inputs()

    @apple.block_until_ready_decorator
    @jax.jit
    def dot_jax(
        a: Float[jax.Array, " N"], b: Float[jax.Array, " N"]
    ) -> Float[jax.Array, ""]:
        return jnp.dot(a, b)

    benchmark(dot_jax, a, b)


@pytest.mark.benchmark(group="dot", warmup=True)
def test_dot_warp(benchmark: BenchmarkFixture) -> None:
    a: Float[jax.Array, " N"]
    b: Float[jax.Array, " N"]
    a, b = inputs()

    @no_type_check
    @wp.kernel
    def dot_warp_kernel(
        a: wp.array(dtype=float),
        b: wp.array(dtype=float),
        result: wp.array(dtype=float),
    ) -> None:
        tid = wp.tid()
        wp.atomic_add(result, 0, a[tid] * b[tid])

    @no_type_check
    def dot_warp_callable(
        a: wp.array(dtype=float),
        b: wp.array(dtype=float),
        result: wp.array(dtype=float),
    ) -> None:
        result.zero_()
        wp.launch(dot_warp_kernel, dim=a.shape, inputs=(a, b), outputs=(result,))

    dot_jax_callable = ffi.jax_callable(dot_warp_callable, output_dims={"result": (1,)})

    def dot_jax(
        a: Float[jax.Array, " N"], b: Float[jax.Array, " N"]
    ) -> Float[jax.Array, ""]:
        result: Float[jax.Array, ""]
        (result,) = dot_jax_callable(a, b)
        return result

    @apple.block_until_ready_decorator
    def dot_warp(
        a: Float[jax.Array, " N"], b: Float[jax.Array, " N"]
    ) -> Float[jax.Array, ""]:
        return dot_jax(a, b)

    result: Float[jax.Array, ""] = benchmark(dot_warp, a, b)
    np.testing.assert_allclose(result, jnp.dot(a, b), rtol=2e-4)
