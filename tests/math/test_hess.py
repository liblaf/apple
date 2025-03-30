import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Float, PRNGKeyArray

from liblaf import apple

N: int = 7


@pytest.fixture(scope="package")
def x() -> Float[jax.Array, "N"]:
    key: PRNGKeyArray = jax.random.key(0)
    return jax.random.uniform(key, (N,))


@pytest.fixture(scope="package")
def v() -> Float[jax.Array, "N"]:
    key: PRNGKeyArray = jax.random.key(1)
    return jax.random.uniform(key, (N,))


@pytest.fixture(scope="package")
def hess(x: Float[jax.Array, "N"]) -> Float[jax.Array, "N N"]:
    return jax.hessian(apple.rosen)(x)


def test_hess_diag(hess: Float[jax.Array, "N N"], x: Float[jax.Array, "N"]) -> None:
    actual: Float[jax.Array, " N"] = apple.hess_diag(apple.rosen, x)
    expected: Float[jax.Array, " N"] = jnp.diagonal(hess)
    np.testing.assert_allclose(actual, expected)


def test_hvp(
    hess: Float[jax.Array, "N N"], x: Float[jax.Array, "N"], v: Float[jax.Array, "N"]
) -> None:
    actual: Float[jax.Array, " N"] = apple.hvp(apple.rosen, x, v)
    expected: Float[jax.Array, " N"] = hess @ v
    np.testing.assert_allclose(actual, expected)


def test_hvp_op(
    hess: Float[jax.Array, "N N"], x: Float[jax.Array, "N"], v: Float[jax.Array, "N"]
) -> None:
    actual: Float[jax.Array, " N"] = apple.hvp_op(apple.rosen, x)(v)
    expected: Float[jax.Array, " N"] = hess @ v
    np.testing.assert_allclose(actual, expected)
