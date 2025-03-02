import jax
import jax.numpy as jnp
import pylops
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
    assert actual == pytest.approx(expected)


def test_hess_as_opearator(
    hess: Float[jax.Array, "N N"], x: Float[jax.Array, "N"], v: Float[jax.Array, "N"]
) -> None:
    op: pylops.LinearOperator = apple.hess_as_operator(apple.rosen, x)
    actual: Float[jax.Array, " N"] = op @ v  # pyright: ignore[reportAssignmentType]
    expected: Float[jax.Array, " N"] = hess @ v
    assert actual == pytest.approx(expected)


def test_hvp(
    hess: Float[jax.Array, "N N"], x: Float[jax.Array, "N"], v: Float[jax.Array, "N"]
) -> None:
    actual: Float[jax.Array, " N"] = apple.hvp(apple.rosen, x, v)
    expected: Float[jax.Array, " N"] = hess @ v
    assert actual == pytest.approx(expected)


def test_hvp_fun(
    hess: Float[jax.Array, "N N"], x: Float[jax.Array, "N"], v: Float[jax.Array, "N"]
) -> None:
    actual: Float[jax.Array, " N"] = apple.hvp_fun(apple.rosen, x)(v)
    expected: Float[jax.Array, " N"] = hess @ v
    assert actual == pytest.approx(expected)
