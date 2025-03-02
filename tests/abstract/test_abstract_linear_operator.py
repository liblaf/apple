import jax
import pytest
from jaxtyping import Float, PRNGKeyArray

from liblaf import apple

M: int = 3
N: int = 5
K: int = 7


@pytest.fixture(scope="package")
def A() -> Float[jax.Array, "M N"]:
    key: PRNGKeyArray = jax.random.key(0)
    A: Float[jax.Array, "M N"] = jax.random.uniform(key, (M, N))
    return A


@pytest.fixture(scope="package")
def op(A: Float[jax.Array, "M N"]) -> apple.LinearOperator:
    return apple.as_linear_operator(A)


def test_matvec(A: Float[jax.Array, "M N"], op: apple.LinearOperator) -> None:
    key: PRNGKeyArray = jax.random.key(0)
    v: Float[jax.Array, " N"] = jax.random.uniform(key, (N,))
    assert op.matvec(v) == pytest.approx(A @ v)


def test_rmatvec(A: Float[jax.Array, "M N"], op: apple.LinearOperator) -> None:
    key: PRNGKeyArray = jax.random.key(0)
    v: Float[jax.Array, " N"] = jax.random.uniform(key, (M,))
    assert op.rmatvec(v) == pytest.approx(A.T @ v)


def test_matmat(A: Float[jax.Array, "M N"], op: apple.LinearOperator) -> None:
    key: PRNGKeyArray = jax.random.key(0)
    V: Float[jax.Array, "N K"] = jax.random.uniform(key, (N, K))
    assert op.matmat(V) == pytest.approx(A @ V)


def test_rmatmat(A: Float[jax.Array, "M N"], op: apple.LinearOperator) -> None:
    key: PRNGKeyArray = jax.random.key(0)
    V: Float[jax.Array, "M K"] = jax.random.uniform(key, (M, K))
    assert op.rmatmat(V) == pytest.approx(A.T @ V)
