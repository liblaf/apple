import jax
import jax.numpy as jnp
import numpy as np
import pylops
from jaxtyping import Float, PRNGKeyArray

from liblaf import apple


def test_diagonal(n: int = 7) -> None:
    key: PRNGKeyArray = jax.random.key(0)
    A: Float[jax.Array, "N N"] = jax.random.uniform(key, (n, n))
    A_op: pylops.LinearOperator = pylops.JaxOperator(pylops.MatrixMult(A))  # pyright: ignore[reportArgumentType]
    actual: Float[jax.Array, " N"] = apple.diagonal(A_op)
    expected: Float[jax.Array, " N"] = jnp.diagonal(A)
    np.testing.assert_allclose(actual, expected)
