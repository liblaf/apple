from collections.abc import Callable

import jax
import numpy as np
import pylops
from jaxtyping import Float, PRNGKeyArray

from liblaf import apple

M: int = 5
N: int = 7


def fun(x: Float[jax.Array, "N"]) -> Float[jax.Array, "M"]:
    return jax.vmap(lambda x: x**2)(x)


def test_jvp_fun() -> None:
    key: PRNGKeyArray = jax.random.key(0)
    subkey: PRNGKeyArray
    key, subkey = jax.random.split(key)
    x: Float[jax.Array, " N"] = jax.random.uniform(subkey, (N,))
    key, subkey = jax.random.split(key)
    v: Float[jax.Array, " N"] = jax.random.uniform(subkey, (N,))
    jvp_fun: Callable[[Float[jax.Array, " N"]], Float[jax.Array, ""]] = apple.jvp_fun(
        fun, x
    )
    actual: Float[jax.Array, " M"] = jvp_fun(v)
    expected: Float[jax.Array, " M"] = jax.jvp(fun, (x,), (v,))[1]
    np.testing.assert_allclose(actual, expected)


def test_jac_as_operator() -> None:
    key: PRNGKeyArray = jax.random.key(0)
    subkey: PRNGKeyArray
    key, subkey = jax.random.split(key)
    x: Float[jax.Array, " N"] = jax.random.uniform(subkey, (N,))
    key, subkey = jax.random.split(key)
    v: Float[jax.Array, " N"] = jax.random.uniform(subkey, (N,))
    jac: pylops.LinearOperator = apple.jac_as_operator(fun, x)
    actual: Float[jax.Array, " M"] = jac @ v  # pyright: ignore[reportAssignmentType]
    expected: Float[jax.Array, " M"] = jax.jvp(fun, (x,), (v,))[1]
    np.testing.assert_allclose(actual, expected)
