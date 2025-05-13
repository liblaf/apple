from collections.abc import Iterable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxtyping import Float, PRNGKeyArray

from liblaf.apple.jax import math


@pytest.mark.parametrize("batch", [(), (5,), (5, 7)])
def test_polar_rv(batch: Iterable[int]) -> None:
    key: PRNGKeyArray = jax.random.key(0)
    subkey: PRNGKeyArray
    key, subkey = jax.random.split(key)
    F: Float[jax.Array, "*B 3 3"] = jax.random.uniform(subkey, (*batch, 3, 3))
    U: Float[jax.Array, "*B 3 3"]
    sigma: Float[jax.Array, "*B 3"]
    V: Float[jax.Array, "*B 3 3"]
    U, sigma, V = math.svd_rv(F)
    Sigma: Float[jax.Array, "*B 3 3"] = math.diag(sigma)
    np.testing.assert_allclose(U @ Sigma @ math.transpose(V), F, rtol=3e-5)
    np.testing.assert_allclose(jnp.abs(jnp.linalg.det(U)), 1.0, rtol=3e-6)
    np.testing.assert_allclose(jnp.abs(jnp.linalg.det(V)), 1.0, rtol=3e-6)
    np.testing.assert_allclose(jnp.linalg.det(U @ V), 1.0, rtol=4e-6)
