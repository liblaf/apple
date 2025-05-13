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
    R: Float[jax.Array, "*B 3 3"]
    S: Float[jax.Array, "*B 3 3"]
    R, S = math.polar_rv(F)
    np.testing.assert_allclose(jnp.linalg.det(R), 1.0, rtol=4e-6)
    np.testing.assert_allclose(R @ S, F, rtol=4e-5)
