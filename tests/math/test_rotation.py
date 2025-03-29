import einops
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Float, PRNGKeyArray

from liblaf import apple


def test_svd_rv(n: int = 7) -> None:
    key: PRNGKeyArray = jax.random.key(0)
    F: Float[jax.Array, "*c 3 3"] = jax.random.uniform(key, (n, 3, 3))
    U: Float[jax.Array, "*c 3 3"]
    S_diag: Float[jax.Array, "*c 3"]
    VH: Float[jax.Array, "*c 3 3"]
    U, S_diag, VH = jax.jit(apple.math.svd_rv)(F)
    S: Float[jax.Array, "*c 3 3"] = jax.vmap(jnp.diagflat)(S_diag)
    F_actual: Float[jax.Array, "*c 3 3"] = U @ S @ VH
    np.testing.assert_allclose(F_actual, F)
    identity: Float[jax.Array, "*c 3 3"] = einops.repeat(
        jnp.identity(3), "i j -> c i j", c=n
    )
    np.testing.assert_allclose(U.swapaxes(-2, -1) @ U, identity, atol=4e-16)
    np.testing.assert_allclose(VH @ VH.swapaxes(-2, -1), identity, atol=4e-16)
    np.testing.assert_allclose(jnp.linalg.det(U @ VH), jnp.ones((n,)))


def test_polar_rv(n: int = 7) -> None:
    key: PRNGKeyArray = jax.random.key(0)
    F: Float[jax.Array, "*c 3 3"] = jax.random.uniform(key, (n, 3, 3))
    R: Float[jax.Array, "*c 3 3"]
    S: Float[jax.Array, "*c 3 3"]
    R, S = jax.jit(apple.math.polar_rv)(F)
    F_actual: Float[jax.Array, "*c 3 3"] = R @ S
    np.testing.assert_allclose(F_actual, F)
    identity: Float[jax.Array, "*c 3 3"] = einops.repeat(
        jnp.identity(3), "i j -> c i j", c=n
    )
    np.testing.assert_allclose(R @ R.swapaxes(-2, -1), identity, atol=2e-15)
    np.testing.assert_allclose(jnp.linalg.det(R), jnp.ones((n,)))
