import einops
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Float, PRNGKeyArray

from liblaf import apple


def test_svd_rv(n: int = 10) -> None:
    key: PRNGKeyArray = jax.random.key(0)
    F: Float[jax.Array, "*c 3 3"] = jax.random.uniform(key, (n, 3, 3))
    U: Float[jax.Array, "*c 3 3"]
    S_diag: Float[jax.Array, "*c 3"]
    VH: Float[jax.Array, "*c 3 3"]
    U, S_diag, VH = jax.jit(apple.math.svd_rv)(F)
    S: Float[jax.Array, "*c 3 3"] = jax.vmap(jnp.diagflat)(S_diag)
    F_actual: Float[jax.Array, "*c 3 3"] = U @ S @ VH
    assert F_actual == pytest.approx(F)
    identity: Float[jax.Array, "*c 3 3"] = einops.repeat(
        jnp.identity(3), "i j -> c i j", c=n
    )
    assert U.swapaxes(-2, -1) @ U == pytest.approx(identity)
    assert VH @ VH.swapaxes(-2, -1) == pytest.approx(identity)
    assert jnp.linalg.det(U @ VH) == pytest.approx(jnp.ones((n,)))


def test_polar_rv(n: int = 10) -> None:
    key: PRNGKeyArray = jax.random.key(0)
    F: Float[jax.Array, "*c 3 3"] = jax.random.uniform(key, (n, 3, 3))
    R: Float[jax.Array, "*c 3 3"]
    S: Float[jax.Array, "*c 3 3"]
    R, S = jax.jit(apple.math.polar_rv)(F)
    F_actual: Float[jax.Array, "*c 3 3"] = R @ S
    assert F_actual == pytest.approx(F)
    identity: Float[jax.Array, "*c 3 3"] = einops.repeat(
        jnp.identity(3), "i j -> c i j", c=n
    )
    assert R.swapaxes(-2, -1) @ R == pytest.approx(identity)
    detR: Float[jax.Array, "*c"] = jnp.linalg.det(R)
    assert detR == pytest.approx(jnp.ones((n,)))
