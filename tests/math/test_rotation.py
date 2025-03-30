import einops
import jax
import jax.numpy as jnp
import numpy as np
import scipy
import scipy.differentiate
from jaxtyping import Bool, Float, PRNGKeyArray

from liblaf import apple


def test_svd_rv(n: int = 7) -> None:
    key: PRNGKeyArray = jax.random.key(0)
    F: Float[jax.Array, "*C 3 3"] = jax.random.uniform(key, (n, 3, 3))
    U: Float[jax.Array, "*C 3 3"]
    S_diag: Float[jax.Array, "*c 3"]
    VH: Float[jax.Array, "*C 3 3"]
    U, S_diag, VH = jax.jit(apple.math.svd_rv)(F)
    S: Float[jax.Array, "*C 3 3"] = jax.vmap(jnp.diagflat)(S_diag)
    F_actual: Float[jax.Array, "*C 3 3"] = U @ S @ VH
    np.testing.assert_allclose(F_actual, F)
    identity: Float[jax.Array, "*C 3 3"] = einops.repeat(
        jnp.identity(3), "i j -> c i j", c=n
    )
    np.testing.assert_allclose(U.swapaxes(-2, -1) @ U, identity, atol=4e-16)
    np.testing.assert_allclose(VH @ VH.swapaxes(-2, -1), identity, atol=4e-16)
    np.testing.assert_allclose(jnp.linalg.det(U @ VH), jnp.ones((n,)))


def test_polar_rv(n: int = 7) -> None:
    key: PRNGKeyArray = jax.random.key(0)
    F: Float[jax.Array, "*C 3 3"] = jax.random.uniform(key, (n, 3, 3))
    R: Float[jax.Array, "*C 3 3"]
    S: Float[jax.Array, "*C 3 3"]
    R, S = jax.jit(apple.math.polar_rv)(F)
    F_actual: Float[jax.Array, "*C 3 3"] = R @ S
    np.testing.assert_allclose(F_actual, F)
    identity: Float[jax.Array, "*C 3 3"] = einops.repeat(
        jnp.identity(3), "i j -> c i j", c=n
    )
    np.testing.assert_allclose(R @ R.swapaxes(-2, -1), identity, atol=2e-15)
    np.testing.assert_allclose(jnp.linalg.det(R), jnp.ones((n,)))


def test_polar_rv_jac(n: int = 1) -> None:
    def fun_flat(x: Float[jax.Array, " C*9"]) -> Float[jax.Array, " C*18"]:
        x = jnp.asarray(x)
        F: Float[jax.Array, "C 3 3"] = x.reshape(n, 3, 3)
        R: Float[jax.Array, "C 3 3"]
        S: Float[jax.Array, "C 3 3"]
        R, S = apple.math.polar_rv(F)
        result: Float[jax.Array, "C 18"] = jnp.concatenate(
            [R.reshape(n, 9), S.reshape(n, 9)], axis=-1
        )
        return result.reshape(n * 18)

    def fun_flat_batch(
        x_flat_batch: Float[jax.Array, " C*9 *B"],
    ) -> Float[jax.Array, " C*18 *B"]:
        return jnp.apply_along_axis(fun_flat, axis=0, arr=x_flat_batch)

    key: PRNGKeyArray = jax.random.key(0)
    F: Float[jax.Array, "*C 3 3"] = jax.random.uniform(key, (n, 3, 3))
    actual: Float[jax.Array, "*C 3 3"] = jax.jacobian(fun_flat)(F.ravel())
    result = scipy.differentiate.jacobian(fun_flat_batch, F.ravel())
    success: Bool[np.ndarray, "C*9 C*18"] = result["success"]
    assert jnp.count_nonzero(success) > 0.9 * success.size
    expected: np.ndarray = result["df"]
    np.testing.assert_allclose(actual[success], expected[success])
