import hypothesis
import jax.numpy as jnp
import numpy as np
import warp as wp
from jaxtyping import Array, ArrayLike, Float

from liblaf.apple.jax import testing
from liblaf.apple.warp import math

ATOL: float = 1e-7


@hypothesis.given(testing.random_mat33(max_dims=1))
def test_svd_rv(F: Float[ArrayLike, "batch 3 3"]) -> None:
    F: Float[Array, "batch 3 3"] = jnp.asarray(F)
    F_wp: wp.array = wp.from_jax(F, wp.mat33d)
    U_wp, s_wp, V_wp = wp.map(math.svd_rv, F_wp)  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    U: Float[Array, "batch 3 3"] = wp.to_jax(U_wp)
    s: Float[Array, "batch 3"] = wp.to_jax(s_wp)
    V: Float[Array, "batch 3 3"] = wp.to_jax(V_wp)
    np.testing.assert_allclose(
        U.mT @ U, jnp.broadcast_to(jnp.identity(3), F.shape), atol=ATOL
    )
    np.testing.assert_allclose(jnp.linalg.det(U), 1.0)
    np.testing.assert_allclose(
        V @ V.mT, jnp.broadcast_to(jnp.identity(3), F.shape), atol=ATOL
    )
    np.testing.assert_allclose(jnp.linalg.det(V), 1.0)
    S: Float[Array, "batch 3 3"] = s[..., jnp.newaxis] * jnp.identity(3)
    np.testing.assert_allclose(U @ S @ V.mT, F, atol=ATOL)


@hypothesis.given(testing.random_mat33(max_dims=1))
def test_polar_rv(F: Float[ArrayLike, "batch 3 3"]) -> None:
    F: Float[Array, "batch 3 3"] = jnp.asarray(F)
    F_wp: wp.array = wp.from_jax(F, wp.mat33d)
    R_wp: wp.array
    S_wp: wp.array
    R_wp, S_wp = wp.map(math.polar_rv, F_wp)  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    R: Float[Array, "batch 3 3"] = wp.to_jax(R_wp)
    S: Float[Array, "batch 3 3"] = wp.to_jax(S_wp)
    np.testing.assert_allclose(
        R.mT @ R, jnp.broadcast_to(jnp.identity(3), F.shape), atol=ATOL
    )
    np.testing.assert_allclose(jnp.linalg.det(R), 1.0)
    np.testing.assert_allclose(R @ S, F, atol=ATOL)
