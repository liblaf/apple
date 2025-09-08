import hypothesis
import hypothesis.extra.numpy as hnp
import jax.numpy as jnp
import numpy as np
import warp as wp
from jaxtyping import Array, ArrayLike, Float

from liblaf.apple.warp import math

ATOL: float = 1e-7


@hypothesis.given(
    hnp.arrays(
        np.float64,
        hnp.array_shapes(min_dims=1, max_dims=1).map(lambda s: (*s, 3, 3)),
        elements=hnp.from_dtype(np.dtype(np.float16), min_value=-1.0, max_value=1.0),
    ),
)
def test_svd_rv(F: Float[ArrayLike, "... 3 3"]) -> None:
    F: Float[Array, "... 3 3"] = jnp.asarray(F)
    F_wp: wp.array = wp.from_jax(F, wp.mat33d)
    U_wp, s_wp, V_wp = wp.map(math.svd_rv, F_wp)  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    U: Float[Array, "... 3 3"] = wp.to_jax(U_wp)
    s: Float[Array, "... 3"] = wp.to_jax(s_wp)
    V: Float[Array, "... 3 3"] = wp.to_jax(V_wp)
    np.testing.assert_allclose(
        U.mT @ U, jnp.broadcast_to(jnp.identity(3), F.shape), atol=ATOL
    )
    np.testing.assert_allclose(jnp.linalg.det(U), 1.0)
    np.testing.assert_allclose(
        V @ V.mT, jnp.broadcast_to(jnp.identity(3), F.shape), atol=ATOL
    )
    np.testing.assert_allclose(jnp.linalg.det(V), 1.0)
    S: Float[Array, "... 3 3"] = s[..., jnp.newaxis] * jnp.identity(3)
    np.testing.assert_allclose(U @ S @ V.mT, F, atol=ATOL)


@hypothesis.given(
    hnp.arrays(
        np.float64,
        hnp.array_shapes(min_dims=1, max_dims=1).map(lambda s: (*s, 3, 3)),
        elements=hnp.from_dtype(np.dtype(np.float16), min_value=-1.0, max_value=1.0),
    ),
)
def test_polar_rv(F: Float[ArrayLike, "... 3 3"]) -> None:
    F: Float[Array, "... 3 3"] = jnp.asarray(F)
    F_wp: wp.array = wp.from_jax(F, wp.mat33d)
    R_wp: wp.array
    S_wp: wp.array
    R_wp, S_wp = wp.map(math.polar_rv, F_wp)  # pyright: ignore[reportAssignmentType, reportGeneralTypeIssues]
    R: Float[Array, "... 3 3"] = wp.to_jax(R_wp)
    S: Float[Array, "... 3 3"] = wp.to_jax(S_wp)
    np.testing.assert_allclose(
        R.mT @ R, jnp.broadcast_to(jnp.identity(3), F.shape), atol=ATOL
    )
    np.testing.assert_allclose(jnp.linalg.det(R), 1.0)
    np.testing.assert_allclose(R @ S, F, atol=ATOL)
