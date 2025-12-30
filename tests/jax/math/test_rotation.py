import hypothesis
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from liblaf.apple.jax import math, testing

ATOL: float = 1e-7

type Mat33 = Float[Array, "*batch 3 3"]
type Vec3 = Float[Array, "*batch 3"]


@hypothesis.given(testing.matrices((3, 3)))
def test_svd_rv(F: Mat33) -> None:
    U: Mat33
    S: Vec3
    Vh: Mat33
    U, S, Vh = math.svd_rv(F)
    np.testing.assert_allclose(
        U.mT @ U, jnp.broadcast_to(jnp.identity(3), F.shape), atol=ATOL
    )
    np.testing.assert_allclose(jnp.linalg.det(U), 1.0)
    np.testing.assert_allclose(
        Vh.mT @ Vh, jnp.broadcast_to(jnp.identity(3), F.shape), atol=ATOL
    )
    np.testing.assert_allclose(jnp.linalg.det(Vh), 1.0)
    S: Mat33 = S[..., jnp.newaxis] * jnp.identity(3)
    np.testing.assert_allclose(U @ S @ Vh, F, atol=ATOL)


@hypothesis.given(testing.matrices((3, 3)))
def test_polar_rv(F: Mat33) -> None:
    R: Mat33
    S: Mat33
    R, S = math.polar_rv(F)
    np.testing.assert_allclose(
        R.mT @ R, jnp.broadcast_to(jnp.identity(3), F.shape), atol=ATOL
    )
    np.testing.assert_allclose(jnp.linalg.det(R), 1.0)
    np.testing.assert_allclose(R @ S, F, atol=ATOL)
