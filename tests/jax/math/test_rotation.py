import equinox as eqx
import hypothesis
import hypothesis.extra.numpy as hnp
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Float

from liblaf.apple.jax import math

ATOL: float = 1e-7


@hypothesis.given(
    hnp.arrays(
        np.float64,
        hnp.array_shapes(min_dims=0).map(lambda s: (*s, 3, 3)),
        elements=hnp.from_dtype(np.dtype(np.float16), min_value=-1.0, max_value=1.0),
    ),
)
def test_svd_rv(F: Float[ArrayLike, "*batch 3 3"]) -> None:
    F: Float[Array, "*batch 3 3"] = jnp.asarray(F)
    u: Float[Array, "*batch 3 3"]
    s: Float[Array, "*batch 3"]
    vh: Float[Array, "*batch 3 3"]
    u, s, vh = eqx.filter_jit(math.svd_rv)(F)
    np.testing.assert_allclose(
        u.mT @ u, jnp.broadcast_to(jnp.identity(3), F.shape), atol=ATOL
    )
    np.testing.assert_allclose(jnp.linalg.det(u), 1.0)
    np.testing.assert_allclose(
        vh @ vh.mT, jnp.broadcast_to(jnp.identity(3), F.shape), atol=ATOL
    )
    np.testing.assert_allclose(jnp.linalg.det(vh), 1.0)
    S: Float[Array, "*batch 3 3"] = s[..., jnp.newaxis] * jnp.identity(3)
    np.testing.assert_allclose(u @ S @ vh, F, atol=ATOL)


@hypothesis.given(
    hnp.arrays(
        np.float64,
        hnp.array_shapes(min_dims=0).map(lambda s: (*s, 3, 3)),
        elements=hnp.from_dtype(np.dtype(np.float16), min_value=-1.0, max_value=1.0),
    ),
)
def test_polar_rv(F: Float[ArrayLike, "*batch 3 3"]) -> None:
    F: Float[Array, "*batch 3 3"] = jnp.asarray(F)
    R: Float[Array, "*batch 3 3"]
    S: Float[Array, "*batch 3 3"]
    R, S = eqx.filter_jit(math.polar_rv)(F)
    np.testing.assert_allclose(
        R.mT @ R, jnp.broadcast_to(jnp.identity(3), F.shape), atol=ATOL
    )
    np.testing.assert_allclose(jnp.linalg.det(R), 1.0)
    np.testing.assert_allclose(R @ S, F, atol=ATOL)
