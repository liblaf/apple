import equinox as eqx
import hypothesis
import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Float

from liblaf.apple.jax import math, testing


@eqx.filter_jit
def rosen_hess_diag(x: Float[Array, " N"]) -> Float[Array, " N"]:
    return math.hess_diag(testing.rosen, x)


@eqx.filter_jit
def rosen_hess_prod(x: Float[Array, " N"], p: Float[Array, " N"]) -> Float[Array, " N"]:
    return math.hess_prod(testing.rosen, x, p)


shapes = hnp.array_shapes(min_dims=1, max_dims=1, min_side=2)


@hypothesis.given(
    hnp.arrays(
        dtype=np.float64,
        shape=shapes,
        elements=hnp.from_dtype(np.dtype(np.float16), min_value=0.0, max_value=2.0),
    )
)
def test_hess_diag(x: ArrayLike) -> None:
    x = jnp.asarray(x)
    diag_actual: Float[Array, " N"] = rosen_hess_diag(x)
    hess_expected: Float[Array, "N N"] = testing.rosen_hess(x)
    diag_expected: Float[Array, " N"] = jnp.diagonal(hess_expected)
    np.testing.assert_allclose(diag_actual, diag_expected)


@hypothesis.given(
    hnp.arrays(
        dtype=np.float64,
        shape=st.shared(shapes, key="x"),
        elements=hnp.from_dtype(np.dtype(np.float16), min_value=0.0, max_value=2.0),
    ),
    hnp.arrays(
        dtype=np.float64,
        shape=st.shared(shapes, key="x"),
        elements=hnp.from_dtype(np.dtype(np.float16), min_value=0.0, max_value=2.0),
    ),
)
def test_hess_prod(x: ArrayLike, p: ArrayLike) -> None:
    x = jnp.asarray(x)
    p = jnp.asarray(p)
    actual: Float[Array, " N"] = rosen_hess_prod(x, p)
    expected: Float[Array, "N N"] = testing.rosen_hess_prod(x, p)
    np.testing.assert_allclose(actual, expected)
