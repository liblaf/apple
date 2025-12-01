from collections.abc import Sequence

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import jax
import numpy as np
from jaxtyping import Array, DTypeLike, Float, Key

from ._random import seed


@st.composite
def matrices(
    draw: st.DrawFn,
    shape: Sequence[int],
    dtype: DTypeLike = np.float64,
    *,
    min_dims: int = 1,
    max_dims: int | None = 1,
) -> Array:
    batch: Sequence[int] = draw(
        st.shared(hnp.array_shapes(min_dims=min_dims, max_dims=max_dims), key="batch")
    )
    key: Key = jax.random.key(draw(seed()))
    arr: Array = jax.random.uniform(
        key, (*batch, *shape), dtype, minval=-1.0, maxval=1.0
    )
    return arr


@st.composite
def spd_matrix(
    draw: st.DrawFn,
    n: int = 3,
    dtype: DTypeLike = np.float64,
    *,
    min_dims: int = 1,
    max_dims: int | None = 1,
) -> Float[Array, "*batch D D"]:
    shape: Sequence[int] = draw(
        st.shared(hnp.array_shapes(min_dims=min_dims, max_dims=max_dims), key="batch")
    )
    key: Key = jax.random.key(draw(seed()))
    arr: Array = jax.random.uniform(key, (*shape, n, n), dtype, minval=-1.0, maxval=1.0)
    arr = 0.5 * (arr.mT + arr) + 2.0 * n * np.identity(n, dtype)
    return arr
