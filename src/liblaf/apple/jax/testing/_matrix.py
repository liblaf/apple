from collections.abc import Sequence

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, DTypeLike, Float


def random_mat33(
    min_dims: int = 1, max_dims: int | None = None
) -> st.SearchStrategy[Float[Array, "*batch 3 3"]]:
    return hnp.arrays(
        np.float64,
        hnp.array_shapes(min_dims=min_dims, max_dims=max_dims).map(
            lambda s: (*s, 3, 3)
        ),
        elements=hnp.from_dtype(np.dtype(np.float16), min_value=-1.0, max_value=1.0),
    ).map(jnp.asarray)


@st.composite
def random_spd_matrix(
    draw: st.DrawFn,
    dtypes: st.SearchStrategy[DTypeLike] | None = None,
    n_dim: int = 3,
    shapes: st.SearchStrategy[Sequence[int]] | None = None,
) -> Float[Array, "*batch D D"]:
    if dtypes is None:
        dtypes = hnp.floating_dtypes(endianness="=", sizes=[32, 64])
    if shapes is None:
        shapes = hnp.array_shapes(min_dims=1, max_dims=1)
    dtype: DTypeLike = draw(dtypes)
    shape: Sequence[int] = draw(shapes)
    A: Float[np.ndarray, "*batch D D"] = draw(
        hnp.arrays(
            dtype,
            (*shape, n_dim, n_dim),
            elements=hnp.from_dtype(np.dtype(np.float16), min_value=1.0, max_value=2.0),
        )
    )
    X: Float[np.ndarray, "*batch D D"] = 0.5 * (A.mT + A) + n_dim * np.identity(
        n_dim, dtype
    )
    return jnp.asarray(X, dtype)
