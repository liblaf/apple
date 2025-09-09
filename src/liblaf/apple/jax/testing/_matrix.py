from collections.abc import Sequence

import hypothesis.extra.numpy as hnp
import hypothesis.strategies as st
import jax.numpy as jnp
import numpy as np
import sklearn.datasets
from jaxtyping import Array, ArrayLike, DTypeLike, Float


def random_mat33(
    min_dims: int | None = 1, max_dims: int | None = None
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
    dtype: DTypeLike = float,
    n_dim: int = 3,
    shapes: st.SearchStrategy[Sequence[int]] | None = None,
) -> Float[Array, "*batch D D"]:
    if shapes is None:
        shapes = hnp.array_shapes(min_dims=1, max_dims=1)
    shape: Sequence[int] = draw(shapes)
    integers: st.SearchStrategy[int] = st.integers(min_value=0, max_value=4294967295)
    matrices: list[Float[ArrayLike, " D D"]] = []
    for _ in range(np.prod(shape)):
        seed: int = draw(integers)
        matrix: Float[ArrayLike, " D D"] = sklearn.datasets.make_spd_matrix(
            n_dim, random_state=seed
        )
        matrices.append(matrix)
    return jnp.asarray(matrices, dtype=dtype).reshape(*shape, n_dim, n_dim)
