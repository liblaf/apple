import functools
from typing import Any

import jax
import numpy as np
import warp as wp


@functools.singledispatch
def to_warp(arr: Any, dtype: Any = None) -> wp.array:
    arr: np.ndarray = np.asarray(arr)
    if dtype is None:
        return wp.from_numpy(arr)
    if isinstance(dtype, int):
        return wp.from_numpy(
            arr, dtype=wp.types.vector(dtype, wp.dtype_from_numpy(arr.dtype))
        )
    if isinstance(dtype, tuple):
        return wp.from_numpy(
            arr, dtype=wp.types.matrix(dtype, wp.dtype_from_numpy(arr.dtype))
        )
    return wp.from_jax(
        arr.astype(wp.dtype_to_numpy(wp.types.type_scalar_type(dtype))), dtype
    )


@to_warp.register(jax.Array)
def jax_to_warp(arr: jax.Array, dtype: Any = None) -> wp.array:
    if dtype is None:
        return wp.from_jax(arr)
    if isinstance(dtype, int):
        return wp.from_jax(
            arr, dtype=wp.types.vector(dtype, wp.dtype_from_jax(arr.dtype))
        )
    if isinstance(dtype, tuple):
        return wp.from_jax(
            arr, dtype=wp.types.matrix(dtype, wp.dtype_from_jax(arr.dtype))
        )
    return wp.from_jax(
        arr.astype(wp.dtype_to_jax(wp.types.type_scalar_type(dtype))), dtype
    )
