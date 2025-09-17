from collections.abc import Sequence
from typing import Any

import attrs
import numpy as np
import warp as wp
import warp.types as wpt
from jaxtyping import Array

from liblaf import grapes


@attrs.define
class MatrixLike:
    shape: tuple[int, int]

    def __init__(self, rows: int, cols: int) -> None:
        self.__attrs_init__(shape=(rows, cols))  # pyright: ignore[reportAttributeAccessIssue]


@attrs.define
class VectorLike:
    length: int


def to_warp(
    a: Any,
    dtype: Any | None = None,
    shape: Sequence[int] | None = None,
    *,
    requires_grad: bool = False,
) -> wp.array:
    match grapes.array_kind(a):
        case "numpy":
            assert isinstance(a, np.ndarray)
            if isinstance(dtype, MatrixLike):
                dtype = wpt.matrix(dtype.shape, wp.dtype_from_numpy(a.dtype))
            elif isinstance(dtype, VectorLike):
                dtype = wpt.vector(dtype.length, wp.dtype_from_numpy(a.dtype))
            return wp.from_numpy(
                a, dtype=dtype, shape=shape, requires_grad=requires_grad
            )
        case "jax":
            assert isinstance(a, Array)
            if isinstance(dtype, MatrixLike):
                dtype = wpt.matrix(dtype.shape, wp.dtype_from_jax(a.dtype))
            elif isinstance(dtype, VectorLike):
                dtype = wpt.vector(dtype.length, wp.dtype_from_jax(a.dtype))
            a_wp: wp.array = wp.from_jax(a, dtype=dtype)
            a_wp.requires_grad = requires_grad
            return a_wp
        case _:
            return wp.from_numpy(
                np.asarray(a), dtype=dtype, shape=shape, requires_grad=requires_grad
            )
