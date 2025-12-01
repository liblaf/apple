import functools
from collections.abc import Callable
from typing import Any, overload

import attrs
import jax
import numpy as np
import warp as wp
from jaxtyping import Array

type WarpDType = Any


@attrs.define
class Adapter:
    array_from: Callable[..., wp.array]
    dtype_from: Callable[[Any], Any]
    dtype_to: Callable[[Any], Any]


@overload
def to_warp(  # pyright: ignore[reportInconsistentOverload]
    arr: Array,
    dtype: int | tuple[int, int] | WarpDType | None = None,
    *,
    requires_grad: bool = ...,
) -> wp.array: ...
def to_warp(arr: Array, dtype: Any = None, **kwargs) -> wp.array:
    adapter: Adapter = _registry(arr)
    if dtype is None:
        return adapter.array_from(arr, **kwargs)
    if isinstance(dtype, int):
        length: int = dtype
        return adapter.array_from(
            arr, dtype=wp.types.vector(length, adapter.dtype_from(arr.dtype)), **kwargs
        )
    if isinstance(dtype, tuple):
        shape: tuple[int, int] = dtype
        return adapter.array_from(
            arr, dtype=wp.types.matrix(shape, adapter.dtype_from(arr.dtype)), **kwargs
        )
    return adapter.array_from(
        arr.astype(adapter.dtype_to(_type_scalar_type(dtype))), dtype, **kwargs
    )


@functools.singledispatch
def _registry(_arr: Any) -> Any:
    return Adapter(
        array_from=wp.from_numpy,
        dtype_from=wp.dtype_from_numpy,
        dtype_to=wp.dtype_to_numpy,
    )


@_registry.register(np.ndarray)
def _registry_numpy(_arr: np.ndarray) -> Adapter:
    return Adapter(
        array_from=wp.from_numpy,
        dtype_from=wp.dtype_from_numpy,
        dtype_to=wp.dtype_to_numpy,
    )


@_registry.register(jax.Array)
def _registry_jax(_arr: jax.Array) -> Adapter:
    return Adapter(
        array_from=wp.from_jax, dtype_from=wp.dtype_from_jax, dtype_to=wp.dtype_to_jax
    )


def _type_scalar_type(dtype: Any) -> Any:
    return getattr(dtype, "_wp_scalar_type_", dtype)
