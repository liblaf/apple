import functools
from collections.abc import Callable, Mapping, Sequence
from typing import Any, overload

import warp as wp
import warp.types as wpt
from jaxtyping import Array
from warp.jax_experimental import ffi

type OutputDims = Mapping[str, int | Sequence[int]]


def from_jax(jax_array: Array, dtype: Any = None) -> wp.array:
    if isinstance(dtype, Sequence):
        if len(dtype) == 1:
            length: int
            (length,) = dtype
            dtype = wpt.vector(length, wp.dtype_from_jax(jax_array.dtype))
        elif len(dtype) == 2:
            shape: tuple[int, int] = tuple(dtype)
            dtype = wpt.matrix(shape, wp.dtype_from_jax(jax_array.dtype))
        else:
            raise NotImplementedError
    return wp.from_jax(jax_array, dtype)


@overload
def jax_callable(
    func: Callable,
    *,
    num_outputs: int = 1,
    graph_mode: ffi.GraphMode = ffi.GraphMode.JAX,
    output_dims: OutputDims | None = None,
    **kwargs,
) -> ffi.FfiCallable: ...
@overload
def jax_callable(
    *,
    num_outputs: int = 1,
    graph_mode: ffi.GraphMode = ffi.GraphMode.JAX,
    output_dims: OutputDims | None = None,
    **kwargs,
) -> Callable[[Callable], ffi.FfiCallable]: ...
def jax_callable(func: Callable | None = None, **kwargs) -> Any:
    if func is None:
        return functools.partial(jax_callable, **kwargs)
    return ffi.jax_callable(func, **kwargs)
