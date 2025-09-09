import functools
from collections.abc import Callable, Mapping, Sequence
from typing import Any, overload

from warp.jax_experimental import ffi

type OutputDims = Mapping[str, int | Sequence[int]]


@overload
def jax_callable(
    func: Callable,
    *,
    num_outputs: int = 1,
    output_dims: OutputDims | None = None,
    **kwargs,
) -> ffi.FfiCallable: ...
@overload
def jax_callable(
    *, num_outputs: int = 1, output_dims: OutputDims | None = None, **kwargs
) -> Callable[[Callable], ffi.FfiCallable]: ...
def jax_callable(func: Callable | None = None, **kwargs) -> Any:
    if func is None:
        return functools.partial(jax_callable, **kwargs)
    return ffi.jax_callable(func, **kwargs)
