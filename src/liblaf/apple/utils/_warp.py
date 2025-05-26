import functools
from collections.abc import Callable
from typing import overload

import warp.jax_experimental.ffi


@overload
def jax_kernel(*, num_outputs: int = 1) -> Callable: ...
@overload
def jax_kernel(fn: Callable, /, *, num_outputs: int = 1) -> Callable: ...
def jax_kernel(fn: Callable | None = None, /, *, num_outputs: int = 1) -> Callable:
    if fn is None:
        return functools.partial(jax_kernel, num_outputs=num_outputs)
    return warp.jax_experimental.ffi.jax_kernel(
        warp.kernel(fn), num_outputs=num_outputs
    )
