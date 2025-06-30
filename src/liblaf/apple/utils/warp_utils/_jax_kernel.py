import functools
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, overload

import warp.jax_experimental.ffi
from jaxtyping import Array

type DimLike = int | Sequence[int]


class FfiKernel(Protocol):
    def __call__(
        self,
        *args,
        output_dims: DimLike | Mapping[str, DimLike] | None = None,
        launch_dims: DimLike | None = None,
        vmap_method: None = None,
    ) -> Sequence[Array]: ...


@overload
def jax_kernel(*, num_outputs: int = 1) -> Callable[[Callable], FfiKernel]: ...
@overload
def jax_kernel(func: Callable, /, *, num_outputs: int = 1) -> FfiKernel: ...
def jax_kernel(func: Callable | None = None, /, **kwargs) -> Callable:
    if func is None:
        return functools.partial(jax_kernel, **kwargs)
    return warp.jax_experimental.ffi.jax_kernel(warp.kernel(func), **kwargs)
