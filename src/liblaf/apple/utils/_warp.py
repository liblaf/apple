import functools
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, overload

import jax
import warp.jax_experimental.ffi

type DimLike = int | Sequence[int]


class JaxKernel(Protocol):
    def __call__(
        self,
        *args,
        output_dims: DimLike | Mapping[str, DimLike] | None = None,
        launch_dims: DimLike | None = None,
        vmap_method: None = None,
    ) -> Sequence[jax.Array]: ...


@overload
def jax_kernel(*, num_outputs: int = 1) -> Callable[[Callable], JaxKernel]: ...
@overload
def jax_kernel(func: Callable, /, *, num_outputs: int = 1) -> JaxKernel: ...
def jax_kernel(func: Callable | None = None, /, *, num_outputs: int = 1) -> Callable:
    if func is None:
        return functools.partial(jax_kernel, num_outputs=num_outputs)
    return warp.jax_experimental.ffi.jax_kernel(
        warp.kernel(func), num_outputs=num_outputs
    )
