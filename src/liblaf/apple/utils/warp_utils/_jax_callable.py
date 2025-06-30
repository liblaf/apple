import functools
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol, overload

import warp.jax_experimental.ffi
from jaxtyping import Array

type DimLike = int | Sequence[int]


class FfiCallable(Protocol):
    def __call__(
        self,
        *args,
        output_dims: DimLike | Mapping[str, DimLike] | None = None,
        vmap_method: None = None,
    ) -> Sequence[Array]: ...


@overload
def jax_callable(*, num_outputs: int = 1) -> Callable[[Callable], FfiCallable]: ...
@overload
def jax_callable(func: Callable, /, *, num_outputs: int = 1) -> FfiCallable: ...
def jax_callable(func: Callable | None = None, /, **kwargs) -> Any:
    if func is None:
        return functools.partial(jax_callable, **kwargs)
    return warp.jax_experimental.ffi.jax_callable(func, **kwargs)
