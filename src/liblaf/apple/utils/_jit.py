import functools
from collections.abc import Callable, Iterable, Sequence
from typing import TypedDict, Unpack, overload

import jax

from liblaf.grapes.typed import Decorator


class JitKwargs(TypedDict, total=False):
    static_argnums: int | Sequence[int] | None
    static_argnames: str | Iterable[str] | None


@overload
def jit(**kwargs: Unpack[JitKwargs]) -> Decorator: ...
@overload
def jit[**P, T](
    func: Callable[P, T], /, **kwargs: Unpack[JitKwargs]
) -> Callable[P, T]: ...
def jit[**P, T](
    func: Callable[P, T] | None = None, /, **kwargs: Unpack[JitKwargs]
) -> Callable[P, T] | Decorator:
    if func is None:
        return functools.partial(jax.jit, **kwargs)
    return jax.jit(func, **kwargs)
