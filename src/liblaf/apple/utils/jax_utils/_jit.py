import functools
from collections.abc import Callable
from typing import Any, TypedDict, Unpack, overload

import equinox as eqx
import jax

from ._validate import validate


class JitKwargs(TypedDict, total=False):
    chexify: bool
    filter: bool
    inline: bool
    validate: bool | None


@overload
def jit[C: Callable](func: C, /, **kwargs: Unpack[JitKwargs]) -> C: ...
@overload
def jit[C: Callable](**kwargs: Unpack[JitKwargs]) -> Callable[[C], C]: ...
def jit(func: Callable | None = None, /, **kwargs) -> Any:
    if func is None:
        return functools.partial(jit, **kwargs)
    if kwargs.pop("validate", True):
        func = validate(func)
    if kwargs.pop("filter", False):
        func = eqx.filter_jit(func, **kwargs)
    else:
        func = jax.jit(func, **kwargs)
    return func


@overload
def jit_method[C: Callable](func: C, /, **kwargs: Unpack[JitKwargs]) -> C: ...
@overload
def jit_method[C: Callable](**kwargs: Unpack[JitKwargs]) -> Callable[[C], C]: ...
def jit_method(*args, **kwargs) -> Any:
    return jit(*args, **kwargs)
