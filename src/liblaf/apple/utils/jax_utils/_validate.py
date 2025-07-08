import functools
from collections.abc import Callable
from typing import Any, overload

import beartype
from jaxtyping import jaxtyped


@overload
def validate[T](*, typechecker: Callable = ...) -> Callable[[T], T]: ...
@overload
def validate[T](func: T, /, *, typechecker: Callable = ...) -> T: ...
def validate(func: Any = None, /, **kwargs) -> Any:
    if func is None:
        return functools.partial(validate, **kwargs)
    kwargs.setdefault("typechecker", beartype.beartype)
    return jaxtyped(func, **kwargs)
