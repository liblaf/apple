import functools
from collections.abc import Callable
from typing import Any, overload

import beartype
from jaxtyping import jaxtyped


@overload
def validate[T](*, typechecker: Callable = ...) -> Callable[[T], T]: ...
@overload
def validate[T](func: T, /, *, typechecker: Callable = ...) -> T: ...
def validate(func: Any = None, /, *, typechecker: Callable = beartype.beartype) -> Any:
    if func is None:
        return functools.partial(validate, typechecker=typechecker)
    return jaxtyped(func, typechecker=typechecker)
