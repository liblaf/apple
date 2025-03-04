import functools
from collections.abc import Callable
from typing import Protocol

import jax


class Decorator(Protocol):
    def __call__[**P, T](self, fn: Callable[P, T]) -> Callable[P, T]: ...


def block_until_ready() -> Decorator:
    def decorator[**P, T](fn: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(fn)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
            args = jax.block_until_ready(args)
            kwargs = jax.block_until_ready(kwargs)
            result: T = fn(*args, **kwargs)
            result = jax.block_until_ready(result)
            return result

        return wrapped

    return decorator
