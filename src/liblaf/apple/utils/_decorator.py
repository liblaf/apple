import functools
from collections.abc import Callable
from typing import Protocol

import jax

from liblaf import grapes


class TimerDecorator(Protocol):
    def __call__[**P, T](
        self, func: Callable[P, T], /
    ) -> grapes.TimedFunction[P, T]: ...


def timer_jax(label: str | None = None) -> TimerDecorator:
    def decorator[**P, T](func: Callable[P, T], /) -> grapes.TimedFunction[P, T]:
        return grapes.timer(label=label)(block_until_ready(func))

    return decorator


def block_until_ready[**P, T](func: Callable[P, T], /) -> Callable[P, T]:
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        args = jax.block_until_ready(args)
        kwargs = jax.block_until_ready(kwargs)
        result: T = func(*args, **kwargs)
        result = jax.block_until_ready(result)
        return result

    return wrapper
