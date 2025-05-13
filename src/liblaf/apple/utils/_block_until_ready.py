import functools
from collections.abc import Callable

import jax


def block_until_ready_decorator[**P, T](func: Callable[P, T]) -> Callable[P, T]:
    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        result: T = func(*args, **kwargs)
        return jax.block_until_ready(result)

    return wrapper
