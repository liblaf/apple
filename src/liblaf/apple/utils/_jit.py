from collections.abc import Callable

import jax


def jit[**P, T](**kwargs) -> Callable[[Callable[P, T]], Callable[P, T]]:
    def decorator(fn: Callable[P, T]) -> Callable[P, T]:
        return jax.jit(fn, **kwargs)

    return decorator
