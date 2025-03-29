from collections.abc import Callable, Iterable, Sequence

import jax

from liblaf.grapes import Decorator


def jit(
    *,
    static_argnums: int | Sequence[int] | None = None,
    static_argnames: str | Iterable[str] | None = None,
    **kwargs,
) -> Decorator:
    def decorator[**P, T](fn: Callable[P, T]) -> Callable[P, T]:
        return jax.jit(
            fn, static_argnums=static_argnums, static_argnames=static_argnames, **kwargs
        )

    return decorator
