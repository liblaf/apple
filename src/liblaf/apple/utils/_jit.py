from collections.abc import Callable, Iterable, Sequence

import jax


def jit[**P, T](
    *,
    static_argnums: int | Sequence[int] | None = None,
    static_argnames: str | Iterable[str] | None = None,
    **kwargs,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    def decorator(fn: Callable[P, T]) -> Callable[P, T]:
        return jax.jit(
            fn, static_argnums=static_argnums, static_argnames=static_argnames, **kwargs
        )

    return decorator
