import functools
from collections.abc import Callable, Mapping, Sequence

import jax
from jaxtyping import PyTree


def partial[T](func: Callable[..., T], /, *args, **kwargs) -> Callable[..., T]:
    partial_args: Sequence = args
    partial_kwargs: Mapping = kwargs

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> T:
        kwargs: Mapping = {**partial_kwargs, **kwargs}
        return func(*args, *partial_args, **kwargs)

    return wrapper


def jvp(func: Callable) -> Callable:
    def jvp(x: PyTree, p: PyTree, /, *args, **kwargs) -> PyTree:
        fun: Callable = partial(func, *args, **kwargs)
        _, tangents_out = jax.jvp(fun, (x,), (p,))
        return tangents_out

    return jvp


def hessp(func: Callable) -> Callable:
    def hessp(x: PyTree, p: PyTree, /, *args, **kwargs) -> PyTree:
        fun: Callable = partial(func, *args, **kwargs)
        _, tangents_out = jax.jvp(jax.grad(fun), (x,), (p,))
        return tangents_out

    return hessp
