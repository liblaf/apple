import functools
from collections.abc import Callable, Mapping, Sequence

import jax
import jax.flatten_util
import jax.numpy as jnp
from jaxtyping import Float, PyTree


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


def hess_diag(func: Callable) -> Callable:
    def hess_diag(x: PyTree, /, *args, **kwargs) -> PyTree:
        x_ravel: Float[jax.Array, " N"]
        unravel: Callable[[jax.Array], PyTree]
        x_ravel, unravel = jax.flatten_util.ravel_pytree(x)

        def fun_ravel(x_ravel: jax.Array, /) -> jax.Array:
            x: PyTree = unravel(x_ravel)
            return func(x, *args, **kwargs)

        hess: Float[jax.Array, "N N"] = jax.hessian(fun_ravel)(x_ravel)
        hess_diag_ravel: Float[jax.Array, " N"] = jnp.diagonal(hess)
        hess_diag: PyTree = unravel(hess_diag_ravel)
        return hess_diag

    return hess_diag
