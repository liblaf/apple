import abc
from collections.abc import Callable
from typing import Protocol, Required, TypedDict, overload

import jax
import jax.numpy as jnp
from jaxtyping import Float
from numpy.typing import ArrayLike

from liblaf import grapes


class OptimizeResult(TypedDict, total=False):
    x: Float[jax.Array, " N"]
    success: Required[bool]
    fun: float
    jac: Float[jax.Array, " N"]
    hess: Float[jax.Array, "N N"]

    n_jac: int
    n_hess: int
    n_hessp: int
    n_hess_diag: int
    n_hess_quad: int
    n_jac_and_hess_diag: int


class Callback(Protocol):
    def __call__(self, intermediate_result: OptimizeResult) -> None: ...


class Optimizer(abc.ABC):
    def minimize(
        self,
        fun: Callable[..., float],
        x0: Float[ArrayLike, " N"],
        *,
        args: tuple = (),
        jac: Callable | None = None,
        hess: Callable | None = None,
        hessp: Callable | None = None,
        hess_diag: Callable | None = None,
        hess_quad: Callable | None = None,
        jac_and_hess_diag: Callable | None = None,
        callback: Callback | None = None,
        **kwargs,
    ) -> OptimizeResult:
        if callable(hess) and (not callable(hessp)):
            hessp = default_hessp(hess)
        if callable(hess) and (not callable(hess_diag)):
            hess_diag = default_hess_diag(hess)
        if callable(hessp) and (not callable(hess_quad)):
            hess_quad = default_hess_quad(hessp)
        if callable(jac) and callable(hess_diag) and (not callable(jac_and_hess_diag)):
            jac_and_hess_diag = default_jac_and_hess_diag(jac, hess_diag)
        if not callable(callback):
            callback = default_callback

        fun = wrap_func(fun, "fun")
        jac = wrap_func(jac, "jac")
        hess = wrap_func(hess, "hess")
        hessp = wrap_func(hessp, "hessp")
        hess_diag = wrap_func(hess_diag, "hess_diag")
        hess_quad = wrap_func(hess_quad, "hess_quad")
        jac_and_hess_diag = wrap_func(jac_and_hess_diag, "jac_and_hess_diag")
        callback = wrap_func(callback, "callback")

        result: OptimizeResult = self._minimize(
            fun,
            x0,
            args=args,
            jac=jac,
            hess=hess,
            hessp=hessp,
            hess_diag=hess_diag,
            hess_quad=hess_quad,
            jac_and_hess_diag=jac_and_hess_diag,
            callback=callback,
            **kwargs,
        )

        result = update_result(result, fun)
        result = update_result(result, jac)
        result = update_result(result, hess)
        result = update_result(result, hessp)
        result = update_result(result, hess_diag)
        result = update_result(result, hess_quad)
        result = update_result(result, jac_and_hess_diag)
        result = update_result(result, callback, "n_iter")

        return result

    @abc.abstractmethod
    def _minimize(
        self,
        fun: Callable[..., float],
        x0: Float[ArrayLike, " N"],
        *,
        args: tuple = (),
        jac: Callable | None = None,
        hess: Callable | None = None,
        hessp: Callable | None = None,
        hess_diag: Callable | None = None,
        hess_quad: Callable | None = None,
        jac_and_hess_diag: Callable | None = None,
        callback: Callback | None = None,
        **kwargs,
    ) -> OptimizeResult: ...


def default_hessp(hess: Callable) -> Callable:
    def hessp(
        x: Float[jax.Array, " N"], p: Float[jax.Array, " N"], *args, **kwargs
    ) -> Float[jax.Array, " N"]:
        x = jnp.asarray(x)
        p = jnp.asarray(p)
        return hess(x, *args, **kwargs) @ p

    return hessp


def default_hess_diag(hess: Callable) -> Callable:
    def hess_diag(x: Float[jax.Array, " N"], *args, **kwargs) -> Float[jax.Array, " N"]:
        x = jnp.asarray(x)
        return jnp.diagonal(hess(x, *args, **kwargs))

    return hess_diag


def default_hess_quad(hessp: Callable) -> Callable:
    def hess_quad(
        x: Float[jax.Array, " N"], p: Float[jax.Array, " N"], *args, **kwargs
    ) -> Float[jax.Array, " N"]:
        x = jnp.asarray(x)
        p = jnp.asarray(p)
        return jnp.dot(p, hessp(x, p, *args, **kwargs))

    return hess_quad


def default_jac_and_hess_diag(jac: Callable, hess_diag: Callable) -> Callable:
    def jac_and_hess_diag(
        x: Float[jax.Array, " N"], *args, **kwargs
    ) -> tuple[Float[jax.Array, " N"], Float[jax.Array, " N"]]:
        x = jnp.asarray(x)
        return jac(x, *args, **kwargs), hess_diag(x, *args, **kwargs)

    return jac_and_hess_diag


def default_callback(intermediate_result: OptimizeResult) -> None: ...


@overload
def wrap_func[**P, T](
    func: Callable[P, T], name: str
) -> grapes.TimedFunction[P, T]: ...
@overload
def wrap_func[**P, T](func: None, name: str) -> None: ...
def wrap_func[**P, T](
    func: Callable[P, T] | None, name: str
) -> grapes.TimedFunction[P, T] | None:
    if func is None:
        return None
    return grapes.timer(func, name=name)


def update_result(
    result: OptimizeResult, func: grapes.TimedFunction | None, key: str | None = None
) -> OptimizeResult:
    if func is None:
        return result
    if key is None:
        key = f"n_{func.timing.name}"
    if key not in result:
        result[key] = func.timing.count
    if callable(func.timing.callback_finally):
        func.timing.callback_finally(func.timing)
    return result
