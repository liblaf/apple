import abc
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, overload

import jax
import jax.numpy as jnp
from jaxtyping import Float

import liblaf.apple as apple  # noqa: PLR0402
from liblaf import grapes

from . import MinimizeResult


class Callback(Protocol):
    def __call__(self, intermediate_result: MinimizeResult) -> None: ...


@overload
def timer_jax[**P, T](
    func: Callable[P, T], label: str
) -> grapes.TimedFunction[P, T]: ...
@overload
def timer_jax[**P, T](func: None, label: str) -> None: ...
def timer_jax[**P, T](
    func: Callable[P, T] | None, label: str
) -> grapes.TimedFunction[P, T] | None:
    if func is None:
        return None
    return apple.timer_jax(label=label)(func)


class MinimizeAlgorithm(abc.ABC):
    def minimize(
        self,
        fun: Callable,
        x0: Float[jax.Array, " N"],
        *,
        args: Sequence | None = None,
        kwargs: Mapping | None = None,
        jac: Callable | None = None,
        hess: Callable | None = None,
        hessp: Callable | None = None,
        hess_diag: Callable | None = None,
        hess_quad: Callable | None = None,
        jac_and_hess_diag: Callable | None = None,
        bounds: Sequence | None = None,
        callback: Callback | None = None,
    ) -> MinimizeResult:
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}
        if hess_quad is None:

            def hess_quad(
                x: Float[jax.Array, " N"], p: Float[jax.Array, " N"], *args, **kwargs
            ) -> Float[jax.Array, ""]:
                assert hessp is not None
                return jnp.vdot(p, hessp(x, p, *args, **kwargs))

        if jac_and_hess_diag is None:

            def jac_and_hess_diag(
                x: Float[jax.Array, " N"], *args, **kwargs
            ) -> tuple[Float[jax.Array, " N"], Float[jax.Array, " N"]]:
                assert jac is not None
                assert hess_diag is not None
                return jac(x, *args, **kwargs), hess_diag(x, *args, **kwargs)

        fun = timer_jax(fun, label="fun(...)")
        jac = timer_jax(jac, label="jac(...)")
        hess = timer_jax(hess, label="hess(...)")
        hessp = timer_jax(hessp, label="hessp(...)")
        hess_diag = timer_jax(hess_diag, label="hess_diag(...)")
        hess_quad = timer_jax(hess_quad, label="hess_quad(...)")
        jac_and_hess_diag = timer_jax(jac_and_hess_diag, label="jac_and_hess_diag(...)")

        @grapes.timer(label="callback()")
        def callback_wrapper(intermediate_result: MinimizeResult) -> None:
            if callback is not None:
                callback(intermediate_result)

        with grapes.timer(label="minimize") as timer:
            result: MinimizeResult = self._minimize(
                fun=fun,
                x0=x0,
                args=args,
                kwargs=kwargs,
                jac=jac,
                hess=hess,
                hessp=hessp,
                hess_diag=hess_diag,
                hess_quad=hess_quad,
                jac_and_hess_diag=jac_and_hess_diag,
                bounds=bounds,
                callback=callback_wrapper,
            )
        for key, value in timer.row(-1).items():
            result[f"time_{key}"] = value
        result["n_iter"] = callback_wrapper.count
        if fun:
            result["n_fun"] = fun.count
            fun.log_summary()
        if jac:
            result["n_jac"] = jac.count
            jac.log_summary()
        if hess:
            result["n_hess"] = hess.count
            hess.log_summary()
        if hessp:
            result["n_hessp"] = hessp.count
            hessp.log_summary()
        if hess_diag:
            result["n_hess_diag"] = hess_diag.count
            hess_diag.log_summary()
        if hess_quad:
            result["n_hess_quad"] = hess_quad.count
            hess_quad.log_summary()
        return result

    @abc.abstractmethod
    def _minimize(
        self,
        fun: Callable,
        x0: Float[jax.Array, " N"],
        *,
        args: Sequence | None = None,
        kwargs: Mapping | None = None,
        jac: Callable | None = None,
        hess: Callable | None = None,
        hessp: Callable | None = None,
        hess_diag: Callable | None = None,
        hess_quad: Callable | None = None,
        jac_and_hess_diag: Callable | None = None,
        bounds: Sequence | None = None,
        callback: Callable,
    ) -> MinimizeResult: ...
