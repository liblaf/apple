import abc
from collections.abc import Callable
from typing import NoReturn, Protocol

import jax
import scipy.optimize
from jaxtyping import Float

from liblaf import grapes


def NOT_IMPLEMENTED() -> NoReturn:
    raise NotImplementedError


class Callback(Protocol):
    def __call__(self, intermediate_result: scipy.optimize.OptimizeResult) -> None: ...


class MinimizeAlgorithm(abc.ABC):
    def minimize(
        self,
        x0: Float[jax.Array, " N"],
        fun: Callable | None = None,
        *,
        jac: Callable | None = None,
        hess: Callable | None = None,
        hessp: Callable | None = None,
        callback: Callback | None = None,
    ) -> scipy.optimize.OptimizeResult:
        fun = grapes.timer()(fun) if fun else None
        jac = grapes.timer()(jac) if jac else None
        hess = grapes.timer()(hess) if hess else None
        hessp = grapes.timer()(hessp) if hessp else None

        @grapes.timer()
        def callback_wrapped(
            intermediate_result: scipy.optimize.OptimizeResult,
        ) -> None:
            if callback:
                callback(intermediate_result)

        with grapes.timer() as timer:
            result: scipy.optimize.OptimizeResult = self._minimize(
                fun=fun,
                x0=x0,
                jac=jac,
                hess=hess,
                hessp=hessp,
                callback=callback_wrapped,
            )
        for key, value in timer.row(-1).items():
            result[f"time_{key}"] = value
        result["n_iter"] = callback_wrapped.count
        if fun:
            result["n_fun"] = fun.count
        if jac:
            result["n_jac"] = jac.count
        if hess:
            result["n_hess"] = hess.count
        if hessp:
            result["n_hessp"] = hessp.count
        return result

    @abc.abstractmethod
    def _minimize(
        self,
        x0: Float[jax.Array, " N"],
        fun: Callable | None = None,
        jac: Callable | None = None,
        hess: Callable | None = None,
        hessp: Callable | None = None,
        *,
        callback: Callable,
    ) -> scipy.optimize.OptimizeResult: ...
