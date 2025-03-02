import abc
from collections.abc import Callable, Sequence
from typing import Protocol

import jax
import scipy.optimize
from jaxtyping import Float

from liblaf import grapes


class Callback(Protocol):
    def __call__(self, intermediate_result: scipy.optimize.OptimizeResult) -> None: ...


class MinimizeAlgorithm(abc.ABC):
    def minimize(
        self,
        x0: Float[jax.Array, " N"],
        fun: Callable | None = None,
        jac: Callable | None = None,
        hess: Callable | None = None,
        hessp: Callable | None = None,
        *,
        bounds: Sequence | None = None,
        callback: Callback | None = None,
    ) -> scipy.optimize.OptimizeResult:
        fun = grapes.timer(label="fun()")(fun) if fun is not None else None
        jac = grapes.timer(label="jac()")(jac) if jac is not None else None
        hess = grapes.timer(label="hess()")(hess) if hess is not None else None
        hessp = grapes.timer(label="hessp()")(hessp) if hessp is not None else None

        @grapes.timer(label="callback()")
        def callback_wrapped(
            intermediate_result: scipy.optimize.OptimizeResult,
        ) -> None:
            if callback:
                callback(intermediate_result)

        with grapes.timer(label="minimize") as timer:
            result: scipy.optimize.OptimizeResult = self._minimize(
                x0=x0,
                fun=fun,
                jac=jac,
                hess=hess,
                hessp=hessp,
                bounds=bounds,
                callback=callback_wrapped,
            )
        for key, value in timer.row(-1).items():
            result[f"time_{key}"] = value
        result["n_iter"] = callback_wrapped.count
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
        bounds: Sequence | None = None,
        callback: Callable,
    ) -> scipy.optimize.OptimizeResult: ...
