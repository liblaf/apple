import abc
from collections.abc import Callable, Sequence
from typing import Protocol

import jax
from jaxtyping import Float

import liblaf.apple as apple  # noqa: PLR0402
import liblaf.grapes as grapes  # noqa: PLR0402

from . import MinimizeResult


class Callback(Protocol):
    def __call__(self, intermediate_result: MinimizeResult) -> None: ...


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
    ) -> MinimizeResult:
        fun = (
            grapes.timer(label="fun()")(apple.utils.block_until_ready()(fun))
            if fun is not None
            else None
        )
        jac = (
            grapes.timer(label="jac()")(apple.utils.block_until_ready()(jac))
            if jac is not None
            else None
        )
        hess = (
            grapes.timer(label="hess()", record_log_level="TRACE")(
                apple.utils.block_until_ready()(hess)
            )
            if hess is not None
            else None
        )
        hessp = (
            grapes.timer(label="hessp()")(apple.utils.block_until_ready()(hessp))
            if hessp is not None
            else None
        )

        @grapes.timer(label="callback()")
        def callback_wrapped(
            intermediate_result: MinimizeResult,
        ) -> None:
            if callback is not None:
                callback(intermediate_result)

        with grapes.timer(label="minimize") as timer:
            result: MinimizeResult = self._minimize(
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
    ) -> MinimizeResult: ...
