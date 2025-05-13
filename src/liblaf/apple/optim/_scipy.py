from collections.abc import Callable
from typing import Any, override

import attrs
import jax
import scipy.optimize
from jaxtyping import Float
from numpy.typing import ArrayLike

from ._abc import Callback, Optimizer, OptimizeResult


@attrs.define(kw_only=True)
class OptimizerScipy(Optimizer):
    method: str = "trust-constr"
    tol: float | None = 5e-5
    options: dict[str, Any] = attrs.field(factory=lambda: {"disp": True})

    @override
    def _minimize(
        self,
        fun: Callable[..., Float[jax.Array, ""]],
        x0: ArrayLike,
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
        scipy_result: scipy.optimize.OptimizeResult = scipy.optimize.minimize(
            fun,
            x0,
            args=args,
            method=self.method,
            jac=jac,
            hess=hess,
            hessp=hessp,
            tol=self.tol,
            options=self.options,
            callback=callback,
            **kwargs,
        )
        ic(scipy_result)
        result: OptimizeResult = OptimizeResult(**scipy_result)
        result = replace_result(result, "nit", "n_iter")
        result = replace_result(result, "nfev", "n_fun")
        result = replace_result(result, "njev", "n_jac")
        return result


def replace_result(result: OptimizeResult, src: str, dst: str) -> OptimizeResult:
    if src in result:
        result[dst] = result[src]
        del result[src]
    return result
