from collections.abc import Sequence
from typing import Any, override

import attrs
import scipy.optimize
from jaxtyping import Float
from numpy.typing import ArrayLike

from liblaf.apple.struct import tree

from .optimizer import Optimizer, OptimizeResult
from .problem import OptimizationProblem


@tree.pytree
class OptimizerScipy(Optimizer):
    method: str = "trust-constr"
    tol: float | None = None
    options: dict[str, Any] = attrs.field(factory=lambda: {"disp": True})

    @override
    def _minimize_impl(
        self,
        problem: OptimizationProblem,
        x0: Float[ArrayLike, " N"],
        args: Sequence,
        **kwargs,
    ) -> OptimizeResult:
        scipy_result: scipy.optimize.OptimizeResult = scipy.optimize.minimize(
            problem.fun,
            x0,
            args=args,
            method=self.method,
            jac=problem.jac,
            hess=problem.hess,
            hessp=problem.hessp,
            tol=self.tol,
            options=self.options,
            callback=problem.callback,
            **kwargs,
        )
        result: OptimizeResult = OptimizeResult(**scipy_result)
        result = replace_result(result, "nfev", "n_fun")
        result = replace_result(result, "nit", "n_iter")
        result = replace_result(result, "niter", "n_iter")
        result = replace_result(result, "njev", "n_jac")
        return result


def replace_result(result: OptimizeResult, src: str, dst: str) -> OptimizeResult:
    if src in result:
        result[dst] = result[src]
        del result[src]
    return result
