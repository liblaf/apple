from collections.abc import Callable, Mapping, Sequence
from typing import Any, override

import attrs
import scipy.optimize
from jaxtyping import Array, Float, PyTree

from liblaf.apple.jax import tree

from ._minimizer import Minimizer, Solution
from ._objective import Objective


@tree.pytree
class MinimizerScipy(Minimizer):
    method: str = "trust-constr"
    tol: float | None = 1e-5
    options: Mapping[str, Any] = {}

    @override
    def _minimize_impl(
        self,
        objective: Objective,
        x0: PyTree,
        args: Sequence[Any] = (),
        kwargs: Mapping[str, Any] = {},
        bounds: Any = None,
        callback: Callable | None = None,
    ) -> Solution:
        x0_flat: Float[Array, " N"]
        unflatten: Callable[[Array], PyTree]
        x0_flat, unflatten = tree.flatten(x0)
        objective = objective.flatten(unflatten).partial(kwargs=kwargs)
        fun: Callable | None
        jac: Callable | bool | None
        if objective.fun_and_jac is not None:
            fun = objective.fun_and_jac
            jac = True
        else:
            fun = objective.fun
            jac = objective.jac
        callback = _CallbackWrapper(callback=callback, unflatten=unflatten)
        result: scipy.optimize.OptimizeResult = scipy.optimize.minimize(
            fun=fun,
            x0=x0_flat,
            args=args,
            method=self.method,
            jac=jac,
            hess=objective.hess,
            hessp=objective.hessp,
            bounds=bounds,
            tol=self.tol,
            callback=callback,
            options=self.options,
        )
        result["x"] = unflatten(result["x"])
        return Solution(result)


@attrs.define
class _CallbackWrapper:
    callback: Callable | None
    unflatten: Callable[[Array], PyTree]
    n_iter: int = 0

    def __call__(self, intermediate_result: scipy.optimize.OptimizeResult) -> Any:
        if self.callback is None:
            return None
        self.n_iter += 1
        intermediate_result = Solution(intermediate_result)
        intermediate_result["n_iter"] = self.n_iter
        intermediate_result["x"] = self.unflatten(intermediate_result["x"])
        return self.callback(intermediate_result)
