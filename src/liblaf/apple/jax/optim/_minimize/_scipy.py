from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, override

import attrs
import scipy.optimize
from jaxtyping import Array, Float, PyTree

from liblaf import grapes
from liblaf.apple.jax import tree

from ._minimizer import Minimizer, Solution
from ._objective import Objective


@tree.pytree
class MinimizerScipy(Minimizer):
    method: str = tree.field(default="trust-constr")
    tol: float | None = tree.field(default=None)
    options: Mapping[str, Any] = tree.field(factory=lambda: {"verbose": 3})

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
class _ProblemWrapper:
    args: Iterable[Any]
    kwargs: Mapping[str, Any]
    unflatten: Callable[[Array], PyTree]

    def wraps(self, fn: Callable, unflatten_args: Sequence[int] = (0,)) -> Callable:
        @grapes.decorator
        def wrapper(
            wrapped: Callable, _instance: None, args: tuple, kwargs: dict
        ) -> Array:
            args: list = list(args)
            for i in unflatten_args:
                args[i] = self.unflatten(args[i])
            result: PyTree = wrapped(*args, *self.args, **kwargs, **self.kwargs)
            result_flat: Float[Array, " ..."]
            result_flat, _ = tree.flatten(result)
            return result_flat

        return wrapper(fn)
