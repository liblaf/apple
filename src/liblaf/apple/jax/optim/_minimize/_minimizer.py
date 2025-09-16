import abc
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import attrs
import scipy.optimize
from jaxtyping import PyTree

from liblaf import grapes
from liblaf.apple.jax import tree

from ._objective import Objective


class Solution(scipy.optimize.OptimizeResult): ...


@tree.pytree
class Minimizer(abc.ABC):
    jit: bool = tree.field(default=True)
    timer: bool = tree.field(default=True)

    def minimize(
        self,
        x0: PyTree,
        *,
        fun: Callable | None = None,
        jac: Callable | None = None,
        hess: Callable | None = None,
        hessp: Callable | None = None,
        hess_diag: Callable | None = None,
        hess_quad: Callable | None = None,
        fun_and_jac: Callable | None = None,
        jac_and_hess_diag: Callable | None = None,
        args: Sequence[Any] = (),
        kwargs: Mapping[str, Any] = {},
        bounds: Any = None,
        callback: Callable | None = None,
    ) -> Solution:
        objective: Objective = Objective(
            fun=fun,
            jac=jac,
            hess=hess,
            hessp=hessp,
            hess_diag=hess_diag,
            hess_quad=hess_quad,
            fun_and_jac=fun_and_jac,
            jac_and_hess_diag=jac_and_hess_diag,
        )
        if self.jit:
            objective = objective.jit()
        if self.timer:
            objective = objective.timer()
        with grapes.timer(name="minimize") as timer:
            solution: Solution = self._minimize_impl(
                objective=objective,
                x0=x0,
                args=args,
                kwargs=kwargs,
                bounds=bounds,
                callback=callback,
            )
        solution["time"] = timer.elapsed()
        for field in attrs.fields(type(objective)):
            fn: Callable | None = getattr(objective, field.name)
            if not callable(fn):
                continue
            timer: grapes.BaseTimer | None = grapes.unbind_getattr(
                fn, "_self_timer", None
            )
            if timer is None:
                continue
            timer.log_summary()
        return solution

    @abc.abstractmethod
    def _minimize_impl(
        self,
        objective: Objective,
        x0: PyTree,
        args: Sequence[Any] = (),
        kwargs: Mapping[str, Any] = {},
        bounds: Any = None,
        callback: Callable | None = None,
    ) -> Solution: ...
