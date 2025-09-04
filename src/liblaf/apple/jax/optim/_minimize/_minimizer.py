import abc
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import scipy.optimize
from jaxtyping import PyTree

from liblaf.apple.jax import tree

from ._objective import Objective


class Solution(scipy.optimize.OptimizeResult): ...


@tree.pytree
class Minimizer(abc.ABC):
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
        callback: Callable | None = None,
    ) -> Solution:
        objective = Objective(
            fun=fun,
            jac=jac,
            hess=hess,
            hessp=hessp,
            hess_diag=hess_diag,
            hess_quad=hess_quad,
            fun_and_jac=fun_and_jac,
            jac_and_hess_diag=jac_and_hess_diag,
        )
        objective.jit()
        objective.timer()
        return self._minimize_impl(
            objective=objective, x0=x0, args=args, kwargs=kwargs, callback=callback
        )

    @abc.abstractmethod
    def _minimize_impl(
        self,
        objective: Objective,
        x0: PyTree,
        args: Sequence[Any] = (),
        kwargs: Mapping[str, Any] = {},
        callback: Callable | None = None,
    ) -> Solution: ...
