import abc
from collections.abc import Callable, Sequence

import scipy.optimize
from jaxtyping import ArrayLike, Float

from liblaf import grapes
from liblaf.apple.struct import tree

from .problem import OptimizationProblem


class OptimizeResult(scipy.optimize.OptimizeResult): ...


@tree.pytree
class Optimizer(tree.PyTreeMixin, abc.ABC):
    autodiff: bool = tree.static(default=False, kw_only=True)
    jit: bool = tree.static(default=False, kw_only=True)

    @property
    def name(self) -> str:
        return type(self).__qualname__

    def minimize(
        self,
        fun: Callable | None,
        x0: Float[ArrayLike, " N"],
        args: Sequence = (),
        *,
        jac: Callable | None = None,
        hess: Callable | None = None,
        hessp: Callable | None = None,
        hess_diag: Callable | None = None,
        hess_quad: Callable | None = None,
        fun_and_jac: Callable | None = None,
        jac_and_hess_diag: Callable | None = None,
        callback: Callable | None = None,
        **kwargs,
    ) -> OptimizeResult:
        problem: OptimizationProblem = OptimizationProblem(
            fun=fun,
            jac=jac,
            hess=hess,
            hessp=hessp,
            hess_diag=hess_diag,
            hess_quad=hess_quad,
            fun_and_jac=fun_and_jac,
            jac_and_hess_diag=jac_and_hess_diag,
            callback=callback,
        )
        problem = problem.implement()
        if self.autodiff:
            problem = problem.autodiff().implement()
        if self.jit:
            problem = problem.jit()
        problem = problem.timer()
        with grapes.timer(label=self.name) as timer:
            result: OptimizeResult = self._minimize_impl(problem, x0, args, **kwargs)
        result["time"] = timer.elapsed()
        result = problem.update_result(result)
        return result

    @abc.abstractmethod
    def _minimize_impl(
        self,
        problem: OptimizationProblem,
        x0: Float[ArrayLike, " N"],
        args: Sequence,
        **kwargs,
    ) -> OptimizeResult:
        raise NotImplementedError
