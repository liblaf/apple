import functools
from collections.abc import Callable, Sequence
from typing import Any, overload, override

import attrs
import equinox as eqx
import jax.numpy as jnp
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
            jax_op(problem.fun),
            x0,
            args=args,
            method=self.method,
            jac=jax_op(problem.jac),
            hess=jax_op(problem.hess),
            hessp=jax_op(problem.hessp),
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


@overload
def jax_op[**P, T](func: Callable[P, T], /) -> Callable[P, T]: ...
@overload
def jax_op(func: None, /) -> None: ...
def jax_op[**P, T](func: Callable[P, T] | None, /) -> Callable[P, T] | None:
    if func is None:
        return None

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        args: tuple = tuple(
            jnp.asarray(arg, dtype=float) if eqx.is_array_like(arg) else arg
            for arg in args
        )
        return func(*args, **kwargs)

    return wrapper
