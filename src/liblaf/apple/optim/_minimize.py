from collections.abc import Callable

from jaxtyping import ArrayLike, Float

from ._scipy import OptimizerScipy
from .optimizer import Optimizer, OptimizeResult


def minimize(
    fun: Callable | None,
    x0: Float[ArrayLike, " N"],
    *,
    args: tuple = (),
    method: Optimizer | None = None,
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
    if method is None:
        method = OptimizerScipy()
    return method.minimize(
        fun,
        x0,
        args=args,
        jac=jac,
        hess=hess,
        hessp=hessp,
        hess_diag=hess_diag,
        hess_quad=hess_quad,
        fun_and_jac=fun_and_jac,
        jac_and_hess_diag=jac_and_hess_diag,
        callback=callback,
        **kwargs,
    )
