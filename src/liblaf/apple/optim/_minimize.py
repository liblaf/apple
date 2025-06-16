from collections.abc import Callable

import jax
from jaxtyping import Float
from numpy.typing import ArrayLike

from ._abc import Callback, Optimizer, OptimizeResult
from ._scipy import OptimizerScipy


def minimize(
    fun: Callable[..., Float[jax.Array, ""]],
    x0: Float[ArrayLike, " N"],
    *,
    args: tuple = (),
    method: Optimizer | None = None,
    jac: Callable | None = None,
    hess: Callable | None = None,
    hessp: Callable | None = None,
    hess_diag: Callable | None = None,
    hess_quad: Callable | None = None,
    jac_and_hess_diag: Callable | None = None,
    prepare: Callable | None = None,
    callback: Callback | None = None,
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
        jac_and_hess_diag=jac_and_hess_diag,
        prepare=prepare,
        callback=callback,
        **kwargs,
    )
