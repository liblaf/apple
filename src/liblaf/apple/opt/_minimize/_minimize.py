from collections.abc import Callable, Mapping, Sequence

import jax
from jaxtyping import Float

from . import MinimizeAlgorithm, MinimizeResult, MinimizeScipy


def minimize(
    fun: Callable,
    x0: Float[jax.Array, " N"],
    *,
    algo: MinimizeAlgorithm | None = None,
    jac: Callable | None = None,
    hess: Callable | None = None,
    hessp: Callable | None = None,
    hess_diag: Callable | None = None,
    hess_quad: Callable | None = None,
    args: Sequence | None = None,
    kwargs: Mapping | None = None,
    bounds: Sequence | None = None,
    callback: Callable | None = None,
) -> MinimizeResult:
    if algo is None:
        algo = MinimizeScipy(
            method="trust-constr", options={"disp": True, "verbose": 3}
        )
        # algo = MinimizePNCG()
    return algo.minimize(
        fun=fun,
        x0=x0,
        args=args,
        kwargs=kwargs,
        jac=jac,
        hess=hess,
        hessp=hessp,
        hess_diag=hess_diag,
        hess_quad=hess_quad,
        bounds=bounds,
        callback=callback,
    )
