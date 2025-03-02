from collections.abc import Callable, Sequence

import jax
import scipy.optimize
from jaxtyping import Float

from . import MinimizeAlgorithm, MinimizeScipy


def minimize(
    x0: Float[jax.Array, " N"],
    fun: Callable | None = None,
    jac: Callable | None = None,
    hess: Callable | None = None,
    hessp: Callable | None = None,
    *,
    algo: MinimizeAlgorithm | None = None,
    bounds: Sequence | None = None,
    callback: Callable | None = None,
) -> scipy.optimize.OptimizeResult:
    if algo is None:
        algo = MinimizeScipy(
            method="trust-constr", options={"disp": True, "verbose": 3}
        )
    return algo.minimize(
        fun=fun,
        x0=x0,
        jac=jac,
        hess=hess,
        hessp=hessp,
        bounds=bounds,
        callback=callback,
    )
