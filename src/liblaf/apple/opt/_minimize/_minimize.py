from collections.abc import Callable, Sequence

import jax
from jaxtyping import Float

from . import MinimizeAlgorithm, MinimizeResult, MinimizeScipy


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
) -> MinimizeResult:
    if algo is None:
        algo = MinimizeScipy(
            method="trust-constr", options={"disp": True, "verbose": 3}
        )
        # algo = MinimizePNCG()
    return algo.minimize(
        fun=fun,
        x0=x0,
        jac=jac,
        hess=hess,
        hessp=hessp,
        bounds=bounds,
        callback=callback,
    )
