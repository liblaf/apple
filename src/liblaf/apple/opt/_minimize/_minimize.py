from collections.abc import Callable
from typing import Any

import jax
import scipy.optimize
from jaxtyping import Float, Scalar


def minimize(
    fun: Callable[[Float[jax.Array, " N"]], Scalar],
    x0: Float[jax.Array, " N"],
    args: tuple[Any, ...] = (),
    method: str | None = "Newton-CG",
    jac: Callable[[Float[jax.Array, " N"]], Float[jax.Array, " N"]] | None = None,
    hess: Callable[[Float[jax.Array, " N"]], Float[jax.Array, "N N"]] | None = None,
    hessp: Callable[
        [Float[jax.Array, " N"], Float[jax.Array, " N"]], Float[jax.Array, " N"]
    ]
    | None = None,
    options: dict | None = None,
    callback: Callable | None = None,
) -> scipy.optimize.OptimizeResult:
    return scipy.optimize.minimize(
        fun=fun,
        x0=x0,
        args=args,
        method=method,
        jac=jac,
        hess=hess,
        hessp=hessp,
        options=options,
        callback=callback,
    )
