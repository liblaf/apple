from collections.abc import Callable, Mapping, Sequence
from typing import Any

import attrs
import jax
import scipy
import scipy.optimize
from jaxtyping import Float

from . import MinimizeAlgorithm


@attrs.frozen
class MinimizeScipy(MinimizeAlgorithm):
    method: str | None = None
    options: Mapping[str, Any] = {"disp": True}
    bounds: Sequence | None = None

    def _minimize(
        self,
        x0: Float[jax.Array, " N"],
        fun: Callable | None = None,
        jac: Callable | None = None,
        hess: Callable | None = None,
        hessp: Callable | None = None,
        *,
        callback: Callable | None = None,
    ) -> scipy.optimize.OptimizeResult:
        return scipy.optimize.minimize(
            fun=fun,
            x0=x0,
            method=self.method,
            jac=jac,
            hess=hess,
            hessp=hessp,
            bounds=self.bounds,
            options=self.options,
            callback=callback,
        )
