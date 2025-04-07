import functools
from collections.abc import Callable, Mapping, Sequence
from typing import Any, override

import attrs
import jax
import jax.numpy as jnp
import numpy as np
import scipy
import scipy.linalg
import scipy.optimize
import scipy.sparse
import scipy.sparse.linalg
from jaxtyping import Float
from loguru import logger

from . import MinimizeAlgorithm, MinimizeResult


@attrs.frozen
class MinimizeScipy(MinimizeAlgorithm):
    method: str | None = None
    tol: float | None = None
    options: Mapping[str, Any] = {"disp": True}

    @override
    def _minimize(
        self,
        fun: Callable,
        x0: Float[jax.Array, " N"],
        *,
        args: Sequence | None = None,
        kwargs: Mapping | None = None,
        jac: Callable | None = None,
        hess: Callable | None = None,
        hessp: Callable | None = None,
        hess_diag: Callable | None = None,
        hess_quad: Callable | None = None,
        jac_and_hess_diag: Callable | None = None,
        bounds: Sequence | None = None,
        callback: Callable,
    ) -> MinimizeResult:
        if kwargs is not None:
            logger.warning("`MinimizeScipy` does not support `kwargs`.")
        if hess_diag is not None:
            logger.warning("`MinimizeScipy` does not support `hess_diag`.")
        if hess_quad is not None:
            logger.warning("`MinimizeScipy` does not support `hess_quad`.")
        if (
            self.method
            and (self.method.lower() in ["newton-cg", "trust-constr"])
            and (hess is not None)
        ):
            hess_raw = hess

            @functools.wraps(hess_raw)
            def hess(
                x: Float[jax.Array, " N"], *args, **kwargs
            ) -> scipy.sparse.linalg.LinearOperator:
                x = jnp.asarray(x, dtype=float)
                H = hess_raw(x, *args, **kwargs)
                if isinstance(H, jax.Array):
                    H = np.asarray(H)
                return scipy.sparse.linalg.aslinearoperator(H)

        scipy_result: scipy.optimize.OptimizeResult = scipy.optimize.minimize(
            fun=fun,
            x0=x0,
            args=args,
            method=self.method,
            jac=jac,
            hess=hess,
            hessp=hessp,
            bounds=bounds,
            tol=self.tol,
            options=self.options,
            callback=callback,
        )
        scipy_result["x"] = jnp.asarray(scipy_result["x"])
        return MinimizeResult(scipy_result)
