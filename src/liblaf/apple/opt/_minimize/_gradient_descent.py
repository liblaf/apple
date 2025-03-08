from collections.abc import Callable, Sequence

import attrs
import jax
from jaxtyping import Float

from . import MinimizeAlgorithm, MinimizeResult


@attrs.define(kw_only=True, on_setattr=attrs.setters.convert)
class MinimizeGradientDescent(MinimizeAlgorithm):
    lr: float = 1e-3
    max_iter: int = 100
    tol: float = 1e-6

    def _minimize(
        self,
        x0: Float[jax.Array, " N"],
        fun: Callable | None = None,
        jac: Callable | None = None,
        hess: Callable | None = None,
        hessp: Callable | None = None,
        *,
        bounds: Sequence | None = None,
        callback: Callable,
    ) -> MinimizeResult:
        assert jac is not None
        result = MinimizeResult()
        x: Float[jax.Array, " N"] = x0
        for _ in range(self.max_iter):
            grad: Float[jax.Array, " N"] = jac(x)
            x_new: Float[jax.Array, " N"] = x - self.lr * grad
            if jax.numpy.linalg.norm(x_new - x) < self.tol:
                x = x_new
                result["x"] = x
                break
            x = x_new
            result["x"] = x
            callback(result)
        return result
