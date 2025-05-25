from collections.abc import Callable
from typing import override

import jax
from jaxtyping import Float
from numpy.typing import ArrayLike

from liblaf.apple.optim._abc import Callback, OptimizeResult

from ._abc import Optimizer


class PNCG(Optimizer):
    @override
    def _minimize(
        self,
        fun: Callable[..., Float[jax.Array, ""]],
        x0: Float[ArrayLike, " N"],
        *,
        args: tuple = (),
        jac: Callable | None = None,
        hess: Callable | None = None,
        hessp: Callable | None = None,
        hess_diag: Callable | None = None,
        hess_quad: Callable | None = None,
        jac_and_hess_diag: Callable | None = None,
        callback: Callback | None = None,
        **kwargs,
    ) -> OptimizeResult:
        raise NotImplementedError
        result = OptimizeResult(success=False)
        return result
