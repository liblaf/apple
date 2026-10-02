from typing import Any, override

import attrs
import cupy as cp
from cupyx.scipy.sparse import linalg
from jaxtyping import Float

from liblaf.apple.solvers.linalg.base import Problem

from ._base import CupySolver

type VectorCupy = Float[cp.ndarray, " N"]


@attrs.define(kw_only=True)
class CupyCG(CupySolver):
    rtol: float = 1e-5
    atol: float = 0.0

    @override
    def _options(self, problem: Problem) -> dict[str, Any]:
        options: dict[str, Any] = super()._options(problem)
        options.update({"atol": self.atol, "rtol": self.rtol})
        return options

    @override
    def _wrapped(self, *args, **kwargs) -> tuple[VectorCupy, int]:
        return linalg.cg(*args, **kwargs)
