from typing import Any, override

import attrs
import cupy as cp
from cupyx.scipy.sparse import linalg
from jaxtyping import Float

from liblaf.apple.solvers.linalg.base import Problem

from ._base import CupySolver

type VectorCupy = Float[cp.ndarray, " N"]


@attrs.define(kw_only=True)
class CupyMinRes(CupySolver):
    shift: float = 0.0
    tol: float = 1e-5

    @override
    def _options(self, problem: Problem) -> dict[str, Any]:
        options: dict[str, Any] = super()._options(problem)
        options.update({"shift": self.shift, "tol": self.tol})
        return options

    @override
    def _wrapped(self, *args, **kwargs) -> tuple[VectorCupy, int]:
        return linalg.minres(*args, **kwargs)
