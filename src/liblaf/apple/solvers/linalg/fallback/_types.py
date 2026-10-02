import attrs
import torch
from jaxtyping import Float, Integer
from torch import Tensor

from liblaf.apple.solvers.linalg.base import Result, Solution, State, Stats

type Scalar = Float[Tensor, ""]
type Vector = Float[Tensor, " N"]


@attrs.define
class FallbackState(State):
    """State collected while trying multiple linear solvers."""

    init_params: Vector
    solutions: list[Solution] = attrs.field(factory=list)
    absolute_residuals: Float[Tensor, " S"] = attrs.field(factory=list)
    relative_residuals: Float[Tensor, " S"] = attrs.field(factory=list)
    best_index: Integer[Tensor, ""] = attrs.field(
        factory=lambda: torch.as_tensor(0, dtype=torch.int32)
    )

    @property
    def best_solution(self) -> Solution:
        return self.solutions[self.best_index]

    @property
    def params(self) -> Vector:
        return self.best_solution.params

    @property
    def result(self) -> Result:
        return self.best_solution.result


@attrs.define
class FallbackStats(Stats):
    """Stats placeholder for fallback linear solves."""
