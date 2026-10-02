from typing import cast

import attrs
from jaxtyping import Float
from torch import Tensor

from ._problem import BaseProblem
from ._types import Result, Solution, State, Stats

type Scalar = Float[Tensor, ""]
type Vector = Float[Tensor, " N"]


@attrs.define
class LinearSolver[S: State, T: Stats]:
    """Base class for linear solvers."""

    from ._types import Solution, State, Stats

    def init(self, problem: BaseProblem, params: Vector) -> S:
        """Create solver state from an initial parameter vector."""
        raise NotImplementedError

    def compute(self, problem: BaseProblem, state: S) -> Result:
        """Run one complete solve from an initialized state."""
        raise NotImplementedError

    def postprocess(
        self, problem: BaseProblem, state: S, result: Result
    ) -> Solution[S, T]:
        """Wrap final state and result metadata in a solution object."""
        del problem
        stats: T = cast("T", self.Stats())  # ty:ignore[call-non-callable]
        return Solution(result=result, state=state, stats=stats)

    def solve(self, problem: BaseProblem, params: Vector) -> Solution[S, T]:
        """Initialize, compute, and postprocess a linear solve."""
        state: S = self.init(problem, params)
        result: Result = self.compute(problem, state)
        return self.postprocess(problem, state, result)
