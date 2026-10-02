from typing import cast, override

import attrs
import torch
from jaxtyping import Float
from torch import Tensor

from liblaf.apple.solvers.linalg.base import (
    BaseProblem,
    LinearSolver,
    Problem,
    Result,
    Solution,
)

from ._types import FallbackState, FallbackStats

type Scalar = Float[Tensor, ""]
type Vector = Float[Tensor, " N"]


@attrs.define
class FallbackSolver(LinearSolver[FallbackState, FallbackStats]):
    from ._types import FallbackState as State
    from ._types import FallbackStats as Stats

    type Solution = LinearSolver.Solution[State, Stats]

    @staticmethod
    def _default_solvers() -> list[LinearSolver]:
        from liblaf.apple.solvers.linalg.cupy import CupyCG, CupyMinRes

        return [CupyCG(), CupyMinRes()]

    solvers: list[LinearSolver] = attrs.field(factory=_default_solvers)

    @override
    def init(self, problem: BaseProblem, params: Vector) -> State:
        return self.State(init_params=params)

    @override
    def compute(self, problem: BaseProblem, state: State) -> Result:
        problem: Problem = cast("Problem", problem)
        results: list[Result] = []
        absolute_residuals: list[Scalar] = []
        relative_residuals: list[Scalar] = []
        for solver in self.solvers:
            solution: Solution = solver.solve(problem, state.init_params)
            state.solutions.append(solution)
            results.append(solution.result)
            absolute_residual: Scalar = torch.linalg.vector_norm(
                problem.matvec(solution.state.params) - problem.b
            )
            relative_residual: Scalar = absolute_residual / torch.linalg.vector_norm(
                problem.b
            )
            absolute_residuals.append(absolute_residual)
            relative_residuals.append(relative_residual)
            if solution.success:
                break
        state.absolute_residuals = torch.as_tensor(absolute_residuals)
        state.relative_residuals = torch.as_tensor(relative_residuals)
        state.best_index = torch.argmin(state.absolute_residuals)
        return state.result
