import abc
from typing import Any, cast, override

import attrs
import cupy as cp
import torch
from cupyx.scipy.sparse import linalg
from jaxtyping import Float
from torch import Tensor

from liblaf.apple.solvers.linalg.base import BaseProblem, LinearSolver, Problem, Result
from liblaf.apple.solvers.utils import is_implemented

from ._types import CupyState, CupyStats

type Vector = Float[Tensor, " N"]
type VectorCupy = Float[cp.ndarray, " N"]


@attrs.define(kw_only=True)
class CupySolver(LinearSolver[CupyState, CupyStats]):
    """Base class for linear solvers backed by CuPy sparse routines."""

    from ._types import CupyState as State
    from ._types import CupyStats as Stats

    type Solution = LinearSolver.Solution[State, Stats]

    maxiter: int | None = None

    @override
    def init(self, problem: BaseProblem, params: Vector) -> State:
        return self.State(params=params)

    @override
    def compute(self, problem: BaseProblem, state: State) -> Result:
        problem: Problem = cast("Problem", problem)
        lop: linalg.LinearOperator = _as_lop(problem)
        options: dict[str, Any] = self._options(problem)
        with cp.cuda.Stream.from_external(torch.cuda.current_stream()):
            x, info = self._wrapped(
                lop, cp.asarray(problem.b), cp.asarray(state.params), **options
            )
        state.params = torch.as_tensor(x)
        result: Result = self._result(state, info)
        return result

    @override
    def postprocess(
        self, problem: BaseProblem, state: State, result: Result
    ) -> Solution:
        problem: Problem = cast("Problem", problem)
        stats: CupySolver.Stats = self.Stats()
        if state.info >= 0:
            stats.absolute_residual = torch.linalg.vector_norm(
                problem.matvec(state.params) - problem.b
            )
            stats.relative_residual = (
                stats.absolute_residual / torch.linalg.vector_norm(problem.b).item()
            )
        return LinearSolver.Solution(result=result, state=state, stats=stats)

    def _options(self, problem: Problem) -> dict[str, Any]:
        return {"maxiter": self.maxiter, "M": _preconditioner(problem)}

    def _result(self, state: State, info: int) -> Result:
        state.info = info
        if info == 0:
            return Result.SUCCESS
        if info < 0:
            return Result.BREAKDOWN
        state.step = info
        return Result.MAX_STEPS_REACHED

    @abc.abstractmethod
    def _wrapped(self, *args, **kwargs) -> tuple[VectorCupy, int]:
        raise NotImplementedError


def _as_lop(problem: Problem) -> linalg.LinearOperator:
    def matvec(x: VectorCupy) -> VectorCupy:
        x_torch: Vector = torch.as_tensor(x)
        y_torch: Vector = problem.matvec(x_torch)
        return cp.asarray(y_torch, copy=True)

    def rmatvec(x: VectorCupy) -> VectorCupy:
        x_torch: Vector = torch.as_tensor(x)
        y_torch: Vector = problem.rmatvec(x_torch)
        return cp.asarray(y_torch, copy=True)

    dim: int
    (dim,) = problem.b.shape
    return linalg.LinearOperator(
        shape=(dim, dim),
        matvec=matvec,  # ty:ignore[unknown-argument]
        rmatvec=rmatvec if is_implemented(problem, Problem.rmatvec) else None,  # ty:ignore[unknown-argument]
        dtype=cp.asarray(problem.b).dtype,
    )


def _preconditioner(problem: Problem) -> linalg.LinearOperator | None:
    if not is_implemented(problem, Problem.precondition):
        return None

    def matvec(x: VectorCupy) -> VectorCupy:
        x_torch: Vector = torch.as_tensor(x)
        y_torch: Vector = problem.precondition(x_torch)
        return cp.asarray(y_torch, copy=True)

    def rmatvec(x: VectorCupy) -> VectorCupy:
        x_torch: Vector = torch.as_tensor(x)
        y_torch: Vector = problem.rprecondition(x_torch)
        return cp.asarray(y_torch, copy=True)

    dim: int
    (dim,) = problem.b.shape
    return linalg.LinearOperator(
        shape=(dim, dim),
        matvec=matvec,  # ty:ignore[unknown-argument]
        rmatvec=rmatvec if is_implemented(problem, Problem.rprecondition) else None,  # ty:ignore[unknown-argument]
        dtype=cp.asarray(problem.b).dtype,
    )
