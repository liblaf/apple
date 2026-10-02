from typing import cast

from jaxtyping import Float
from torch import Tensor

from liblaf.apple.solvers.utils import is_implemented

from ._protocols import BaseProblem, Problem
from ._types import Solution, State, Stats

type Vector = Float[Tensor, " N"]


class Optimizer[S: State, T: Stats]:
    """Base class for iterative optimizers.

    Subclasses keep optimizer-specific data in a mutable state object. A
    [`step`][liblaf.apple.solvers.optim.base.Optimizer.step] implementation mutates the
    model state and optimizer state in place, so callbacks and
    [`postprocess`][liblaf.apple.solvers.optim.base.Optimizer.postprocess] observe the
    same final objects.
    """

    from ._types import Result, Solution, State, Stats

    def init[X](self, problem: BaseProblem[X], model_state: X, params: Vector) -> S:
        """Create optimizer state from a model state and parameter vector."""
        raise NotImplementedError

    def step[X](self, problem: BaseProblem[X], model_state: X, opt_state: S) -> None:
        """Advance the optimizer by one step.

        Implementations should mutate `model_state` and `opt_state` with any
        accepted parameter, gradient, or diagnostic updates.
        """
        raise NotImplementedError

    def terminate[X](
        self, problem: BaseProblem[X], model_state: X, opt_state: S
    ) -> tuple[bool, Result]:
        """Return whether optimization should stop and why."""
        raise NotImplementedError

    def postprocess[X](
        self, problem: BaseProblem[X], model_state: X, opt_state: S, result: Result
    ) -> Solution[S, T]:
        """Build the final solution object."""
        del problem, model_state
        stats: T = cast("T", {})
        return Optimizer.Solution(result=result, state=opt_state, stats=stats)

    def minimize[X](
        self, problem: BaseProblem[X], model_state: X, params: Vector
    ) -> Solution[S, T]:
        """Run optimization until the configured termination rule stops.

        After every step, [`Problem.callback`][liblaf.apple.solvers.optim.base.Problem.callback]
        is called when the concrete problem implements it. The callback receives
        the current model state and the same mutable optimizer state that will be
        stored on the returned [`Solution`][liblaf.apple.solvers.optim.base.Solution].

        Args:
            problem: Optimization problem that supplies objective hooks.
            model_state: Initial model state used by objective hooks.
            params: Initial optimizer parameter vector.

        Returns:
            The final solution.
        """
        problem: Problem[X] = cast("Problem[X]", problem)
        opt_state: S = self.init(problem, model_state, params)
        while True:
            self.step(problem, model_state, opt_state)
            if is_implemented(problem, Problem.callback):
                problem.callback(model_state, opt_state)
            ok, result = self.terminate(problem, model_state, opt_state)
            if ok:
                break
        solution: Solution[S, T] = self.postprocess(
            problem, model_state, opt_state, result
        )
        return solution
