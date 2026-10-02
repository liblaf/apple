# ruff: noqa: EM101, PT017, PT018, TRY003
"""CPU-only behavioral checks for the experimental forward-solver variants.

Run this directly.  These fixtures deliberately avoid Warp, IPC, and CUDA: they
exercise the contracts that must hold before a physical solve is benchmarked.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from liblaf.apple.solvers.optim import Pncg

_HERE = Path(__file__).resolve().parent
_JOINT_SOURCE = _HERE.parents[2] / "21" / "joint-activation-material-mandible" / "src"
sys.path[:0] = [str(_HERE), str(_JOINT_SOURCE)]
_SPEC = importlib.util.spec_from_file_location(
    "accelerated_solvers", _HERE / "accelerated_solvers.py"
)
assert _SPEC is not None and _SPEC.loader is not None
accelerated_solvers = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = accelerated_solvers
_SPEC.loader.exec_module(accelerated_solvers)


class _DofMap:
    def to_free(self, values: torch.Tensor) -> torch.Tensor:
        return values


class _QuadraticModel:
    def __init__(self) -> None:
        self.dof_map = _DofMap()


class _QuadraticProblem:
    """Small coupled SPD energy with visible derivative and update counts."""

    def __init__(self, matrix: torch.Tensor, target: torch.Tensor) -> None:
        self.matrix = matrix
        self.target = target
        self.model = _QuadraticModel()
        self.gradient_evaluations = 0
        self.updates = 0

    def update(self, state: SimpleNamespace, free: torch.Tensor) -> None:
        self.updates += 1
        state.u.copy_(free)

    def fun(self, state: SimpleNamespace) -> torch.Tensor:
        residual = state.u - self.target
        return 0.5 * torch.dot(residual, self.matrix @ residual)

    def grad(self, state: SimpleNamespace) -> torch.Tensor:
        self.gradient_evaluations += 1
        return self.matrix @ (state.u - self.target)

    def hess_diag(self, _state: SimpleNamespace) -> torch.Tensor:
        return torch.diagonal(self.matrix)

    def hess_prod(
        self, _state: SimpleNamespace, direction: torch.Tensor
    ) -> torch.Tensor:
        return self.matrix @ direction

    def hess_quad(
        self, state: SimpleNamespace, direction: torch.Tensor
    ) -> torch.Tensor:
        return torch.dot(direction, self.hess_prod(state, direction))

    def max_step_size(
        self, _state: SimpleNamespace, _direction: torch.Tensor
    ) -> torch.Tensor:
        return torch.tensor(1.0, dtype=self.matrix.dtype)


def _problem() -> _QuadraticProblem:
    matrix = torch.tensor(
        [[7.0, 2.0, -1.0], [2.0, 5.0, 1.0], [-1.0, 1.0, 4.0]],
        dtype=torch.float64,
    )
    return _QuadraticProblem(
        matrix, torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    )


def _pncg() -> object:
    criteria = Pncg.ConvergenceCriteria(
        max_steps=30,
        atol_primary=1.0e-12,
        rtol_primary=0.0,
        atol_secondary=1.0e-12,
        rtol_secondary=0.0,
    )
    return accelerated_solvers.AcceptedForcePncg(criteria=criteria)


def check_gradient_cache_invalidation() -> None:
    delegate = _problem()
    problem = accelerated_solvers.CachedProblem(delegate)
    state = SimpleNamespace(u=torch.zeros(3, dtype=torch.float64))

    first = problem.grad(state)
    assert problem.grad(state) is first
    assert delegate.gradient_evaluations == 1
    # An external in-place tensor mutation must not return a stale force.
    state.u.add_(torch.tensor([0.1, 0.0, 0.0], dtype=torch.float64))
    changed = problem.grad(state)
    assert delegate.gradient_evaluations == 2
    assert not torch.equal(changed, first)
    # Delegate updates also invalidate, even when the replacement vector has
    # identical values to the current state.
    problem.update(state, state.u.clone())
    problem.grad(state)
    assert delegate.gradient_evaluations == 3
    assert problem.counts["grad_requests"] == 4
    assert problem.counts["grad_evaluations"] == 3


def check_cached_pncg_preserves_accepted_force() -> None:
    plain_delegate = _problem()
    plain_state = SimpleNamespace(u=torch.zeros(3, dtype=torch.float64))
    plain = accelerated_solvers.CachedProblem(plain_delegate, cache_gradient=False)
    plain_solution = _pncg().minimize(plain, plain_state, plain_state.u)
    assert plain_solution.success
    plain_force = torch.linalg.vector_norm(plain.grad(plain_state))

    cached_delegate = _problem()
    cached_state = SimpleNamespace(u=torch.zeros(3, dtype=torch.float64))
    cached = accelerated_solvers.CachedProblem(cached_delegate, cache_gradient=True)
    cached_solution = _pncg().minimize(cached, cached_state, cached_state.u)
    assert cached_solution.success
    cached_force = torch.linalg.vector_norm(cached.grad(cached_state))

    torch.testing.assert_close(
        cached_state.u, plain_state.u, rtol=1.0e-11, atol=1.0e-12
    )
    torch.testing.assert_close(cached_force, plain_force, rtol=1.0e-10, atol=1.0e-13)
    assert float(cached_force) <= 1.0e-12
    assert cached.counts["grad_evaluations"] < plain.counts["grad_evaluations"]


def check_newton_solves_coupled_spd_system() -> None:
    delegate = _problem()
    state = SimpleNamespace(u=torch.zeros(3, dtype=torch.float64))
    problem = accelerated_solvers.CachedProblem(delegate)
    solved, receipt = accelerated_solvers.safeguarded_newton(
        problem,
        state,
        atol=1.0e-12,
        linear_rtol=1.0e-12,
        max_steps=4,
        max_step_norm=10.0,
        armijo=0.25,
    )
    assert solved is state
    torch.testing.assert_close(state.u, delegate.target, rtol=1.0e-11, atol=1.0e-12)
    assert receipt["steps"] == 1
    assert receipt["trace"][0]["shift"] == 0.0
    assert receipt["trace"][0]["linear"]["relative_residual"] <= 1.05e-12


class _NonfiniteHvpProblem(_QuadraticProblem):
    def hess_prod(
        self, _state: SimpleNamespace, direction: torch.Tensor
    ) -> torch.Tensor:
        return torch.full_like(direction, torch.nan)


class _ArmijoFailureProblem(_QuadraticProblem):
    def fun(self, state: SimpleNamespace) -> torch.Tensor:
        if not torch.equal(state.u, torch.zeros_like(state.u)):
            return torch.tensor(torch.inf, dtype=state.u.dtype)
        return super().fun(state)


class _TerminalNonfiniteProblem(_QuadraticProblem):
    """Finite initial force and energy, then a nonfinite accepted endpoint force."""

    def grad(self, state: SimpleNamespace) -> torch.Tensor:
        self.gradient_evaluations += 1
        if torch.equal(state.u, torch.zeros_like(state.u)):
            return super().grad(state)
        return torch.full_like(state.u, torch.nan)


def check_newton_rejects_nonfinite_and_failed_trials() -> None:
    matrix = _problem().matrix
    target = _problem().target
    nonfinite_state = SimpleNamespace(u=torch.zeros(3, dtype=torch.float64))
    nonfinite = accelerated_solvers.CachedProblem(_NonfiniteHvpProblem(matrix, target))
    try:
        accelerated_solvers.safeguarded_newton(
            nonfinite, nonfinite_state, atol=1.0e-12, max_steps=1, max_step_norm=10.0
        )
    except accelerated_solvers.ForwardConvergenceError as error:
        assert "regularization exhausted" in str(error)
    else:  # pragma: no cover
        raise AssertionError("nonfinite Hessian product was silently accepted")
    torch.testing.assert_close(nonfinite_state.u, torch.zeros_like(nonfinite_state.u))

    armijo_state = SimpleNamespace(u=torch.zeros(3, dtype=torch.float64))
    armijo = accelerated_solvers.CachedProblem(_ArmijoFailureProblem(matrix, target))
    try:
        accelerated_solvers.safeguarded_newton(
            armijo, armijo_state, atol=1.0e-12, max_steps=1, max_step_norm=10.0
        )
    except accelerated_solvers.ForwardConvergenceError as error:
        assert "regularization exhausted" in str(error)
    else:  # pragma: no cover
        raise AssertionError("Armijo-exhausted Newton trial was silently accepted")
    torch.testing.assert_close(armijo_state.u, torch.zeros_like(armijo_state.u))


def check_newton_rejects_nonfinite_last_iteration_force() -> None:
    delegate = _TerminalNonfiniteProblem(_problem().matrix, _problem().target)
    state = SimpleNamespace(u=torch.zeros(3, dtype=torch.float64))
    problem = accelerated_solvers.CachedProblem(delegate)
    try:
        accelerated_solvers.safeguarded_newton(
            problem,
            state,
            atol=1.0e-12,
            linear_rtol=1.0e-12,
            max_steps=1,
            max_step_norm=10.0,
            armijo=0.25,
        )
    except accelerated_solvers.ForwardConvergenceError as error:
        assert "iteration budget exhausted" in str(error)
        assert error.receipt["grad_norm"] != error.receipt["grad_norm"]
    else:  # pragma: no cover
        raise AssertionError("nonfinite final Newton force was silently accepted")


if __name__ == "__main__":
    check_gradient_cache_invalidation()
    check_cached_pncg_preserves_accepted_force()
    check_newton_solves_coupled_spd_system()
    check_newton_rejects_nonfinite_and_failed_trials()
    check_newton_rejects_nonfinite_last_iteration_force()
    print("accelerated-solver CPU checks passed")
