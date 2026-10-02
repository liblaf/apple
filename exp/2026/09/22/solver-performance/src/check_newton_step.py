# ruff: noqa: EM101, PT017, TRY003
"""CPU contract checks for one accepted safeguarded Newton correction."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent
JOINT_SOURCE = HERE.parents[2] / "21/joint-activation-material-mandible/src"
sys.path[:0] = [str(HERE), str(JOINT_SOURCE)]
spec = importlib.util.spec_from_file_location(
    "accelerated_solvers", HERE / "accelerated_solvers.py"
)
assert spec is not None
assert spec.loader is not None
accelerated_solvers = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = accelerated_solvers
spec.loader.exec_module(accelerated_solvers)


class DofMap:
    def to_free(self, values: torch.Tensor) -> torch.Tensor:
        return values


class Model:
    dof_map = DofMap()


class Quadratic:
    model = Model()

    def update(self, state: SimpleNamespace, values: torch.Tensor) -> None:
        state.u.copy_(values)

    def fun(self, state: SimpleNamespace) -> torch.Tensor:
        return 0.5 * torch.dot(state.u, state.u)

    def grad(self, state: SimpleNamespace) -> torch.Tensor:
        return state.u.clone()

    def hess_diag(self, state: SimpleNamespace) -> torch.Tensor:
        return torch.ones_like(state.u)

    def hess_prod(
        self, _state: SimpleNamespace, direction: torch.Tensor
    ) -> torch.Tensor:
        return direction

    def max_step_size(
        self, _state: SimpleNamespace, _direction: torch.Tensor
    ) -> torch.Tensor:
        return torch.ones((), dtype=torch.float64)


class NegativeCurvature(Quadratic):
    def hess_prod(
        self, _state: SimpleNamespace, direction: torch.Tensor
    ) -> torch.Tensor:
        return -direction


class InvalidCcd(Quadratic):
    def max_step_size(
        self, _state: SimpleNamespace, _direction: torch.Tensor
    ) -> torch.Tensor:
        return torch.zeros((), dtype=torch.float64)


def step(problem: object, state: SimpleNamespace) -> tuple[SimpleNamespace, dict]:
    return accelerated_solvers.safeguarded_newton_step(
        problem,
        state,
        atol=1e-12,
        linear_rtol=1e-12,
        max_step_norm=0.1,
        armijo=0.25,
        preconditioner="diag",
    )


def check_single_step_improves_without_a_whole_solve_gate() -> None:
    state = SimpleNamespace(u=torch.tensor([1.0], dtype=torch.float64))
    solved, receipt = step(Quadratic(), state)
    assert solved is state
    assert 0 < float(state.u[0]) < 1
    assert receipt["force"] > 1e-12
    assert receipt["next_shift_ratio"] == 0


def check_full_solver_retains_iteration_budget_failure() -> None:
    state = SimpleNamespace(u=torch.tensor([1.0], dtype=torch.float64))
    try:
        accelerated_solvers.safeguarded_newton(
            Quadratic(),
            state,
            atol=1e-12,
            linear_rtol=1e-12,
            max_steps=1,
            max_step_norm=0.1,
        )
    except accelerated_solvers.ForwardConvergenceError as error:
        assert str(error) == "Newton iteration budget exhausted"
        assert error.receipt["grad_norm"] > 1e-12
    else:
        raise AssertionError("one limited Newton step unexpectedly converged")


def check_negative_curvature_retries_with_a_shift() -> None:
    state = SimpleNamespace(u=torch.tensor([1.0], dtype=torch.float64))
    _solved, receipt = step(NegativeCurvature(), state)
    assert receipt["shift"] > 1
    assert any(
        item["reason"] == "nonpositive or nonfinite CG curvature"
        for item in receipt["regularization_retries"]
    )
    assert 0 < float(state.u[0]) < 1


def check_invalid_ccd_failure_is_not_hidden() -> None:
    state = SimpleNamespace(u=torch.tensor([1.0], dtype=torch.float64))
    try:
        step(InvalidCcd(), state)
    except accelerated_solvers.ForwardConvergenceError as error:
        assert str(error) == "Newton CCD fraction is invalid"
    else:
        raise AssertionError("invalid CCD fraction unexpectedly accepted")


if __name__ == "__main__":
    check_single_step_improves_without_a_whole_solve_gate()
    check_full_solver_retains_iteration_budget_failure()
    check_negative_curvature_retries_with_a_shift()
    check_invalid_ccd_failure_is_not_hidden()
    print("Newton step CPU check passed")
