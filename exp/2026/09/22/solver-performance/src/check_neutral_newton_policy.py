# ruff: noqa: EM101, PT017, TRY003
"""CPU checks for opt-in mean-absolute Newton regularization."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import torch
from check_newton_policy import Quadratic, accelerated_solvers, state, with_pcg


class MixedDiagonal(Quadratic):
    def hess_diag(self, _state: SimpleNamespace) -> torch.Tensor:
        return torch.tensor([-1.0, 5.0], dtype=torch.float64)


def check_retry_scales() -> None:
    """Both linear and Armijo rejection escalate from the selected scale."""
    for policy, scale in (("signed_mean", 2.0), ("mean_abs", 3.0)):
        for rejection in ("linear", "armijo"):
            check_retry_case(policy, scale, rejection)


def check_retry_case(policy: str, scale: float, rejection: str) -> None:
    current = state()
    problem = MixedDiagonal()
    if rejection == "armijo":
        # Every trial is rejected, while the starting energy is finite.
        initial = current.u.clone()
        problem.fun = lambda s: torch.tensor(
            0.0 if torch.equal(s.u, initial) else torch.inf,
            dtype=torch.float64,
        )

    def linear(*_args: Any, **_kwargs: Any) -> tuple[torch.Tensor, dict]:
        if rejection == "linear":
            raise accelerated_solvers.LinearRejection("test rejection")
        return -current.u.clone(), {"steps": 1, "relative_residual": 0.0}

    def run() -> None:
        try:
            accelerated_solvers.safeguarded_newton_step(
                problem,
                current,
                max_step_norm=1.0,
                max_shift_attempts=3,
                max_backtracking_trials=2,
                shift_scale_policy=policy,
            )
        except accelerated_solvers.ForwardConvergenceError as error:
            assert str(error) == "Newton regularization exhausted"
            retries = error.receipt["retries"]
            assert [row["shift"] for row in retries] == [0.0, scale, 10 * scale]
            assert all(row["shift_scale"] == scale for row in retries)
            assert all(row["shift_scale_policy"] == policy for row in retries)
        else:
            raise AssertionError("deliberate rejection unexpectedly succeeded")

    with_pcg(linear, run)
    torch.testing.assert_close(current.u, state().u, rtol=0, atol=0)


class DoubleWell(Quadratic):
    """Bounded energy whose starting Hessian has a negative signed mean."""

    def fun(self, current: SimpleNamespace) -> torch.Tensor:
        x, y = current.u
        return 0.25 * x**4 - x**2 + 0.5 * y**2

    def grad(self, current: SimpleNamespace) -> torch.Tensor:
        x, y = current.u
        return torch.stack((x**3 - 2 * x, y))

    def hess_diag(self, current: SimpleNamespace) -> torch.Tensor:
        x, _y = current.u
        return torch.stack((3 * x**2 - 2, x.new_tensor(1.0)))

    def hess_prod(
        self, current: SimpleNamespace, direction: torch.Tensor
    ) -> torch.Tensor:
        return self.hess_diag(current) * direction


def check_real_pcg_and_full_solver() -> None:
    problem = DoubleWell()
    current = SimpleNamespace(u=torch.tensor([0.1, 0.1], dtype=torch.float64))
    assert float(problem.hess_diag(current).mean()) < 0
    solved, result = accelerated_solvers.safeguarded_newton(
        problem,
        current,
        atol=1e-10,
        max_steps=100,
        max_step_norm=0.25,
        shift_scale_policy="mean_abs",
    )
    assert float(torch.linalg.vector_norm(problem.grad(solved))) <= 1e-10
    torch.testing.assert_close(
        solved.u,
        torch.tensor([2**0.5, 0.0], dtype=torch.float64),
        atol=1e-9,
        rtol=0,
    )
    first = result["trace"][0]
    assert first["regularization_retries"][0]["shift"] == 0
    assert first["shift"] > 0
    assert math.isclose(first["shift_scale"], 1.485, rel_tol=0, abs_tol=1e-15)
    assert all(row["shift_scale_policy"] == "mean_abs" for row in result["trace"])


if __name__ == "__main__":
    check_retry_scales()
    check_real_pcg_and_full_solver()
    print("Neutral Newton mean-absolute shift policy CPU checks passed")
