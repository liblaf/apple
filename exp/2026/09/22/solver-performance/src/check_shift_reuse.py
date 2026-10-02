# ruff: noqa: E402
"""CPU checks that shift reuse reduces retries without changing equilibrium."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [
    str(HERE),
    str(HERE.parents[2] / "21/joint-activation-material-mandible/src"),
]
from accelerated_solvers import CachedProblem, safeguarded_newton
from check_accelerated_solvers import _problem, _QuadraticProblem


class DoubleWell(_QuadraticProblem):
    def fun(self, state: SimpleNamespace) -> torch.Tensor:
        x, y = state.u
        return 0.25 * x**4 - 0.5 * x**2 + 0.5 * y**2

    def grad(self, state: SimpleNamespace) -> torch.Tensor:
        x, y = state.u
        return torch.stack((x**3 - x, y))

    def hess_diag(self, state: SimpleNamespace) -> torch.Tensor:
        x, _ = state.u
        return torch.stack((3 * x * x - 1, x.new_tensor(1.0)))

    def hess_prod(
        self, state: SimpleNamespace, direction: torch.Tensor
    ) -> torch.Tensor:
        return self.hess_diag(state) * direction


def check_reuse():
    results = {}
    for policy in ("reset", "reuse"):
        raw = DoubleWell(
            torch.eye(2, dtype=torch.float64), torch.zeros(2, dtype=torch.float64)
        )
        problem = CachedProblem(raw)
        state = SimpleNamespace(u=torch.tensor([0.1, 0.3], dtype=torch.float64))
        state, receipt = safeguarded_newton(
            problem,
            state,
            atol=1e-10,
            linear_rtol=1e-10,
            max_steps=50,
            max_step_norm=0.25,
            shift_policy=policy,
        )
        assert float(torch.linalg.vector_norm(problem.grad(state))) <= 1e-10
        torch.testing.assert_close(
            state.u, torch.tensor([1.0, 0.0], dtype=torch.float64), atol=1e-9, rtol=0
        )
        retries = sum(len(row["regularization_retries"]) for row in receipt["trace"])
        results[policy] = retries
        if policy == "reuse":
            assert any(row["initial_shift"] > 0 for row in receipt["trace"])
            assert receipt["trace"][-1]["shift"] < receipt["trace"][0]["shift"]
    assert results["reuse"] < results["reset"], results
    print("Double-well retry counts:", results)


def check_spd_and_default():
    endpoints = []
    for policy in ("reset", "reuse"):
        raw = _problem()
        state = SimpleNamespace(u=torch.zeros(3, dtype=torch.float64))
        state, receipt = safeguarded_newton(
            CachedProblem(raw),
            state,
            atol=1e-12,
            linear_rtol=1e-12,
            max_step_norm=10.0,
            shift_policy=policy,
        )
        assert receipt["steps"] == 1
        assert receipt["trace"][0]["shift"] == 0
        endpoints.append(state.u.clone())
    torch.testing.assert_close(*endpoints, rtol=0, atol=0)


if __name__ == "__main__":
    check_reuse()
    check_spd_and_default()
    print("shift-reuse CPU checks passed")
