# ruff: noqa: EM101, PT017, TRY003
"""CPU-only behavioral checks for the safeguarded Newton policy defaults.

These checks use a tiny quadratic problem and replace only the linear solve.
They intentionally observe public effects (PCG arguments, trial updates, and
receipts), rather than reimplementing the Newton routine.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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
    """Diagonal SPD energy with update and CCD probes."""

    model = Model()

    def __init__(self, diagonal: tuple[float, float] = (2.0, 6.0)) -> None:
        self.diagonal = torch.tensor(diagonal, dtype=torch.float64)
        self.updates: list[torch.Tensor] = []
        self.ccd_directions: list[torch.Tensor] = []
        self.ccd_fraction = 1.0

    def update(self, state: SimpleNamespace, values: torch.Tensor) -> None:
        self.updates.append(values.detach().clone())
        state.u.copy_(values)

    def fun(self, state: SimpleNamespace) -> torch.Tensor:
        return 0.5 * torch.dot(state.u, self.diagonal * state.u)

    def grad(self, state: SimpleNamespace) -> torch.Tensor:
        return self.diagonal * state.u

    def hess_diag(self, _state: SimpleNamespace) -> torch.Tensor:
        return self.diagonal

    def hess_prod(
        self, _state: SimpleNamespace, direction: torch.Tensor
    ) -> torch.Tensor:
        return self.diagonal * direction

    def max_step_size(
        self, _state: SimpleNamespace, direction: torch.Tensor
    ) -> torch.Tensor:
        self.ccd_directions.append(direction.detach().clone())
        return torch.tensor(self.ccd_fraction, dtype=torch.float64)


def state() -> SimpleNamespace:
    return SimpleNamespace(u=torch.tensor([1.0, -1.0], dtype=torch.float64))


def with_pcg(replacement: Any, action: Any) -> None:
    original = accelerated_solvers.pcg
    accelerated_solvers.pcg = replacement
    try:
        action()
    finally:
        accelerated_solvers.pcg = original


def check_pcg_does_not_relax_the_requested_true_residual() -> None:
    """A recursively underestimated residual cannot relax the true-residual gate."""
    diagonal = torch.tensor([1.0, 2.0], dtype=torch.float64)
    rhs = torch.ones(2, dtype=torch.float64)
    original = torch.linalg.vector_norm
    calls = 0
    initial_norm: torch.Tensor | None = None

    def understate_recursive_norm(
        values: torch.Tensor, *args: Any, **kwargs: Any
    ) -> torch.Tensor:
        nonlocal calls, initial_norm
        calls += 1
        actual = original(values, *args, **kwargs)
        if calls == 1:
            initial_norm = actual
        # Calls are: RHS norm, recursive residual norm, true residual norm.
        # Report 0.30 so the recursive check enters the true-residual branch,
        # while the real first-step residual is 1/3 (> .32 but <= 1.05*.32).
        if calls == 2:
            assert initial_norm is not None
            return initial_norm * 0.30
        return actual

    torch.linalg.vector_norm = understate_recursive_norm
    try:
        _solution, receipt = accelerated_solvers.pcg(
            lambda vector: diagonal * vector,
            lambda vector: vector,
            rhs,
            rtol=0.32,
            max_steps=2,
        )
    finally:
        torch.linalg.vector_norm = original
    assert receipt["steps"] == 2
    assert receipt["relative_residual"] <= 0.32


def check_default_linear_policy_and_abs_jacobi() -> None:
    problem = Quadratic()
    current = state()
    calls: list[dict[str, Any]] = []

    def fake_pcg(
        matvec: Any, precondition: Any, _rhs: torch.Tensor, **kwargs: Any
    ) -> tuple[torch.Tensor, dict]:
        probe = torch.tensor([3.0, -12.0], dtype=torch.float64)
        calls.append(
            {
                "kwargs": kwargs,
                "preconditioned": precondition(probe),
                "matvec": matvec(probe),
            }
        )
        return -current.u.clone(), {"steps": 1, "relative_residual": 0.0}

    def run() -> None:
        _solved, receipt = accelerated_solvers.safeguarded_newton_step(
            problem, current, max_step_norm=1.0
        )
        assert receipt["shift"] == 0.0

    with_pcg(fake_pcg, run)
    assert len(calls) == 1
    assert calls[0]["kwargs"] == {"rtol": 1.0e-3, "max_steps": 1000}
    torch.testing.assert_close(
        calls[0]["preconditioned"],
        torch.tensor([1.5, -2.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        calls[0]["matvec"], torch.tensor([6.0, -72.0], dtype=torch.float64)
    )


def check_shift_attempt_sequence_uses_raw_mean_diagonal() -> None:
    class RawDiagonalPolicy(Quadratic):
        def hess_diag(self, _state: SimpleNamespace) -> torch.Tensor:
            # Its signed mean is +2 whereas the mean absolute diagonal is 3.
            # The fake linear solver below means this need not be an SPD model.
            return torch.tensor([-1.0, 5.0], dtype=torch.float64)

    problem = RawDiagonalPolicy()
    current = state()
    shifts: list[float] = []

    def reject(
        _matvec: Any, precondition: Any, rhs: torch.Tensor, **_kwargs: Any
    ) -> tuple[torch.Tensor, dict]:
        # This probe distinguishes the required abs(diag + shift) Jacobi
        # denominator from a raw diagonal denominator.
        probe = torch.ones_like(rhs)
        shifts.append(float((probe / precondition(probe)).mean()))
        raise accelerated_solvers.LinearRejection("deliberate linear rejection")

    def run() -> None:
        try:
            accelerated_solvers.safeguarded_newton_step(
                problem, current, max_step_norm=1.0
            )
        except accelerated_solvers.ForwardConvergenceError as error:
            assert str(error) == "Newton regularization exhausted"
            retries = error.receipt["retries"]
            assert len(retries) == 8
            assert [item["shift"] for item in retries] == [
                0.0,
                2.0,
                20.0,
                200.0,
                2_000.0,
                20_000.0,
                200_000.0,
                2_000_000.0,
            ]
        else:  # pragma: no cover
            raise AssertionError("linear rejections were silently accepted")

    with_pcg(reject, run)
    # The initial probe sees abs(raw diagonal); subsequent values prove that
    # regularization is scaled from the literal, signed raw mean (2), not 3.
    torch.testing.assert_close(
        torch.tensor(shifts),
        torch.tensor(
            [3.0, 4.0, 22.0, 202.0, 2_002.0, 20_002.0, 200_002.0, 2_000_002.0]
        ),
    )


def check_jacobi_uses_absolute_shifted_system() -> None:
    class RawDiagonalPolicy(Quadratic):
        def hess_diag(self, _state: SimpleNamespace) -> torch.Tensor:
            return torch.tensor([-1.0, 5.0], dtype=torch.float64)

        # Keep the acceptance calculation independently convex.  The PCG
        # replacement is what controls the candidate direction here.
        def fun(self, current: SimpleNamespace) -> torch.Tensor:
            return 0.5 * torch.dot(current.u, current.u)

        def grad(self, current: SimpleNamespace) -> torch.Tensor:
            return current.u.clone()

    problem = RawDiagonalPolicy()
    current = state()
    denominators: list[torch.Tensor] = []

    def reject_then_accept(
        _matvec: Any, precondition: Any, _rhs: torch.Tensor, **_kwargs: Any
    ) -> tuple[torch.Tensor, dict]:
        count = len(denominators)
        probe = torch.tensor([1.0, 7.0], dtype=torch.float64)
        denominators.append(probe / precondition(probe))
        if count == 0:
            raise accelerated_solvers.LinearRejection("force the raw-mean shift")
        return -current.u.clone(), {"steps": 1, "relative_residual": 0.0}

    with_pcg(
        reject_then_accept,
        lambda: accelerated_solvers.safeguarded_newton_step(
            problem, current, max_step_norm=1.0
        ),
    )
    torch.testing.assert_close(
        denominators[0], torch.tensor([1.0, 5.0], dtype=torch.float64)
    )
    # shift=2: abs([-1, 5] + 2) == [1, 7], rather than abs([-1, 5]) + 2.
    torch.testing.assert_close(
        denominators[1], torch.tensor([1.0, 7.0], dtype=torch.float64)
    )


def check_armijo_trial_limit_and_rollback() -> None:
    class AlwaysReject(Quadratic):
        def fun(self, current: SimpleNamespace) -> torch.Tensor:
            if torch.equal(current.u, torch.tensor([1.0, -1.0], dtype=torch.float64)):
                return super().fun(current)
            return torch.tensor(math.inf, dtype=torch.float64)

    problem = AlwaysReject()
    current = state()

    def first_then_abort(*_args: Any, **_kwargs: Any) -> tuple[torch.Tensor, dict]:
        # The deliberate uncaught error stops immediately after this Armijo
        # episode, leaving its update count observable without testing retries.
        raise_after = getattr(first_then_abort, "called", False)
        first_then_abort.called = True
        if raise_after:
            raise RuntimeError("stop after one Armijo episode")
        return -current.u.clone(), {"steps": 1, "relative_residual": 0.0}

    def run() -> None:
        try:
            accelerated_solvers.safeguarded_newton_step(
                problem, current, max_step_norm=1.0
            )
        except RuntimeError as error:
            assert str(error) == "stop after one Armijo episode"
        else:  # pragma: no cover
            raise AssertionError("Armijo rejection unexpectedly accepted")

    with_pcg(first_then_abort, run)
    # Eight trial updates, then the exact original free state is restored.
    assert len(problem.updates) == 9
    torch.testing.assert_close(
        problem.updates[-1], torch.tensor([1.0, -1.0], dtype=torch.float64)
    )
    torch.testing.assert_close(
        current.u, torch.tensor([1.0, -1.0], dtype=torch.float64)
    )


def check_accepted_receipt_counts_all_line_search_trials() -> None:
    class RejectTwice(Quadratic):
        def __init__(self) -> None:
            super().__init__()
            self.trials = 0

        def fun(self, current: SimpleNamespace) -> torch.Tensor:
            if not torch.equal(
                current.u, torch.tensor([1.0, -1.0], dtype=torch.float64)
            ):
                self.trials += 1
                if self.trials <= 2:
                    return torch.tensor(math.inf, dtype=torch.float64)
            return super().fun(current)

    problem = RejectTwice()
    current = state()

    def direction(*_args: Any, **_kwargs: Any) -> tuple[torch.Tensor, dict]:
        return -current.u.clone(), {"steps": 1, "relative_residual": 0.0}

    def run() -> None:
        _solved, receipt = accelerated_solvers.safeguarded_newton_step(
            problem, current, max_step_norm=1.0
        )
        assert receipt["backtracks"] == 2
        assert receipt["line_search_trials"] == 3
        assert receipt["alpha"] == 0.25

    with_pcg(direction, run)
    assert problem.trials == 3


def check_coordinate_cap_and_single_ccd_margin() -> None:
    problem = Quadratic()
    # FeasibleExpressionProblem has already turned a raw 0.5 CCD fraction
    # into 0.45.  Newton must consume that result once, without a second .9.
    problem.ccd_fraction = 0.45
    current = state()

    def direction(*_args: Any, **_kwargs: Any) -> tuple[torch.Tensor, dict]:
        return torch.tensor([-10.0, 0.0], dtype=torch.float64), {
            "steps": 1,
            "relative_residual": 0.0,
        }

    def run() -> None:
        _solved, receipt = accelerated_solvers.safeguarded_newton_step(
            problem,
            current,
            # Explicit cap isolates the cap and CCD policy from rest geometry.
            max_step_norm=1.0,
        )
        assert receipt["ccd"] == 0.45
        assert math.isclose(receipt["alpha"], 0.045, rel_tol=0.0, abs_tol=1.0e-15)

    with_pcg(direction, run)
    assert len(problem.ccd_directions) == 1
    torch.testing.assert_close(
        problem.ccd_directions[0], torch.tensor([-1.0, 0.0], dtype=torch.float64)
    )


def check_feasible_expression_problem_applies_ccd_safety_once() -> None:
    """Newton scopes .9 to its CCD query and restores the ordinary .95 FEP safety."""
    feasible = object.__new__(accelerated_solvers.FeasibleExpressionProblem)
    feasible.model = Model()
    feasible.collision_step_safety = 0.95
    parent = accelerated_solvers.FeasibleExpressionProblem.__mro__[1]
    original = parent.max_step_size
    raw_fraction = torch.tensor(0.5, dtype=torch.float64)

    def raw_ccd(
        _self: object, _state: object, _direction: torch.Tensor
    ) -> torch.Tensor:
        return raw_fraction

    parent.max_step_size = raw_ccd
    try:
        direction = torch.ones(2, dtype=torch.float64)
        ordinary = float(feasible.max_step_size(object(), direction))
        assert ordinary == 0.475
        wrapped = accelerated_solvers.CachedProblem(feasible)
        assert (
            accelerated_solvers.newton_ccd_fraction(wrapped, object(), direction)
            == 0.45
        )
        assert feasible.collision_step_safety == 0.95
        raw_fraction = torch.tensor(1.0, dtype=torch.float64)
        assert float(feasible.max_step_size(object(), direction)) == 1.0
        assert (
            accelerated_solvers.newton_ccd_fraction(wrapped, object(), direction) == 1.0
        )
        assert feasible.collision_step_safety == 0.95
    finally:
        parent.max_step_size = original


def check_cached_problem_optional_wall_budget() -> None:
    """No deadline is imposed unless the caller explicitly supplies one."""
    delegate = Quadratic()
    current = state()
    original = accelerated_solvers.time.perf_counter
    clock = 0.0

    def now() -> float:
        return clock

    accelerated_solvers.time.perf_counter = now
    try:
        unbudgeted = accelerated_solvers.CachedProblem(delegate)
        bounded = accelerated_solvers.CachedProblem(delegate, wall_seconds=1.0)
        clock = 1.0e12
        assert math.isfinite(float(unbudgeted.fun(current)))
        try:
            bounded.fun(current)
        except accelerated_solvers.ForwardConvergenceError as error:
            assert str(error) == "declared forward wall budget exhausted"
        else:  # pragma: no cover
            raise AssertionError("explicit wall budget did not fail at its deadline")
    finally:
        accelerated_solvers.time.perf_counter = original


def check_runtime_requires_rest_points_and_records_one_ccd_margin() -> None:
    """The runtime computes the mesh cap and keeps its ordinary CCD policy."""
    import mesh_step_scale

    reference = SimpleNamespace(
        forward=SimpleNamespace(model=object()), collision_step_safety=0.95
    )
    try:
        accelerated_solvers.accelerate_runtime(reference, "newton_diag")
    except AssertionError as error:
        assert "rest_points" in str(error)
    else:  # pragma: no cover
        raise AssertionError("Newton runtime accepted an unspecified rest geometry")

    original = mesh_step_scale.mean_rest_edge_length
    seen: list[tuple[object, object]] = []
    marker = object()

    def fake_mean(model: object, rest_points: object) -> float:
        seen.append((model, rest_points))
        return 0.006

    mesh_step_scale.mean_rest_edge_length = fake_mean
    try:
        runtime = accelerated_solvers.accelerate_runtime(
            reference, "newton_diag", rest_points=marker
        )
    finally:
        mesh_step_scale.mean_rest_edge_length = original
    assert seen == [(reference.forward.model, marker)]
    assert runtime.newton_max_step_norm == 0.003
    assert runtime.collision_step_safety == 0.95
    assert runtime.newton_parameters == {
        "linear_rtol": 1.0e-3,
        "linear_max_steps": 1000,
        "preconditioner": "abs_jacobi",
        "initial_shift": 0.0,
        "shift_policy": "reset",
        "shift_scale": "mean(diag(H))",
        "shift_multiplier": 10.0,
        "max_shift_attempts": 8,
        "mean_mesh_edge_length_m": 0.006,
        "max_coordinate_displacement_m": 0.003,
        "armijo": 1.0e-4,
        "backtracking_factor": 0.5,
        "max_backtracking_trials": 8,
        "ccd_safety_factor": 0.9,
        "max_steps": 100,
        "wall_seconds": None,
    }


if __name__ == "__main__":
    check_pcg_does_not_relax_the_requested_true_residual()
    check_default_linear_policy_and_abs_jacobi()
    check_shift_attempt_sequence_uses_raw_mean_diagonal()
    check_jacobi_uses_absolute_shifted_system()
    check_armijo_trial_limit_and_rollback()
    check_accepted_receipt_counts_all_line_search_trials()
    check_coordinate_cap_and_single_ccd_margin()
    check_feasible_expression_problem_applies_ccd_safety_once()
    check_cached_problem_optional_wall_budget()
    check_runtime_requires_rest_points_and_records_one_ccd_margin()
    print("Newton policy CPU check passed")
