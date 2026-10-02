# Copyright (c) 2026 liblaf
"""Regression coverage for the no-skin stress Newton-CG runtime adapter."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import pyvista as pv
import torch
import warp as wp

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "exp/2026/09/21/stress-activation-loss/src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from stress_material import StableNeoHookeanStress  # noqa: E402
from stress_physics import (  # noqa: E402
    FacePhysics,
    NewtonCgForwardOptimizer,
    SuccessPreferredFallbackSolver,
    _newton_policy,
    install_exact_bulk_diagonal,
)
from stress_study import deformation_diagnostics  # noqa: E402

from liblaf.apple.common import FIXED_MASK, FIXED_VALUE, LAMBDA, MU  # noqa: E402
from liblaf.apple.forward import Forward, ModelBuilder  # noqa: E402
from liblaf.apple.solvers.linalg.base import Result, Solution  # noqa: E402
from liblaf.apple.solvers.linalg.fallback._types import FallbackState  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="The stress FEM runtime requires CUDA."
)


class _DofMap:
    def to_free(self, u: torch.Tensor) -> torch.Tensor:
        return u


class _Model:
    dof_map = _DofMap()


class _State:
    def __init__(self, u: torch.Tensor) -> None:
        self.u = u


class _QuadraticProblem:
    model = _Model()

    def __init__(self, target: torch.Tensor) -> None:
        self.target = target
        self.grad_calls = 0

    def update(self, state: _State, free: torch.Tensor) -> None:
        state.u = free.clone()

    def fun(self, state: _State) -> torch.Tensor:
        residual = state.u - self.target
        return 0.5 * torch.dot(residual, residual)

    def grad(self, state: _State) -> torch.Tensor:
        self.grad_calls += 1
        return state.u - self.target

    def hess_diag(self, state: _State) -> torch.Tensor:
        return torch.ones_like(state.u)

    def hess_prod(self, state: _State, direction: torch.Tensor) -> torch.Tensor:
        del state
        return direction

    def hess_quad(self, state: _State, direction: torch.Tensor) -> torch.Tensor:
        del state
        return torch.dot(direction, direction)

    def max_step_size(self, state: _State, direction: torch.Tensor) -> torch.Tensor:
        del state, direction
        return torch.tensor(1.0)


def test_newton_runtime_uses_fresh_accepted_force() -> None:
    state = _State(torch.tensor([0.0, 0.0], dtype=torch.float64))
    problem = _QuadraticProblem(torch.tensor([2.0, -3.0], dtype=torch.float64))
    optimizer = NewtonCgForwardOptimizer(
        force_atol=1e-12,
        force_rtol=0.0,
        max_steps=4,
        max_step_norm=10.0,
        initial_shift_scale=0.0,
    )

    solution = optimizer.minimize(problem, state, state.u)

    assert solution.success
    torch.testing.assert_close(solution.params, problem.target)
    assert optimizer.last_receipt is not None
    assert optimizer.last_receipt["accepted_force_norm"] <= 1e-12
    assert optimizer.last_receipt["implicit_hessian"] == "unshifted physical Hessian"
    assert optimizer.last_receipt["work"]["grad_evaluations"] >= 2
    # The terminal gate must bypass the solve-local gradient cache.
    assert problem.grad_calls > optimizer.last_receipt["work"]["grad_evaluations"]


def test_newton_runtime_returns_finite_approximate_state() -> None:
    state = _State(torch.tensor([0.0], dtype=torch.float64))
    problem = _QuadraticProblem(torch.tensor([2.0], dtype=torch.float64))
    optimizer = NewtonCgForwardOptimizer(
        force_atol=1e-12,
        force_rtol=0.0,
        max_steps=1,
        max_step_norm=0.1,
        initial_shift_scale=0.0,
        require_convergence=False,
    )

    solution = optimizer.minimize(problem, state, state.u)

    assert not solution.success
    assert optimizer.last_receipt is not None
    assert optimizer.last_receipt["success"] is False
    assert optimizer.last_receipt["accepted_force_norm"] > 1e-12
    assert torch.isfinite(solution.params).all()
    torch.testing.assert_close(
        solution.params, torch.tensor([0.1], dtype=torch.float64)
    )


def test_mean_abs_shift_starts_without_zero_retry() -> None:
    cached_type, _, safeguarded_newton, _ = _newton_policy()
    state = _State(torch.tensor([0.0], dtype=torch.float64))
    problem = cached_type(_QuadraticProblem(torch.tensor([2.0], dtype=torch.float64)))

    _, receipt = safeguarded_newton(
        problem,
        state,
        atol=1e-2,
        max_steps=10,
        max_step_norm=10.0,
        initial_shift_scale=1.0,
        shift_scale_policy="mean_abs",
    )

    first = receipt["trace"][0]
    assert first["initial_shift_scale"] == 1.0
    assert first["initial_shift"] == first["shift_scale"]
    assert first["shift"] == first["shift_scale"]
    assert first["regularization_retries"] == []


def test_adjoint_fallback_prefers_finite_failed_candidate() -> None:
    class Problem:
        b = torch.tensor([1.0])

        @staticmethod
        def matvec(value: torch.Tensor) -> torch.Tensor:
            return value

    class Solver:
        def __init__(self, value: float) -> None:
            self.value = value

        def solve(self, problem: Problem, initial: torch.Tensor) -> Solution:
            del problem, initial
            state = SimpleNamespace(params=torch.tensor([self.value]))
            return Solution(result=Result.MAX_STEPS_REACHED, state=state, stats={})

    state = FallbackState(init_params=torch.zeros(1))
    solver = SuccessPreferredFallbackSolver(solvers=[Solver(float("nan")), Solver(2.0)])

    solver.compute(Problem(), state)

    assert int(state.best_index) == 1
    torch.testing.assert_close(state.params, torch.tensor([2.0]))


def test_finite_inverted_determinants_are_diagnostics() -> None:
    result = deformation_diagnostics(np.array([0.2, -0.3, 0.0]))

    assert result == {
        "detF_min": -0.3,
        "detF_max": 0.2,
        "inverted_all_cells": 2,
    }


def test_nonfinite_determinants_are_numerical_failure() -> None:
    from liblaf.apple.inverse import ImplicitNumericalError

    with pytest.raises(ImplicitNumericalError, match="determinant"):
        deformation_diagnostics(np.array([1.0, np.nan]))


@pytest.mark.parametrize("activation_model", ["stress", "strain"])
def test_face_solve_records_approximate_policy_failure(activation_model: str) -> None:
    physics = object.__new__(FacePhysics)
    physics.activation_model = activation_model
    physics.ids = np.array([0])
    physics.id_t = torch.tensor([0], device="cuda")
    physics.mesh = SimpleNamespace(n_cells=1)
    physics.points = np.zeros((1, 3), dtype=np.float64)
    physics.materials = {"muscle": {}}
    physics.solve_count = 0
    physics.forward_tolerance = {"force_atol": 1e-10}
    policy = NewtonCgForwardOptimizer(
        force_atol=1e-10,
        force_rtol=0.0,
        max_steps=1,
        max_step_norm=1.0,
        require_convergence=False,
    )
    policy.last_receipt = {
        "newton": None,
        "accepted_force_norm": 1.0,
        "accepted_force_threshold": 1e-10,
        "success": False,
    }
    physics.forward = SimpleNamespace(
        optimizer=policy,
        state=SimpleNamespace(
            u=torch.zeros((1, 3), device="cuda", dtype=torch.float64)
        ),
        model=SimpleNamespace(update=lambda state, seed: setattr(state, "u", seed)),
    )
    receipt = SimpleNamespace(
        success=False,
        absolute_residual=torch.tensor(1.0, device="cuda"),
        reference_norm=torch.tensor(1.0, device="cuda"),
        threshold=torch.tensor(1e-10, device="cuda"),
    )
    physics.diff = SimpleNamespace(
        require_convergence=False,
        last_adjoint_solution=None,
        last_solution=SimpleNamespace(success=False, result="max_steps"),
        last_forward_receipt=receipt,
        forward=lambda _materials: torch.zeros(
            (1, 3), device="cuda", dtype=torch.float64
        ),
    )

    output = physics.solve(
        torch.zeros((1, 3, 3), device="cuda", dtype=torch.float64),
        np.zeros((1, 3), dtype=np.float64),
    )

    assert torch.isfinite(output).all()
    assert physics.last_forward["success"] is False
    assert physics.last_forward["steps"] is None
    assert physics.last_forward["solver_valid"] is False
    field = "active_stress" if activation_model == "stress" else "activation_inv"
    expected_shape = (1, 3, 3) if activation_model == "stress" else (1, 6)
    assert physics.materials["muscle"][field].shape == expected_shape


def test_accuracy_context_restores_all_runtime_tolerances() -> None:
    optimizer = NewtonCgForwardOptimizer(
        force_atol=1e-10, force_rtol=1e-3, max_steps=2, max_step_norm=1.0
    )
    physics = object.__new__(FacePhysics)
    physics.forward = SimpleNamespace(optimizer=optimizer)
    physics.diff = SimpleNamespace(
        require_convergence=True,
        forward_residual_atol=1e-10,
        forward_residual_rtol=1e-3,
        adjoint_residual_atol=0.0,
        adjoint_residual_rtol=1e-7,
        adjoint_solver=SimpleNamespace(solvers=[SimpleNamespace(rtol=1e-7, atol=0.0)]),
    )
    physics.forward_tolerance = {
        "force_atol": 1e-10,
        "force_rtol": 1e-3,
        "adjoint_atol": 0.0,
        "adjoint_rtol": 1e-7,
        "newton_linear_rtol": 1e-3,
    }

    with physics.accuracy(0.1):
        assert optimizer.force_atol == pytest.approx(1e-11)
        assert optimizer.force_rtol == pytest.approx(1e-4)
        assert physics.diff.adjoint_residual_rtol == pytest.approx(1e-8)
        assert physics.diff.adjoint_solver.solvers[0].rtol == pytest.approx(1e-8)

    assert optimizer.force_atol == pytest.approx(1e-10)
    assert optimizer.force_rtol == pytest.approx(1e-3)
    assert physics.diff.forward_residual_atol == pytest.approx(1e-10)
    assert physics.diff.adjoint_residual_rtol == pytest.approx(1e-7)
    assert physics.forward_tolerance["adjoint_rtol"] == pytest.approx(1e-7)


def test_approximate_solves_context_restores_strict_mode() -> None:
    optimizer = NewtonCgForwardOptimizer(
        force_atol=1e-10, force_rtol=0.0, max_steps=2, max_step_norm=1.0
    )
    physics = object.__new__(FacePhysics)
    physics.forward = SimpleNamespace(optimizer=optimizer)
    physics.diff = SimpleNamespace(require_convergence=True)

    with physics.approximate_solves():
        assert optimizer.require_convergence is False
        assert physics.diff.require_convergence is False

    assert optimizer.require_convergence is True
    assert physics.diff.require_convergence is True


def _stress_runtime() -> tuple[Forward, torch.Tensor]:
    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.25, 0.25, 0.25],
        ],
        dtype=np.float64,
    )
    cells = np.array(
        [
            4,
            0,
            1,
            2,
            4,
            4,
            0,
            1,
            4,
            3,
            4,
            0,
            4,
            2,
            3,
            4,
            4,
            1,
            2,
            3,
        ],
        dtype=np.int64,
    )
    mesh = pv.UnstructuredGrid(
        cells, np.full(4, pv.CellType.TETRA, dtype=np.uint8), points
    )
    mesh.cell_data[LAMBDA.vtk] = np.full(mesh.n_cells, 2.0, dtype=np.float64)
    mesh.cell_data[MU.vtk] = np.full(mesh.n_cells, 1.0, dtype=np.float64)
    fixed = np.zeros((mesh.n_points, 3), dtype=bool)
    fixed[:4] = True
    mesh.point_data[FIXED_MASK.vtk] = fixed
    mesh.point_data[FIXED_VALUE.vtk] = np.zeros_like(points)
    builder = ModelBuilder()
    builder.add_vertices(mesh)
    builder.add_fixed(mesh)
    builder.add_potential(StableNeoHookeanStress.from_pyvista(mesh, name="stress"))
    install_exact_bulk_diagonal()
    forward = Forward(builder.finalize())
    forward.optimizer = NewtonCgForwardOptimizer(
        force_atol=1e-11,
        force_rtol=0.0,
        max_steps=20,
        max_step_norm=0.5,
        initial_shift_scale=0.0,
    )
    return forward, torch.as_tensor(points, device="cuda", dtype=torch.float64)


def test_newton_stress_fem_implicit_directional_derivative() -> None:
    previous_device = torch.get_default_device()
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_device("cuda")
        torch.set_default_dtype(torch.float64)
        wp.init()
        direction = torch.zeros((4, 3, 3), device="cuda", dtype=torch.float64)
        direction[0, 2, 2] = 1.0
        for scale in (0.01, 0.02):
            forward, _ = _stress_runtime()
            from liblaf.apple.inverse import DifferentiableForward

            diff = DifferentiableForward(
                forward,
                forward_residual_atol=1e-11,
                forward_residual_rtol=0.0,
                adjoint_residual_atol=0.0,
                adjoint_residual_rtol=1e-8,
            )
            q = torch.nn.Parameter(scale * direction)
            materials = forward.model.get_materials()
            materials["stress"]["active_stress"] = q
            u = diff.forward(materials)
            objective = u[4, 0]
            objective.backward()
            assert q.grad is not None
            analytic = float(torch.sum(q.grad * direction))
            assert abs(analytic) > 1e-8
            values = []
            for sign in (-1.0, 1.0):
                trial_forward, _ = _stress_runtime()
                trial = trial_forward.model.get_materials()
                trial["stress"]["active_stress"] = (
                    q + sign * 1e-5 * direction
                ).detach()
                trial_diff = DifferentiableForward(
                    trial_forward,
                    forward_residual_atol=1e-11,
                    forward_residual_rtol=0.0,
                )
                values.append(float(trial_diff.forward(trial)[4, 0]))
            finite_difference = (values[1] - values[0]) / (2e-5)
            assert abs(finite_difference - analytic) / abs(analytic) < 2e-3
            assert diff.last_forward_receipt is not None
            assert diff.last_forward_receipt.success
            assert diff.last_adjoint_receipt is not None
            assert diff.last_adjoint_receipt.success
    finally:
        torch.set_default_device(previous_device)
        torch.set_default_dtype(previous_dtype)
