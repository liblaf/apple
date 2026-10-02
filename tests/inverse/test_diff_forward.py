from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from liblaf.apple.forward import Forward
from liblaf.apple.inverse import DifferentiableForward, ImplicitNumericalError
from liblaf.apple.solvers.linalg import LinearSolver
from liblaf.apple.solvers.linalg import Result as LinearResult
from liblaf.apple.solvers.optim import Result as OptimizerResult


class _DofMap:
    def __init__(self) -> None:
        self.fixed_values = torch.tensor([0.0])

    def to_free_grad(self, full: torch.Tensor) -> torch.Tensor:
        return full.flatten()

    def to_full_grad(self, free: torch.Tensor) -> torch.Tensor:
        return free.reshape(1, 1)

    def to_free_hess_diag(self, full: torch.Tensor) -> torch.Tensor:
        return full.flatten()


class _Model:
    def __init__(self) -> None:
        self.dof_map = _DofMap()
        self.materials: dict[str, dict[str, torch.Tensor]] = {
            "material": {"x": torch.tensor([0.0])}
        }

    def get_materials(self) -> dict[str, dict[str, torch.Tensor]]:
        return {name: dict(values) for name, values in self.materials.items()}

    def set_materials(self, materials: dict[str, dict[str, torch.Tensor]]) -> None:
        self.materials = {name: dict(values) for name, values in materials.items()}

    def init(self) -> SimpleNamespace:
        return SimpleNamespace(u=torch.full((1, 1), -99.0))

    def update(self, state: SimpleNamespace, u: torch.Tensor) -> None:
        state.u.copy_(u)

    def grad(self, state: SimpleNamespace) -> torch.Tensor:
        x = self.materials["material"]["x"]
        return 2.0 * state.u - x.reshape(1, 1)

    def hess_diag(self, state: SimpleNamespace) -> torch.Tensor:
        return torch.full_like(state.u, 2.0)

    def hess_prod(self, state: SimpleNamespace, p: torch.Tensor) -> torch.Tensor:
        del state
        return 2.0 * p

    def mixed_derivative_prod(self, state: SimpleNamespace, p: torch.Tensor) -> None:
        del state
        x = self.materials["material"]["x"]
        x.grad = -p.flatten().clone()


class _Forward:
    def __init__(self, model: _Model, *, mode: str = "exact") -> None:
        self.model = model
        self.state = model.init()
        self.mode = mode
        self.last_solution: SimpleNamespace | None = None

    def step(self) -> SimpleNamespace:
        x = self.model.materials["material"]["x"]
        if self.mode == "nan":
            self.state.u.fill_(torch.nan)
        elif self.mode == "residual":
            self.state.u.zero_()
        else:
            self.state.u.copy_(x.reshape(1, 1) / 2.0)
        result = (
            OptimizerResult.MAX_STEPS_REACHED
            if self.mode == "failure"
            else OptimizerResult.SUCCESS
        )
        self.last_solution = SimpleNamespace(result=result, success=result.success)
        return self.last_solution


class _AdjointSolver:
    def __init__(self, mode: str = "exact") -> None:
        self.mode = mode
        self.calls = 0
        self.states: list[SimpleNamespace] = []
        self.fixed_values: list[torch.Tensor] = []

    def solve(self, problem: object, params: torch.Tensor) -> SimpleNamespace:
        del params
        self.calls += 1
        typed_problem = cast("Any", problem)
        self.states.append(typed_problem.model_state)
        self.fixed_values.append(typed_problem.model.dof_map.fixed_values.clone())
        if self.mode == "nan":
            p = torch.full_like(typed_problem.b, torch.nan)
        elif self.mode == "residual":
            p = torch.zeros_like(typed_problem.b)
        else:
            p = typed_problem.b / 2.0
        result = (
            LinearResult.MAX_STEPS_REACHED
            if self.mode == "failure"
            else LinearResult.SUCCESS
        )
        return SimpleNamespace(params=p, result=result, success=result.success)


def _differentiable(
    *,
    forward_mode: str = "exact",
    adjoint_mode: str = "exact",
    require_convergence: bool = True,
) -> tuple[DifferentiableForward, _AdjointSolver]:
    solver = _AdjointSolver(adjoint_mode)
    differentiable = DifferentiableForward(
        cast("Forward", _Forward(_Model(), mode=forward_mode)),
        adjoint_solver=cast("LinearSolver", solver),
        forward_residual_rtol=1.0e-6,
        adjoint_residual_rtol=1.0e-6,
        require_convergence=require_convergence,
    )
    return differentiable, solver


def test_interleaved_forwards_keep_independent_solved_state_and_restore_current_state() -> (
    None
):
    differentiable, solver = _differentiable()
    first = torch.tensor([2.0], requires_grad=True)
    second = torch.tensor([8.0], requires_grad=True)

    u_first = differentiable.forward({"material": {"x": first}})
    u_second = differentiable.forward({"material": {"x": second}})
    current_state = differentiable.state
    current_material = differentiable.model.get_materials()["material"]["x"]
    current_fixed_values = torch.tensor([8.0])
    differentiable.model.dof_map.fixed_values = current_fixed_values

    u_first.sum().backward()

    torch.testing.assert_close(u_first, torch.tensor([[1.0]]))
    torch.testing.assert_close(u_second, torch.tensor([[4.0]]))
    torch.testing.assert_close(first.grad, torch.tensor([0.5]))
    assert solver.states[0] is not current_state
    torch.testing.assert_close(solver.states[0].u, torch.tensor([[1.0]]))
    torch.testing.assert_close(solver.fixed_values[0], torch.tensor([0.0]))
    assert differentiable.state is current_state
    assert differentiable.model.dof_map.fixed_values is current_fixed_values
    assert differentiable.model.get_materials()["material"]["x"] is current_material


@pytest.mark.parametrize("mode", ["failure", "residual", "nan"])
def test_forward_rejects_failed_nonconverged_or_nonfinite_solution(mode: str) -> None:
    differentiable, _ = _differentiable(forward_mode=mode)

    error = ImplicitNumericalError if mode == "nan" else RuntimeError
    with pytest.raises(error):
        differentiable.forward({"material": {"x": torch.tensor([2.0])}})

    assert differentiable.last_forward_receipt is not None
    if mode != "failure":
        assert not differentiable.last_forward_receipt.success


@pytest.mark.parametrize("mode", ["failure", "residual", "nan"])
def test_adjoint_rejects_failed_nonconverged_or_nonfinite_solution(mode: str) -> None:
    differentiable, _ = _differentiable(adjoint_mode=mode)
    x = torch.tensor([2.0], requires_grad=True)
    u = differentiable.forward({"material": {"x": x}})
    current_state = differentiable.state
    current_material = differentiable.model.get_materials()["material"]["x"]
    current_fixed_values = torch.tensor([5.0])
    differentiable.model.dof_map.fixed_values = current_fixed_values

    error = ImplicitNumericalError if mode == "nan" else RuntimeError
    with pytest.raises(error):
        u.sum().backward()

    assert differentiable.state is current_state
    assert differentiable.model.get_materials()["material"]["x"] is current_material
    assert differentiable.model.dof_map.fixed_values is current_fixed_values


@pytest.mark.parametrize("mode", ["failure", "residual"])
def test_permissive_forward_accepts_finite_approximate_solutions(mode: str) -> None:
    differentiable, _ = _differentiable(forward_mode=mode, require_convergence=False)
    x = torch.tensor([2.0], requires_grad=True)

    u = differentiable.forward({"material": {"x": x}})
    u.sum().backward()

    assert differentiable.last_forward_receipt is not None
    assert not differentiable.last_forward_receipt.success
    assert differentiable.last_forward_receipt.finite
    if mode == "failure":
        assert not differentiable.last_forward_receipt.solver_success
    else:
        assert not differentiable.last_forward_receipt.residual_success
    assert x.grad is not None
    assert bool(torch.isfinite(x.grad).all())


@pytest.mark.parametrize(("mode", "expected"), [("failure", 0.5), ("residual", 0.0)])
def test_permissive_adjoint_accepts_finite_approximate_gradients(
    mode: str, expected: float
) -> None:
    differentiable, _ = _differentiable(adjoint_mode=mode, require_convergence=False)
    x = torch.tensor([2.0], requires_grad=True)
    u = differentiable.forward({"material": {"x": x}})

    u.sum().backward()

    assert differentiable.last_adjoint_receipt is not None
    assert not differentiable.last_adjoint_receipt.success
    assert differentiable.last_adjoint_receipt.finite
    assert x.grad is not None
    torch.testing.assert_close(x.grad, torch.tensor([expected]))


def test_zero_adjoint_rhs_skips_solver_and_clears_stale_solution() -> None:
    differentiable, solver = _differentiable()
    x = torch.tensor([2.0], requires_grad=True)
    u = differentiable.forward({"material": {"x": x}})
    differentiable.last_adjoint_solution = cast(
        "LinearSolver.Solution", SimpleNamespace()
    )

    (u * 0.0).sum().backward()

    assert solver.calls == 0
    assert differentiable.last_adjoint_solution is None
    assert differentiable.last_adjoint_receipt is not None
    assert differentiable.last_adjoint_receipt.success
    assert differentiable.last_adjoint_receipt.zero_rhs
    torch.testing.assert_close(x.grad, torch.zeros_like(x))
