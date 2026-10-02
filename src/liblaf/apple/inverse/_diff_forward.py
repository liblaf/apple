import logging
from collections.abc import Mapping
from typing import Any, Never, cast, override

import attrs
import optree
import torch
from jaxtyping import Float
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from liblaf.apple.forward import Forward, Model
from liblaf.apple.solvers.linalg import FallbackSolver, LinearSolver
from liblaf.apple.solvers.optim import Optimizer

type Free = Float[Tensor, " free"]
type Full = Float[Tensor, "points dim"]

logger: logging.Logger = logging.getLogger(__name__)


@attrs.define(frozen=True)
class SolveReceipt:
    """Fresh residual evidence for one implicit forward or adjoint solve."""

    success: bool
    solver_success: bool
    residual_success: bool
    finite: bool
    absolute_residual: Tensor
    reference_norm: Tensor
    threshold: Tensor
    zero_rhs: bool = False


class ImplicitSolveError(RuntimeError):
    """A finite implicit solve did not meet its declared convergence contract."""


class ImplicitNumericalError(ImplicitSolveError):
    """An implicit solve or its pullback produced a non-finite numerical value."""


@attrs.define
class DifferentiableForward:
    __wrapped__: Forward
    adjoint_solver: LinearSolver = attrs.field(factory=FallbackSolver)
    forward_residual_atol: float = 0.0
    forward_residual_rtol: float = 5.0e-4
    adjoint_residual_atol: float = 0.0
    adjoint_residual_rtol: float = 1.0e-5
    require_convergence: bool = True
    last_adjoint_solution: LinearSolver.Solution | None = None
    last_forward_receipt: SolveReceipt | None = None
    last_adjoint_receipt: SolveReceipt | None = None

    @property
    def last_solution(self) -> Optimizer.Solution | None:
        return self.__wrapped__.last_solution

    @property
    def model(self) -> Model:
        return self.__wrapped__.model

    @property
    def state(self) -> Model.State:
        return self.__wrapped__.state

    def receipt(
        self,
        residual: Free,
        reference: Free,
        *,
        atol: float,
        rtol: float,
        solver_success: bool = True,
        finite: bool = True,
        zero_rhs: bool = False,
    ) -> SolveReceipt:
        absolute_residual: Tensor = torch.linalg.vector_norm(residual)
        reference_norm: Tensor = torch.linalg.vector_norm(reference)
        threshold: Tensor = torch.maximum(
            torch.as_tensor(atol, dtype=reference.dtype, device=reference.device),
            torch.as_tensor(rtol, dtype=reference.dtype, device=reference.device)
            * reference_norm,
        )
        residual_success: bool = bool(
            torch.isfinite(absolute_residual)
            and torch.isfinite(reference_norm)
            and absolute_residual <= threshold
        )
        finite = finite and bool(
            torch.isfinite(absolute_residual) and torch.isfinite(reference_norm)
        )
        return SolveReceipt(
            success=solver_success and residual_success and finite,
            solver_success=solver_success,
            residual_success=residual_success,
            finite=finite,
            absolute_residual=absolute_residual,
            reference_norm=reference_norm,
            threshold=threshold,
            zero_rhs=zero_rhs,
        )

    def adjoint_solve(
        self, u_grad: Full, model_state: Model.State | None = None
    ) -> LinearSolver.Solution | None:
        u_grad: Free = self.model.dof_map.to_free_grad(u_grad)
        if not bool(torch.isfinite(u_grad).all()):
            _raise_numerical("Adjoint right-hand side contains non-finite values")
        if bool(torch.count_nonzero(u_grad) == 0):
            zero: Free = torch.zeros_like(u_grad)
            self.last_adjoint_solution = None
            self.last_adjoint_receipt = self.receipt(
                zero,
                zero,
                atol=self.adjoint_residual_atol,
                rtol=self.adjoint_residual_rtol,
                zero_rhs=True,
            )
            return None
        if model_state is None:
            model_state = self.state
        problem: _AdjointProblem = _AdjointProblem(
            b=-u_grad, model=self.model, model_state=model_state
        )
        if (
            self.last_adjoint_solution is not None
            and self.last_adjoint_solution.success
        ):
            params: Free = self.last_adjoint_solution.params
        else:
            params: Free = torch.zeros_like(u_grad)
        solution: LinearSolver.Solution = self.adjoint_solver.solve(problem, params)
        self.last_adjoint_solution = solution
        residual: Free = problem.matvec(solution.params) - problem.b
        self.last_adjoint_receipt = self.receipt(
            residual,
            problem.b,
            atol=self.adjoint_residual_atol,
            rtol=self.adjoint_residual_rtol,
            solver_success=solution.success,
            finite=bool(torch.isfinite(solution.params).all()),
        )
        if not self.last_adjoint_receipt.finite:
            _raise_numerical("Adjoint solve returned non-finite parameters or residual")
        if self.require_convergence and not solution.success:
            _raise_solve(f"Adjoint solve failed: {solution.result}")
        if self.require_convergence and not self.last_adjoint_receipt.residual_success:
            _raise_solve(
                "Adjoint solve residual exceeds tolerance: "
                f"{self.last_adjoint_receipt.absolute_residual.item():.3e} > "
                f"{self.last_adjoint_receipt.threshold.item():.3e}"
            )
        logger.info(solution)
        return solution

    def forward(self, materials: Mapping[str, Mapping[str, Tensor]]) -> Full:
        leaves, spec = optree.tree_flatten(cast("Any", materials))
        return _DifferentiableForward.apply(self, spec, *leaves)

    def step(self) -> Optimizer.Solution:
        return self.__wrapped__.step()


@attrs.define
class _AdjointProblem:
    def _default_preconditioner(self) -> Free:
        H_diag: Full = self.model.hess_diag(self.model_state)
        H_diag: Free = self.model.dof_map.to_free_hess_diag(H_diag)
        H_diag: Free = H_diag.abs()
        return H_diag.reciprocal()

    b: Free
    model: Model
    model_state: Model.State
    _preconditioner: Free = attrs.field(
        default=attrs.Factory(_default_preconditioner, takes_self=True)
    )

    def matvec(self, p_free: Free) -> Free:
        p_full: Full = self.model.dof_map.to_full_grad(p_free)
        output_full: Full = self.model.hess_prod(self.model_state, p_full)
        return self.model.dof_map.to_free_grad(output_full)

    def rmatvec(self, p_free: Free) -> Free:
        return self.matvec(p_free)

    def precondition(self, p_free: Free) -> Free:
        return self._preconditioner * p_free

    def rprecondition(self, p_free: Free) -> Free:
        return self.precondition(p_free)

    def preconditioner(self, p_free: Free) -> Free:
        return self.precondition(p_free)

    def rpreconditioner(self, p_free: Free) -> Free:
        return self.rprecondition(p_free)


class FunctionCtx(torch.autograd.function.FunctionCtx):
    needs_input_grad: tuple[bool, ...]
    saved_tensors: tuple[Tensor, ...]
    forward: DifferentiableForward
    spec: optree.PyTreeSpec
    solved_u: Full
    solved_fixed_values: Tensor


class _DifferentiableForward(Function):
    @staticmethod
    @override
    def forward(
        forward: DifferentiableForward, spec: optree.PyTreeSpec, *args: Tensor
    ) -> Tensor:
        materials: dict[str, dict[str, Tensor]] = cast(
            "dict[str, dict[str, Tensor]]", spec.unflatten(args)
        )
        forward.model.set_materials(materials)
        initial_residual: Free = forward.model.dof_map.to_free_grad(
            forward.model.grad(forward.state)
        )
        solution: Optimizer.Solution = forward.step()
        solved_u: Full = forward.state.u.detach().clone()
        solved_state: Model.State = _state_at(forward.model, solved_u)
        residual: Free = forward.model.dof_map.to_free_grad(
            forward.model.grad(solved_state)
        )
        forward.last_forward_receipt = forward.receipt(
            residual,
            initial_residual,
            atol=forward.forward_residual_atol,
            rtol=forward.forward_residual_rtol,
            solver_success=solution.success,
            finite=(
                bool(torch.isfinite(solved_u).all())
                and bool(torch.isfinite(forward.model.dof_map.fixed_values).all())
            ),
        )
        if not forward.last_forward_receipt.finite:
            _raise_numerical(
                "Forward solve returned non-finite displacement or residual"
            )
        if forward.require_convergence and not solution.success:
            _raise_solve(f"Forward solve failed: {solution.result}")
        if (
            forward.require_convergence
            and not forward.last_forward_receipt.residual_success
        ):
            _raise_solve(
                "Forward solve residual exceeds tolerance: "
                f"{forward.last_forward_receipt.absolute_residual.item():.3e} > "
                f"{forward.last_forward_receipt.threshold.item():.3e}"
            )
        return solved_u

    @staticmethod
    @override
    def setup_context(
        ctx: FunctionCtx, inputs: tuple[Any, ...], output: Tensor
    ) -> None:
        forward, spec, *args = inputs
        ctx.forward = forward
        ctx.spec = spec
        ctx.solved_u = output.detach().clone()
        ctx.solved_fixed_values = forward.model.dof_map.fixed_values.detach().clone()
        ctx.save_for_backward(*args)

    @staticmethod
    @once_differentiable
    @override
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> tuple[Tensor | None, ...]:
        original_materials: dict[str, dict[str, Tensor]] = (
            ctx.forward.model.get_materials()
        )
        original_state: Model.State = ctx.forward.__wrapped__.state
        original_fixed_values: Tensor = ctx.forward.model.dof_map.fixed_values
        try:
            leaves: list[Tensor] = [
                leaf.detach().requires_grad_(needs_grad)
                for leaf, needs_grad in zip(
                    ctx.saved_tensors, ctx.needs_input_grad[2:], strict=True
                )
            ]
            tmp_materials: dict[str, dict[str, Tensor]] = cast(
                "dict[str, dict[str, Tensor]]", ctx.spec.unflatten(leaves)
            )
            ctx.forward.model.set_materials(tmp_materials)
            ctx.forward.model.dof_map.fixed_values = ctx.solved_fixed_values
            solved_state: Model.State = _state_at(ctx.forward.model, ctx.solved_u)
            ctx.forward.__wrapped__.state = solved_state
            solution: LinearSolver.Solution | None = ctx.forward.adjoint_solve(
                grad_output, solved_state
            )
            p: Free = (
                torch.zeros_like(ctx.forward.model.dof_map.to_free_grad(grad_output))
                if solution is None
                else solution.params
            )
            p: Full = ctx.forward.model.dof_map.to_full_grad(p)
            ctx.forward.model.mixed_derivative_prod(solved_state, p)
            leaves: list[Tensor] = optree.tree_leaves(cast("Any", tmp_materials))
            grads: list[Tensor | None] = [leaf.grad for leaf in leaves]
            if any(
                grad is not None and not bool(torch.isfinite(grad).all())
                for grad in grads
            ):
                _raise_numerical(
                    "Implicit material pullback returned non-finite gradients"
                )
        finally:
            ctx.forward.__wrapped__.state = original_state
            ctx.forward.model.dof_map.fixed_values = original_fixed_values
            ctx.forward.model.set_materials(original_materials)
        return (None, None, *grads)


def _state_at(model: Model, u: Full) -> Model.State:
    """Build independent boundary and contact state at an immutable displacement."""
    state: Model.State = model.init()
    model.update(state, u)
    return state


def _raise_solve(message: str) -> Never:
    raise ImplicitSolveError(message)


def _raise_numerical(message: str) -> Never:
    raise ImplicitNumericalError(message)
