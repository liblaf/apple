import logging
from typing import Any, Mapping, cast, override

import attrs
import optree
import torch
from jaxtyping import Float
from liblaf.peach.linalg import FallbackSolver, LinearSolver
from liblaf.peach.optim import Optimizer
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from liblaf.apple.forward import Forward, Model

type Free = Float[Tensor, " free"]
type Full = Float[Tensor, "points dim"]

logger: logging.Logger = logging.getLogger(__name__)


@attrs.define
class DifferentiableForward:
    __wrapped__: Forward
    adjoint_solver: LinearSolver = attrs.field(factory=FallbackSolver)

    @property
    def model(self) -> Model:
        return self.__wrapped__.model

    @property
    def state(self) -> Model.State:
        return self.__wrapped__.state

    def adjoint_solve(self, u_grad: Full) -> LinearSolver.Solution:
        u_grad: Free = self.model.dof_map.to_free_grad(u_grad)
        problem: _AdjointProblem = _AdjointProblem(
            b=-u_grad, model=self.model, model_state=self.state
        )
        solution: LinearSolver.Solution = self.adjoint_solver.solve(problem, problem.b)
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

    def preconditioner(self, p_free: Free) -> Free:
        return self._preconditioner * p_free

    def rpreconditioner(self, p_free: Free) -> Free:
        return self.preconditioner(p_free)


class FunctionCtx(torch.autograd.function.FunctionCtx):
    needs_input_grad: tuple[bool, ...]
    saved_tensors: tuple[Tensor, ...]
    forward: DifferentiableForward
    spec: optree.PyTreeSpec


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
        forward.step()
        return forward.state.u

    @staticmethod
    @override
    def setup_context(
        ctx: FunctionCtx, inputs: tuple[Any, ...], output: Tensor
    ) -> None:
        forward, spec, *args = inputs
        ctx.forward = forward
        ctx.spec = spec
        ctx.save_for_backward(*args)

    @staticmethod
    @once_differentiable
    @override
    def backward(ctx: FunctionCtx, grad_output: Tensor) -> tuple[Tensor | None, ...]:
        original_materials: dict[str, dict[str, Tensor]] = (
            ctx.forward.model.get_materials()
        )
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
            solution: LinearSolver.Solution = ctx.forward.adjoint_solve(grad_output)
            p: Free = solution.params
            p: Full = ctx.forward.model.dof_map.to_full_grad(p)
            result_materials: dict[str, dict[str, Tensor]] = (
                ctx.forward.model.get_materials()
            )
            leaves: list[Tensor] = optree.tree_leaves(cast("Any", result_materials))
            ctx.forward.model.mixed_derivative_prod(ctx.forward.state, p)
            grads: list[Tensor | None] = [leaf.grad for leaf in leaves]
        finally:
            ctx.forward.model.set_materials(original_materials)
        return (None, None, *grads)
