from typing import override

import attrs
import torch
from jaxtyping import Float
from liblaf.peach.optim import Problem
from torch import Tensor

from liblaf.apple.torch.utils import method_with_device

from ._model import Model

type Free = Float[Tensor, " free"]
type Full = Float[Tensor, "points dim"]
type Scalar = Float[Tensor, ""]


@attrs.define
class ForwardProblem(Problem[Model.State]):
    type State = Model.State

    model: Model

    @property
    def device(self) -> torch.device:
        return self.model.device

    @method_with_device
    @override
    def max_step_size(self, state: State, p: Free) -> Scalar:
        p_full: Full = self.model.dof_map.to_full_grad(p)
        return self.model.max_step_size(state, p_full)

    @method_with_device
    @override
    def update(self, state: State, u: Free) -> None:
        u_full: Full = self.model.dof_map.to_full(u)
        self.model.update(state, u_full)

    @method_with_device
    @override
    def fun(self, state: State) -> Scalar:
        return self.model.fun(state)

    @method_with_device
    @override
    def grad(self, state: State) -> Free:
        grad_full: Full = self.model.grad(state)
        return self.model.dof_map.to_free_grad(grad_full)

    @method_with_device
    @override
    def hess_diag(self, state: State) -> Free:
        H_diag_full: Full = self.model.hess_diag(state)
        return self.model.dof_map.to_free_hess_diag(H_diag_full)

    @method_with_device
    @override
    def hess_prod(self, state: State, p: Free) -> Free:
        p_full: Full = self.model.dof_map.to_full_grad(p)
        Hp_full: Full = self.model.hess_prod(state, p_full)
        return self.model.dof_map.to_free_grad(Hp_full)

    @method_with_device
    @override
    def hess_quad(self, state: State, p: Free) -> Scalar:
        p_full: Full = self.model.dof_map.to_full_grad(p)
        return self.model.hess_quad(state, p_full)
