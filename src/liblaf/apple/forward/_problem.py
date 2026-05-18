from typing import override

import attrs
import jax.numpy as jnp
from jaxtyping import Array, Float
from liblaf.peach.optim import Problem

from liblaf import jarp

from ._model import Model
from ._state import ModelState

type Free = Float[Array, " free"]
type Full = Float[Array, "points dim"]
type Scalar = Float[Array, ""]


@jarp.define
class ForwardProblem(Problem[ModelState]):
    from ._state import ModelState as State

    model: Model

    @override
    def before_trial(self, state: State, u: Free) -> State:
        u_full: Full = self.model.dof_map.to_full(u)
        return attrs.evolve(state, u=u_full)

    @override
    def max_step_size(self, state: State, p: Free) -> Scalar:
        if self.model.collision is None:
            return jnp.ones(())
        p_full: Full = self.model.dof_map.to_full_grad(p)
        return self.model.collision.max_step_size(state.u, p_full)

    @override
    def fun(self, state: State) -> Scalar:
        return self.model.fun(state.u)

    @override
    def grad(self, state: State) -> Free:
        grad_full: Full = self.model.grad(state.u)
        return self.model.dof_map.to_free_grad(grad_full)

    @override
    def hess_diag(self, state: State) -> Free:
        hess_diag_full: Full = self.model.hess_diag(state.u)
        return self.model.dof_map.to_free_hess_diag(hess_diag_full)

    @override
    def hess_prod(self, state: State, p: Free) -> Free:
        p_full: Full = self.model.dof_map.to_full_grad(p)
        hess_prod_full: Full = self.model.hess_prod(state.u, p_full)
        return self.model.dof_map.to_free_grad(hess_prod_full)

    @override
    def hess_quad(self, state: State, p: Free) -> Scalar:
        p_full: Full = self.model.dof_map.to_full_grad(p)
        return self.model.hess_quad(state.u, p_full)
