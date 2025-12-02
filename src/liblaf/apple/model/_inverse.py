import abc
import logging
import operator
from collections.abc import Callable, Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from liblaf.peach import tree
from liblaf.peach.linalg import JaxCompositeSolver, LinearSolver, LinearSystem
from liblaf.peach.optim import Optimizer

from ._forward import Forward
from ._model import Model

type EnergyParams = Mapping[str, Array]
type Free = Float[Array, " free"]
type Full = Float[Array, "points dim"]
type ModelParams = Mapping[str, EnergyParams]
type Scalar = Float[Array, ""]


logger: logging.Logger = logging.getLogger(__name__)


@tree.define
class Inverse(abc.ABC):
    @tree.define
    class Aux:
        pass

    @tree.define
    class Params:
        pass

    forward: Forward
    adjoint_solver: LinearSolver = tree.field(factory=JaxCompositeSolver)

    @property
    def model(self) -> Model:
        return self.forward.model

    def fun(self, params: Params) -> tuple[Scalar, Aux]:
        model_params: ModelParams = self.make_params(params)
        self.model.update_params(model_params)
        solution: Optimizer.Solution = self.forward.step()
        ic(solution)
        return self.loss(model_params, self.model.u_full)

    def value_and_grad(self, params: Params) -> tuple[Scalar, Params]:
        model_params: ModelParams
        model_params_vjp: Callable[[ModelParams], Inverse.Params]
        model_params, model_params_vjp = jax.vjp(self.make_params, params)
        self.model.update_params(model_params)
        solution: Optimizer.Solution = self.forward.step()
        ic(solution)
        u_full: Full = self.model.u_full
        loss, dLdq, dLdu, _aux = self.loss_and_grad(model_params, u_full)
        p: Full = self.adjoint(u_full, dLdu)
        prod: ModelParams = self.model.mixed_derivative_prod(u_full, p)
        model_params_grad: ModelParams = jax.tree.map(operator.add, dLdq, prod)
        grad: Inverse.Params = model_params_vjp(model_params_grad)
        return loss, grad

    def adjoint(self, u: Full, dLdu: Full) -> Full:
        u_free: Free = self.model.to_free(u)
        preconditioner: Free = jnp.reciprocal(self.model.hess_diag(u_free))
        system = LinearSystem(
            lambda p_free: self.model.hess_prod(u_free, p_free),
            b=-self.model.to_free(dLdu),
            preconditioner=lambda p_free: preconditioner * p_free,
        )
        solution: LinearSolver.Solution = self.adjoint_solver.solve(
            system, jnp.zeros_like(u_free)
        )
        ic(solution)
        return self.model.to_full(solution.params, 0.0)

    @abc.abstractmethod
    def loss(self, params: ModelParams, u: Full) -> tuple[Scalar, Aux]:
        raise NotImplementedError

    @eqx.filter_jit
    def loss_and_grad(
        self, params: ModelParams, u: Full
    ) -> tuple[Scalar, ModelParams, Full, Aux]:
        loss: Scalar
        aux: Inverse.Aux
        dLdu: Full
        dLdq: ModelParams
        (loss, aux), (dLdq, dLdu) = jax.value_and_grad(
            self.loss, argnums=(0, 1), has_aux=True
        )(params, u)
        return loss, dLdq, dLdu, aux

    @abc.abstractmethod
    def make_params(self, params: Params) -> ModelParams:
        raise NotImplementedError
