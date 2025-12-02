import abc
import logging
import operator
from collections.abc import Callable, Mapping

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from liblaf.peach import tree
from liblaf.peach.linalg import CompositeSolver, LinearSolver, LinearSystem
from liblaf.peach.optim import Callback, Objective, Optimizer, ScipyOptimizer

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
    adjoint_solver: LinearSolver = tree.field(factory=CompositeSolver, kw_only=True)
    optimizer: Optimizer = tree.field(
        factory=lambda: ScipyOptimizer(method="L-BFGS-B"), kw_only=True
    )

    @property
    def model(self) -> Model:
        return self.forward.model

    def adjoint(self, u: Full, dLdu: Full) -> Full:
        u_free: Free = self.model.to_free(u)
        preconditioner: Free = jnp.reciprocal(self.model.hess_diag(u_free))

        def matvec(p_free: Free) -> Free:
            return self.model.hess_prod(u_free, p_free)

        def preconditioner_fn(p_free: Free) -> Free:
            return preconditioner * p_free

        system = LinearSystem(
            matvec=matvec,
            b=-self.model.to_free(dLdu),
            rmatvec=matvec,
            preconditioner=preconditioner_fn,
            rpreconditioner=preconditioner_fn,
        )
        solution: LinearSolver.Solution = self.adjoint_solver.solve(
            system, jnp.zeros_like(u_free)
        )
        if not solution.success:
            logger.warning("Adjoint fail: %r", solution)
        return self.model.to_full(solution.params, 0.0)

    def fun(self, params: Params) -> tuple[Scalar, Aux]:
        model_params: ModelParams = self.make_params(params)
        self.model.update_params(model_params)
        solution: Optimizer.Solution = self.forward.step()
        if not solution.success:
            logger.warning("Forward fail: %r", solution)
        return self.loss(self.model.u_full, model_params)

    @abc.abstractmethod
    def loss(self, u: Full, params: ModelParams) -> tuple[Scalar, Aux]:
        raise NotImplementedError

    @eqx.filter_jit
    def loss_and_grad(
        self, u: Full, params: ModelParams
    ) -> tuple[Scalar, Full, ModelParams, Aux]:
        loss: Scalar
        aux: Inverse.Aux
        dLdu: Full
        dLdq: ModelParams
        (loss, aux), (dLdu, dLdq) = jax.value_and_grad(
            self.loss, argnums=(0, 1), has_aux=True
        )(u, params)
        return loss, dLdu, dLdq, aux

    @abc.abstractmethod
    def make_params(self, params: Params) -> ModelParams:
        raise NotImplementedError

    def solve(
        self, params: Params, callback: Callback | None = None
    ) -> Optimizer.Solution:
        objective = Objective(value_and_grad=self.value_and_grad)
        optimizer_solution: Optimizer.Solution = self.optimizer.minimize(
            objective, params, callback=callback
        )
        if not optimizer_solution.success:
            logger.warning("Inverse fail: %r", optimizer_solution)
        return optimizer_solution

    def value_and_grad(self, params: Params) -> tuple[Scalar, Params]:
        model_params: ModelParams
        model_params_vjp: Callable[[ModelParams], Inverse.Params]
        model_params, model_params_vjp = jax.vjp(self.make_params, params)
        self.model.update_params(model_params)
        solution: Optimizer.Solution = self.forward.step()
        if not solution.success:
            logger.warning("Forward fail: %r", solution)
        u_full: Full = self.model.u_full
        loss, dLdu, dLdq, _aux = self.loss_and_grad(u_full, model_params)
        p: Full = self.adjoint(u_full, dLdu)
        prod: ModelParams = self.model.mixed_derivative_prod(u_full, p)
        model_params_grad: ModelParams = jax.tree.map(operator.add, dLdq, prod)
        grad: Inverse.Params = model_params_vjp(model_params_grad)
        return loss, grad
