from __future__ import annotations

import abc
import logging
import operator
from collections.abc import Callable, Iterable, Mapping

import attrs
import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float
from liblaf.peach import tree
from liblaf.peach.constraints import Constraint
from liblaf.peach.linalg import (
    CompositeSolver,
    JaxCG,
    LinearSolver,
    LinearSystem,
    ScipyMinRes,
)
from liblaf.peach.optim import Callback, Objective, Optimizer, ScipyOptimizer

from liblaf import peach
from liblaf.apple.model import Forward, Model

from ._types import Aux, Params

type EnergyParams = Mapping[str, Array]
type Free = Float[Array, " free"]
type Full = Float[Array, "points dim"]
type ModelParams = Mapping[str, EnergyParams]
type Scalar = Float[Array, ""]


logger: logging.Logger = logging.getLogger(__name__)


@tree.define
class Inverse[ParamsT: Params, AuxT: Aux](abc.ABC):
    from ._types import Aux, Params

    def default_adjoint_solver(self, *, rtol: float = 1e-3) -> LinearSolver:
        cg_max_steps: int = max(1000, int(jnp.ceil(5 * jnp.sqrt(self.model.n_free))))
        minres_max_steps: int = max(
            1000, int(jnp.ceil(10 * jnp.sqrt(self.model.n_free)))
        )
        # return CompositeSolver(
        #     [
        #         JaxCG(max_steps=cg_max_steps, rtol=rtol),
        #         JaxBiCGStab(max_steps=minres_max_steps, rtol=rtol),
        #     ]
        # )
        if peach.cuda.is_available():
            from liblaf.peach.linalg import CupyMinRes

            return CompositeSolver(
                [
                    JaxCG(max_steps=cg_max_steps, rtol=rtol),
                    CupyMinRes(max_steps=minres_max_steps, rtol=rtol),
                ]
            )
        return CompositeSolver(
            [
                JaxCG(max_steps=cg_max_steps, rtol=rtol),
                ScipyMinRes(max_steps=minres_max_steps, rtol=rtol),
            ]
        )

    forward: Forward
    adjoint_solver: LinearSolver = tree.field(
        default=attrs.Factory(default_adjoint_solver, takes_self=True), kw_only=True
    )
    optimizer: Optimizer = tree.field(
        factory=lambda: ScipyOptimizer(method="L-BFGS-B", tol=1e-5), kw_only=True
    )

    adjoint_vector: Free = tree.array(default=None, init=False, kw_only=True)
    last_adjoint_success: Bool[Array, ""] = tree.array(
        default=False, init=False, kw_only=True
    )
    last_forward_success: Bool[Array, ""] = tree.array(
        default=False, init=False, kw_only=True
    )

    @property
    def model(self) -> Model:
        return self.forward.model

    def adjoint(self, u: Full, dLdu: Full) -> Full:
        solution: LinearSolver.Solution = self.adjoint_inner(u, dLdu)
        if not solution.success:
            logger.warning("Adjoint fail: %r", solution)
        logger.info("Adjoint Statistics: %r", solution.stats)
        return self.model.to_full(solution.params, 0.0)

    def adjoint_inner(self, u: Full, dLdu: Full) -> LinearSolver.Solution:
        u_free: Free = self.model.to_free(u)
        preconditioner: Free = jnp.reciprocal(self.model.hess_diag(u_free))

        @jax.custom_jvp
        def matvec(p_free: Free) -> Free:
            return self.model.hess_prod(u_free, p_free)

        @matvec.defjvp
        def matvec_jvp(
            primals: tuple[Free,], tangents: tuple[Free,]
        ) -> tuple[Free, Free]:
            (p_free,) = primals
            (dp_free,) = tangents
            Hp: Free = matvec(p_free)
            dHp: Free = matvec(dp_free)
            return Hp, dHp

        def preconditioner_fn(p_free: Free) -> Free:
            return preconditioner * p_free

        system = LinearSystem(
            matvec=matvec,
            b=-self.model.to_free(dLdu),
            rmatvec=matvec,
            preconditioner=preconditioner_fn,
            rpreconditioner=preconditioner_fn,
        )
        params: Free = (
            self.adjoint_vector if self.last_adjoint_success else jnp.zeros_like(u_free)
        )
        self.model.frozen = True  # make jax.jit happy
        solution: LinearSolver.Solution = self.adjoint_solver.solve(system, params)
        self.model.frozen = False
        self.adjoint_vector = solution.params
        self.last_adjoint_success = jnp.asarray(
            solution.success, self.last_adjoint_success.dtype
        )
        return solution

    def fun(self, params: ParamsT) -> tuple[Scalar, AuxT]:
        model_params: ModelParams = self.make_params(params)
        u_full: Full = self._forward(model_params)
        return self.loss(u_full, model_params)

    def grad(self, params: ParamsT) -> tuple[ParamsT, AuxT]:
        aux: AuxT
        grad: ParamsT
        _loss, grad, aux = self.value_and_grad(params)
        return grad, aux

    @abc.abstractmethod
    def loss(self, u: Full, params: ModelParams) -> tuple[Scalar, AuxT]:
        raise NotImplementedError

    @eqx.filter_jit
    def loss_and_grad(
        self, u: Full, params: ModelParams
    ) -> tuple[Scalar, Full, ModelParams, AuxT]:
        loss: Scalar
        aux: AuxT
        dLdu: Full
        dLdq: ModelParams
        (loss, aux), (dLdu, dLdq) = jax.value_and_grad(
            self.loss, argnums=(0, 1), has_aux=True
        )(u, params)
        return loss, dLdu, dLdq, aux

    @abc.abstractmethod
    def make_params(self, params: ParamsT) -> ModelParams:
        raise NotImplementedError

    def solve(
        self,
        params: ParamsT,
        *,
        constraints: Iterable[Constraint] = (),
        callback: Callback | None = None,
    ) -> Optimizer.Solution:
        objective = Objective(grad=self.grad, value_and_grad=self.value_and_grad)
        optimizer_solution: Optimizer.Solution = self.optimizer.minimize(
            objective, params, constraints=constraints, callback=callback
        )
        if not optimizer_solution.success:
            logger.warning("Inverse fail: %r", optimizer_solution)
        return optimizer_solution

    def value_and_grad(self, params: ParamsT) -> tuple[Scalar, ParamsT, AuxT]:
        model_params: ModelParams
        model_params_vjp: Callable[[ModelParams], tuple[ParamsT]]
        model_params, model_params_vjp = jax.vjp(self.make_params, params)
        u_full: Full = self._forward(model_params)
        loss: Scalar
        dLdu: Full
        dLdq: ModelParams
        aux: AuxT
        loss, dLdu, dLdq, aux = self.loss_and_grad(u_full, model_params)
        p: Full = self.adjoint(u_full, dLdu)
        prod: ModelParams = self.model.mixed_derivative_prod(u_full, p)
        model_params_grad: ModelParams = jax.tree.map(operator.add, dLdq, prod)
        grad: ParamsT
        (grad,) = model_params_vjp(model_params_grad)
        return loss, grad, aux

    def _forward(
        self, model_params: ModelParams, *, callback: Callback | None = None
    ) -> Full:
        solution: Optimizer.Solution = self._forward_inner(
            model_params, callback=callback
        )
        logger.info("Forward Statistics: %r", solution.stats)
        return self.model.u_full

    def _forward_inner(
        self, model_params: ModelParams, *, callback: Callback | None = None
    ) -> Optimizer.Solution:
        self.model.update_params(model_params)
        if not self.last_forward_success:
            self.model.u_free = jnp.zeros((self.model.n_free,))
        solution: Optimizer.Solution = self.forward.step(callback=callback)
        self.last_forward_success = jnp.asarray(
            solution.success, self.last_forward_success.dtype
        )
        return solution
