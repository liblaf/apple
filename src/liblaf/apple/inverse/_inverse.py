from __future__ import annotations

import logging
from typing import Self, override

import jarp
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Bool, Float
from liblaf.peach.linalg import (
    CupyMinRes,
    FallbackSolver,
    JaxCG,
    LinearSolver,
    LinearSystem,
    Result,
)
from liblaf.peach.linalg import utils as linalg_utils
from liblaf.peach.optim import PNCG, Optax, Optimizer

from liblaf import cherries
from liblaf.apple.model import Forward, Free, Full, Model, ModelMaterials, ModelState

from .loss import Loss

type BoolNumeric = Bool[Array, ""]
type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]

logger: logging.Logger = logging.getLogger(__name__)


@jarp.define
class AdjointLinearSystem:
    b: Free
    model: Model
    model_state: ModelState
    _preconditioner: Free = jarp.field(alias="preconditioner")

    @classmethod
    def new(cls, model: Model, model_state: ModelState, dLdu: Full) -> Self:
        b: Free = -model.dirichlet.get_free(dLdu)
        h_diag_full: Full = model.hess_diag(model_state)
        h_diag_free: Free = model.dirichlet.get_free(h_diag_full)
        preconditioner: Free = jnp.reciprocal(h_diag_free)
        return cls(
            b=b, model=model, model_state=model_state, preconditioner=preconditioner
        )

    def matvec(self, p_free: Free) -> Free:
        p_full: Full = self.model.dirichlet.to_full(p_free, dirichlet=0.0)
        output_full: Full = self.model.hess_prod(self.model_state, p_full)
        output_free: Free = self.model.dirichlet.get_free(output_full)
        return output_free

    def rmatvec(self, p_free: Free) -> Free:
        return self.matvec(p_free)

    def preconditioner(self, p_free: Free) -> Free:
        return self._preconditioner * p_free

    def rpreconditioner(self, p_free: Free) -> Free:
        return self.preconditioner(p_free)


@jarp.define
class InverseObjective[T]:
    inverse: Inverse[T]
    structure: jarp.Structure[T]

    def update(self, _state: Vector, params_flat: Vector) -> Vector:
        return params_flat

    def value_and_grad(self, params_flat: Vector) -> tuple[Scalar, Vector]:
        params_tree: T = self.structure.unravel(params_flat)
        materials: ModelMaterials
        materials, vjp = jax.vjp(self.inverse.make_materials, params_tree)
        self.inverse.update(materials)
        value, grad = self.inverse.value_and_grad(materials)
        grad_tree: T
        (grad_tree,) = vjp(grad)
        grad_flat: Vector = self.structure.ravel(grad_tree)
        return value, grad_flat


@jarp.define
class AdjointSolver(FallbackSolver):
    @staticmethod
    def _default_solvers() -> list[LinearSolver]:
        return [
            JaxCG(rtol=jnp.asarray(1e-3), rtol_primary=jnp.asarray(1e-6)),
            CupyMinRes(tol=1e-6),
        ]

    solvers: list[LinearSolver] = jarp.field(factory=_default_solvers)

    @override
    def compute(
        self,
        system: LinearSystem,
        state: FallbackSolver.State,
        stats: FallbackSolver.Stats,
    ) -> tuple[FallbackSolver.State, FallbackSolver.Stats, Result]:
        state, stats, result = super().compute(system, state, stats)
        min_residual: Scalar = jnp.asarray(jnp.inf)
        for state_i in state.state:
            if state_i.params is not None:
                residual: Scalar = linalg_utils.absolute_residual(
                    system.matvec, state_i.params, system.b
                )
                if residual < min_residual:
                    state.params = state_i.params
                    min_residual = residual
        return state, stats, result


@jarp.define
class Inverse[T]:
    forward: Forward
    losses: list[Loss]
    adjoint_solver: LinearSolver = jarp.field(factory=AdjointSolver, kw_only=True)
    optimizer: Optimizer = jarp.field(
        factory=lambda: Optax(optax.sgd(0.1), patience=jnp.asarray(1000)),
        kw_only=True,
    )

    adjoint_vector: Free = jarp.array(default=None, kw_only=True)
    last_adjoint_success: BoolNumeric = jarp.array(default=False, kw_only=True)
    last_forward_success: BoolNumeric = jarp.array(default=False, kw_only=True)

    @property
    def model(self) -> Model:
        return self.forward.model

    def update(self, materials: ModelMaterials) -> None:
        self.model.update_materials(materials)
        if not self.last_forward_success:
            self.model.u_free = jnp.zeros_like(self.model.u_free)
        # self.model.u_free = jnp.zeros_like(self.model.u_free)
        solution: PNCG.Solution = self.forward.step()
        cherries.log_metrics(
            {
                "forward": {
                    "decrease": solution.state.best_decrease,
                    "relative_decrease": solution.stats.relative_decrease,
                }
            }
        )
        self.last_forward_success = jnp.asarray(solution.success)

    @jarp.jit(filter=True, inline=True)
    def fun(self, materials: ModelMaterials) -> Scalar:
        losses = [loss.fun(self.model.u_full, materials) for loss in self.losses]
        return jnp.sum(jnp.stack(losses))

    def value_and_grad(
        self, materials: ModelMaterials
    ) -> tuple[Scalar, ModelMaterials]:
        L: Scalar
        dLdu: Full
        dLdq: dict[str, dict[str, Array]]
        L, dLdu, dLdq = self.loss_and_grad(materials)
        jax.debug.print("L: {L}", L=L)
        p: Free = self.adjoint(dLdu)
        # jax.debug.print("p: {p}", p=p)
        ic(jnp.count_nonzero(dLdq["muscle"]["activation"]))
        mixed_prod: ModelMaterials = self.model.mixed_derivative_prod(
            self.forward.state, p
        )
        for energy_id, energy in mixed_prod.items():
            for mat_name, v in energy.items():
                # jax.debug.print(
                #     "dLdq['{energy_id}']['{mat_name}']: {v}",
                #     energy_id=energy_id,
                #     mat_name=mat_name,
                #     v=v,
                # )
                dLdq[energy_id][mat_name] += v
        return L, dLdq

    @jarp.jit(filter=True, inline=True)
    def loss_and_grad(
        self, materials: ModelMaterials
    ) -> tuple[Scalar, Full, dict[str, dict[str, Array]]]:
        L: Scalar = jnp.zeros(())
        dLdu: Full = jnp.zeros_like(self.model.u_full)
        dLdq: dict[str, dict[str, Array]] = {
            energy_id: {
                mat_name: jnp.zeros_like(mat_value)
                for mat_name, mat_value in energy.items()
            }
            for energy_id, energy in materials.items()
        }
        for loss in self.losses:
            L_i, (dLdu_i, dLdq_i) = loss.value_and_grad(self.model.u_full, materials)
            jax.debug.callback(
                lambda name, value: cherries.log_metric(name, value), loss.name, L_i
            )
            L += L_i
            dLdu += dLdu_i
            dLdq = jax.tree.map(jnp.add, dLdq, dLdq_i)
        jax.debug.callback(lambda value: cherries.log_metric("loss", value), L)
        return L, dLdu, dLdq

    def adjoint(self, dLdu: Full) -> Full:
        system = AdjointLinearSystem.new(self.model, self.forward.state, dLdu)
        p_free: Free = (
            self.adjoint_vector
            if self.last_adjoint_success
            else jnp.zeros_like(self.model.u_free)
        )
        # p_free: Free = jnp.zeros_like(self.model.u_free)
        solution: LinearSolver.Solution = self.adjoint_solver.solve(system, p_free)
        ic(solution)
        cherries.log_metrics(
            {
                "adjoint": {
                    "success": solution.success,
                    "relative_residual": linalg_utils.relative_residual(
                        system.matvec, solution.params, system.b
                    ),
                }
            }
        )
        if solution.success:
            logger.info("Adjoint success")
        else:
            logger.warning("Adjoint failed")
        self.last_adjoint_success = jnp.asarray(solution.success)
        self.adjoint_vector = solution.params
        return self.model.dirichlet.to_full(self.adjoint_vector, dirichlet=0.0)

    def make_materials(self, params: T) -> ModelMaterials:
        return params  # pyright: ignore[reportReturnType]

    def solve(self, params: T, callback: Optimizer.Callback | None = None) -> T:
        params_flat: Vector
        structure: jarp.Structure[T]
        params_flat, structure = jarp.ravel(params)
        objective = InverseObjective(inverse=self, structure=structure)
        solution: Optimizer.Solution
        solution, _ = self.optimizer.minimize(
            objective, params, params_flat, callback=callback
        )
        ic(solution)
        params_tree: T = structure.unravel(solution.params)
        return params_tree
