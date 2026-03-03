from typing import Self

import jarp
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float
from liblaf.peach.linalg import LinearSolver
from liblaf.peach.optim import Optimizer

from liblaf.apple.model import Forward, Free, Full, Model, ModelMaterials, ModelState

from .loss import Loss

type BoolNumeric = Bool[Array, ""]
type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]


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
        return self.model.hess_prod(self.model_state, p_free)

    def rmatvec(self, p_free: Free) -> Free:
        return self.matvec(p_free)

    def preconditioner(self, p_free: Free) -> Free:
        return self._preconditioner * p_free

    def rpreconditioner(self, p_free: Free) -> Free:
        return self.preconditioner(p_free)


class InverseObjective:
    def value_and_grad(self, params: ModelMaterials) -> tuple[Scalar, ModelMaterials]:
        raise NotImplementedError

    def make_params(self, params: Vector) -> ModelMaterials:
        raise NotImplementedError


@jarp.define
class Inverse:
    forward: Forward
    losses: list[Loss]
    adjoint_solver: LinearSolver
    optimizer: Optimizer

    adjoint_vector: Free = jarp.array(default=None)
    last_adjoint_success: BoolNumeric = jarp.array(default=False)
    last_forward_success: BoolNumeric = jarp.array(default=False)

    @property
    def model(self) -> Model:
        return self.forward.model

    def update(self, materials: ModelMaterials) -> None:
        self.model.update_materials(materials)
        if not self.last_forward_success:
            self.model.u_free = jnp.zeros_like(self.model.u_free)
        solution: Optimizer.Solution = self.forward.step()
        self.last_forward_success = jnp.asarray(solution.success)

    @jarp.jit(inline=True)
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
        p: Free = self.adjoint(dLdu)
        mixed_prod: ModelMaterials = self.model.mixed_derivative_prod(
            self.forward.state, p
        )
        for energy_id, energy in mixed_prod.items():
            for mat_name, v in energy.items():
                dLdq[energy_id][mat_name] += v
        return L, dLdq

    @jarp.jit(inline=True)
    def loss_and_grad(
        self, materials: ModelMaterials
    ) -> tuple[Scalar, Full, dict[str, dict[str, Array]]]:
        L: Scalar = self.fun(materials)
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
            L += L_i
            dLdu += dLdu_i
            for energy_id, energy in dLdq_i.items():
                for mat_name, v in energy.items():
                    dLdq[energy_id][mat_name] += v
        return L, dLdu, dLdq

    def adjoint(self, dLdu: Full) -> Full:
        system = AdjointLinearSystem.new(self.model, self.forward.state, dLdu)
        p_free: Free = (
            self.adjoint_vector
            if self.last_adjoint_success
            else jnp.zeros_like(self.model.u_free)
        )
        solution: LinearSolver.Solution = self.adjoint_solver.solve(system, p_free)
        self.last_adjoint_success = jnp.asarray(solution.success)
        self.adjoint_vector = solution.params
        return self.model.dirichlet.to_full(self.adjoint_vector, dirichlet=0.0)
