from jaxtyping import Array, Float
from liblaf.peach import tree
from liblaf.peach.optim import PNCG, Callback, Objective, Optimizer

from ._model import Model

type Free = Float[Array, " free"]


@tree.define
class Forward:
    model: Model
    optimizer: Optimizer = tree.field(factory=PNCG)

    def step(self, callback: Callback | None = None) -> Optimizer.Solution:
        objective = Objective(
            fun=self.model.fun,
            grad=self.model.grad,
            hess_diag=self.model.hess_diag,
            hess_prod=self.model.hess_prod,
            hess_quad=self.model.hess_quad,
            value_and_grad=self.model.value_and_grad,
            grad_and_hess_diag=self.model.grad_and_hess_diag,
        )
        u_free: Free = self.model.to_free(self.model.u_full)
        solution: Optimizer.Solution = self.optimizer.minimize(
            objective, u_free, callback=callback
        )
        self.model.update(solution.params)
        return solution
