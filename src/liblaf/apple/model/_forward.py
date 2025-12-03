import logging
from collections.abc import Mapping

from jaxtyping import Array, Float
from liblaf.peach import tree
from liblaf.peach.optim import PNCG, Callback, Objective, Optimizer

from ._model import Model

logger = logging.getLogger(__name__)

type Free = Float[Array, " free"]
type EnergyParams = Mapping[str, Array]
type ModelParams = Mapping[str, EnergyParams]


@tree.define
class Forward:
    model: Model
    optimizer: Optimizer = tree.field(factory=lambda: PNCG(max_steps=1000))

    @property
    def u_full(self) -> Float[Array, "points dim"]:
        return self.model.u_full

    def update_params(self, params: ModelParams) -> None:
        self.model.update_params(params)

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
        solution: Optimizer.Solution = self.optimizer.minimize(
            objective, self.model.u_free, callback=callback
        )
        if not solution.success:
            logger.warning("Forward fail: %r", solution)
        self.model.u_free = solution.params
        return solution
