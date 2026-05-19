import functools
import logging

import attrs
import torch
from jaxtyping import Float
from liblaf.peach.optim import Optimizer
from torch import Tensor

from ._model import Model
from ._problem import ForwardProblem

type Free = Float[Tensor, " free"]
type Full = Float[Tensor, "points dim"]

logger: logging.Logger = logging.getLogger(__name__)


@attrs.define
class Forward:
    def _default_optimizer(self) -> Optimizer:
        from liblaf.peach.optim import Pncg

        max_step_norm: float = torch.inf
        if self.model.collision is not None:
            max_step_norm: float = 0.5 * self.model.collision.potential.dhat
        criteria: Pncg.ConvergenceCriteria = Pncg.ConvergenceCriteria(max_steps=1500)
        line_search: Pncg.LineSearch = Pncg.LineSearch(max_step_norm=max_step_norm)
        return Pncg(criteria=criteria, line_search=line_search)

    def _default_state(self) -> Model.State:
        return self.model.init()

    model: Model
    optimizer: Optimizer = attrs.field(
        default=attrs.Factory(_default_optimizer, takes_self=True)
    )
    state: Model.State = attrs.field(
        default=attrs.Factory(_default_state, takes_self=True)
    )

    @property
    def free(self) -> Free:
        return self.model.dof_map.to_free(self.state.u)

    @functools.cached_property
    def problem(self) -> ForwardProblem:
        return ForwardProblem(model=self.model)

    def step(self) -> Optimizer.Solution:
        solution: Optimizer.Solution = self.optimizer.minimize(
            self.problem, self.state, self.free
        )
        logger.info(solution)
        return solution
