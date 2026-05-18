import functools

import attrs
import jax.numpy as jnp
from jaxtyping import Array, Float, Integer
from liblaf.peach.optim import Optimizer

from liblaf import jarp

from ._model import Model
from ._problem import ForwardProblem
from ._state import ModelState

type Free = Float[Array, " free"]
type Full = Float[Array, "points dim"]


@jarp.define
class Forward:
    model: Model

    def _default_optimizer(self) -> Optimizer:
        from liblaf.peach.optim import PNCG
        from liblaf.peach.optim.pncg import ConvergenceCriteria, LineSearch

        max_steps: Integer[Array, ""] = jnp.asarray(1500)
        criteria = ConvergenceCriteria(max_steps=max_steps)
        # TODO: add max_step_norm bound from edge lengths
        line_search = LineSearch()
        return PNCG(criteria=criteria, line_search=line_search)

    def _default_state(self) -> ModelState:
        u_free: Free = jnp.zeros((self.model.n_free,))
        u_full: Full = self.model.dof_map.to_full(u_free)
        return ModelState(u=u_full)

    optimizer: Optimizer = jarp.field(
        default=attrs.Factory(_default_optimizer, takes_self=True)
    )
    state: ModelState = attrs.field(
        default=attrs.Factory(_default_state, takes_self=True)
    )

    @property
    def free(self) -> Free:
        return self.model.dof_map.to_free(self.state.u)

    @functools.cached_property
    def problem(self) -> ForwardProblem:
        return ForwardProblem(model=self.model)

    def step(self) -> Optimizer.Solution:
        solution, self.state = self.optimizer.minimize(
            self.problem, self.state, self.free
        )
        return solution
