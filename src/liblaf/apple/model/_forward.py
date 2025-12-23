from __future__ import annotations

import logging
from collections.abc import Mapping

import attrs
import jax.numpy as jnp
from jaxtyping import Array, Float
from liblaf.peach import tree
from liblaf.peach.optim import PNCG, Callback, Objective, Optimizer

from ._model import Model

logger: logging.Logger = logging.getLogger(__name__)

type EnergyParams = Mapping[str, Array]
type Free = Float[Array, " free"]
type ModelParams = Mapping[str, EnergyParams]
type Scalar = Float[Array, ""]


def _default_optimizer(self: Forward) -> Optimizer:
    max_steps: int = max(1000, jnp.ceil(20 * jnp.sqrt(self.model.n_free)).item())
    max_delta: Scalar = (
        0.15 * self.model.edges_length_mean
        if self.model.edges_length_mean > 0
        else jnp.asarray(jnp.inf)
    )
    return PNCG(
        max_steps=max_steps,
        atol=1e-10,
        rtol=1e-5,
        beta_non_negative=True,
        beta_restart_threshold=2.0,
        max_delta=max_delta,
        timer=True,
    )


@tree.define
class Forward:
    model: Model
    optimizer: Optimizer = tree.field(
        default=attrs.Factory(_default_optimizer, takes_self=True), kw_only=True
    )

    @property
    def u_full(self) -> Float[Array, "points dim"]:
        return self.model.u_full

    def update_params(self, params: ModelParams) -> None:
        self.model.update_params(params)

    def step(
        self, callback: Callback | None = None, *, logging: bool = True
    ) -> Optimizer.Solution:
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
        if logging:
            if solution.success:
                logger.info("Forward success: %r", solution.stats)
            else:
                logger.warning("Forward fail: %r", solution)
        self.model.u_free = solution.params
        return solution
