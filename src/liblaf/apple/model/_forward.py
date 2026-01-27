from __future__ import annotations

import logging
from collections.abc import Mapping

import attrs
import jarp
import jax.numpy as jnp
from jaxtyping import Array, Float
from liblaf.peach.optim import PNCG, Objective, Optimizer
from liblaf.peach.transforms import FixedTransform

from ._model import Model, ModelState

logger: logging.Logger = logging.getLogger(__name__)

type EnergyMaterials = Mapping[str, Array]
type Free = Float[Array, " free"]
type Full = Float[Array, " full"]
type ModelMaterials = Mapping[str, EnergyMaterials]
type Scalar = Float[Array, ""]


@jarp.define
class Forward:
    model: Model

    def _default_state(self) -> ModelState:
        return self.model.init_state(self.model.u_full)

    state: ModelState = jarp.field(
        default=attrs.Factory(_default_state, takes_self=True), kw_only=True
    )

    def _default_optimizer(self: Forward) -> Optimizer:
        max_steps: int = max(1000, jnp.ceil(20 * jnp.sqrt(self.model.n_free)).item())
        max_delta: Scalar = (
            0.15 * self.model.edges_length_mean
            if self.model.edges_length_mean > 0
            else jnp.asarray(jnp.inf)
        )
        return PNCG(
            max_steps=jnp.asarray(max_steps),
            atol=jnp.asarray(1e-10),
            rtol=jnp.asarray(1e-3),
            atol_primary=jnp.asarray(1e-10),
            rtol_primary=jnp.asarray(1e-5),
            beta_non_negative=True,
            beta_reset_threshold=jnp.asarray(10.0),
            max_delta=max_delta,
            stagnation_max_restarts=jnp.asarray(20),
            stagnation_patience=jnp.asarray(50),
        )

    optimizer: Optimizer = jarp.field(
        default=attrs.Factory(_default_optimizer, takes_self=True), kw_only=True
    )

    @property
    def u_full(self) -> Float[Array, "points dim"]:
        return self.model.u_full

    def update_materials(self, materials: ModelMaterials) -> None:
        self.model.update_materials(materials)

    def step(
        self, callback: Optimizer.Callback | None = None, *, logging: bool = True
    ) -> Optimizer.Solution:
        objective: Objective[ModelState, Full] = Objective(
            update=Model.update,
            fun=Model.fun,
            grad=Model.grad,
            hess_diag=Model.hess_diag,
            hess_prod=Model.hess_prod,
            hess_quad=Model.hess_quad,
            args=(self.model,),
            transform=FixedTransform(
                self.model.dirichlet.fixed_mask, self.model.u_full
            ),
        )
        solution: Optimizer.Solution
        solution, self.state = self.optimizer.minimize(
            objective, self.state, self.model.u_full, callback=callback
        )
        if logging:
            if solution.success:
                logger.info("Forward success: %r", solution.stats)
            else:
                logger.warning("Forward fail: %r", solution)
        self.model.u_free = solution.params
        return solution
