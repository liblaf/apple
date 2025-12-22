import collections
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import override

import equinox as eqx
import jax.numpy as jnp
import pyvista as pv
from jaxtyping import Array, Float, Integer
from liblaf.peach import tree
from liblaf.peach.constraints import Constraint
from liblaf.peach.optim import PNCG as OrigPNCG
from liblaf.peach.optim.abc._types import OptimizeSolution, Result
from liblaf.peach.optim.objective import Objective
from liblaf.peach.optim.pncg._state import PNCGState
from liblaf.peach.optim.pncg._stats import PNCGStats

from liblaf import cherries, grapes, melon
from liblaf.apple import Forward, Model, ModelBuilder
from liblaf.apple.constants import POINT_ID
from liblaf.apple.warp import Phace

type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]

logger: logging.Logger = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    input: Path = cherries.temp(
        "20-inverse-adam-123k.vtu.d/20-inverse-adam-123k_000087.vtu"
    )


def build_model(mesh: pv.UnstructuredGrid) -> Model:
    builder = ModelBuilder()
    mesh = builder.assign_global_ids(mesh)
    builder.add_dirichlet(mesh)
    builder.add_energy(Phace.from_pyvista(mesh, id="elastic", clamp_lambda=False))
    model: Model = builder.finalize()
    return model


@tree.define
class PNCG(OrigPNCG):
    @tree.define
    class State(OrigPNCG.State):
        delta_x_history: collections.deque[Vector] = tree.field(
            factory=lambda: collections.deque(maxlen=50)
        )
        total_path_length: Scalar = tree.array(default=0.0)
        net_displacement: Scalar = tree.array(default=0.0)
        path_efficiency: Scalar = tree.array(default=0.0)
        grad_norm_min: Scalar = tree.array(default=jnp.inf)
        stagnation_count: Integer[Array, ""] = tree.array(default=0)

    @eqx.filter_jit
    def _compute_beta(self, g_prev: Vector, g: Vector, p: Vector, P: Vector) -> Scalar:
        y: Vector = g - g_prev
        yTp: Scalar = jnp.vdot(y, p)
        Py: Scalar = P * y
        beta: Scalar = jnp.vdot(g, Py) / yTp - (jnp.vdot(y, Py) / yTp) * (
            jnp.vdot(p, g) / yTp
        )
        beta = jnp.nan_to_num(beta, nan=0.0)
        beta = jnp.where(self.clamp_beta, jnp.maximum(beta, 0.0), beta)
        beta = jnp.where(beta > 2.0, 0.0, beta)
        return beta

    @override
    def step(
        self,
        objective: Objective,
        state: State,
        *,
        constraints: Iterable[Constraint] = (),
    ) -> State:
        if constraints:
            raise NotImplementedError
        assert objective.grad_and_hess_diag is not None
        assert objective.hess_quad is not None
        g: Vector
        H_diag: Vector
        g, H_diag = objective.grad_and_hess_diag(state.params_flat)
        H_diag = jnp.where(H_diag <= 0.0, 1.0, H_diag)
        P: Vector = jnp.reciprocal(H_diag)
        beta: Scalar
        p: Vector
        if state.search_direction_flat is None:
            beta = jnp.zeros(())
            p = -P * g
        else:
            grad_norm = jnp.linalg.norm(state.grad_flat)
            state.grad_norm_min = jnp.minimum(state.grad_norm_min, grad_norm)
            state.stagnation_count = jnp.where(
                grad_norm < state.grad_norm_min, 0, state.stagnation_count + 1
            )
            if state.stagnation_count > 20:
                logger.warning(
                    "Stagnation detected: gradient norm has not decreased for 20 steps."
                )
                beta = jnp.zeros(())
            else:
                beta = self._compute_beta(
                    g_prev=state.grad_flat, g=g, p=state.search_direction_flat, P=P
                )
            p = -P * g + beta * state.search_direction_flat
        pHp: Scalar = objective.hess_quad(state.params_flat, p)
        alpha: Scalar = self.line_search.search(objective, state.params_flat, g, p)
        # alpha *= 0.5
        state.params_flat += alpha * p
        DeltaE: Scalar = -alpha * jnp.vdot(g, p) - 0.5 * alpha**2 * pHp
        if state.first_decrease is None:
            state.first_decrease = DeltaE
        state.alpha = alpha
        state.beta = beta
        state.decrease = DeltaE
        state.grad_flat = g
        state.hess_diag_flat = H_diag
        state.hess_quad = pHp
        state.preconditioner_flat = P
        state.search_direction_flat = p
        # return state

        delta_x: Vector = state.alpha * state.search_direction_flat
        state.delta_x_history.append(delta_x)
        # if len(state.delta_x_history) == 20:
        total_path_len: Scalar = jnp.sum(
            jnp.stack([jnp.linalg.norm(dx) for dx in state.delta_x_history])
        )
        net_disp: Scalar = jnp.linalg.norm(
            jnp.sum(jnp.stack(state.delta_x_history), axis=0)
        )
        path_efficiency: Scalar = net_disp / total_path_len
        state.path_efficiency = path_efficiency
        state.total_path_length = total_path_len
        state.net_displacement = net_disp
        return state

    def postprocess(
        self,
        objective: Objective,
        state: PNCGState,
        stats: PNCGStats,
        result: Result,
        *,
        constraints: Iterable[Constraint] = (),
    ) -> OptimizeSolution[PNCGState, PNCGStats]:
        solution = super().postprocess(
            objective, state, stats, result, constraints=constraints
        )
        if self.timer:
            for name in ["fun", "grad_and_hess_diag", "hess_quad"]:
                timer = grapes.get_timer(getattr(objective, name, None), None)
                if timer is not None:
                    timer.finish()
        return solution


def main(cfg: Config) -> None:
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.input)
    model: Model = build_model(mesh)
    forward = Forward(
        model=model,
        optimizer=PNCG(
            max_steps=1000,
            timer=True,
            line_search=PNCG.default_line_search(d_hat=1, line_search_steps=0),
        ),
    )

    def callback(state: PNCG.State, stats: PNCG.Stats) -> None:
        cherries.log_metrics(
            {
                "fun": model.fun(state.params_flat),
                "relative_decrease": stats.relative_decrease,
                "grad_norm": jnp.linalg.norm(state.grad_flat),
                "grad_max_norm": jnp.linalg.norm(state.grad_flat, ord=jnp.inf),
                "alpha": state.alpha,
                "beta": state.beta,
                "delta_x_norm": jnp.linalg.norm(
                    state.alpha * state.search_direction_flat
                ),
                "delta_x_max_norm": jnp.linalg.norm(
                    state.alpha * state.search_direction_flat, ord=jnp.inf
                ),
                "path_efficiency": state.path_efficiency,
                "total_path_length": state.total_path_length,
                "net_displacement": state.net_displacement,
                "stagnation_count": state.stagnation_count,
            },
            step=stats.n_steps,
        )

    solution: PNCG.Solution = forward.step(callback=callback)
    mesh.point_data["Displacement"] = solution.params[mesh.point_data[POINT_ID]]
    melon.save(cherries.temp("21-forward.vtu"), mesh)


if __name__ == "__main__":
    cherries.main(main)
