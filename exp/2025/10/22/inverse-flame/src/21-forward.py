import logging
from pathlib import Path

import jax.numpy as jnp
import pyvista as pv
from jaxtyping import Array, Float
from liblaf.peach.optim import PNCG

from liblaf import cherries, melon
from liblaf.apple import Forward, Model, ModelBuilder
from liblaf.apple.constants import POINT_ID
from liblaf.apple.warp import Phace

type Scalar = Float[Array, ""]
type Vector = Float[Array, " N"]

logger: logging.Logger = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    input: Path = cherries.temp(
        "20-inverse-adam-123k.vtu.d/20-inverse-adam-515k_000945.vtu"
    )


def build_model(mesh: pv.UnstructuredGrid) -> Model:
    builder = ModelBuilder()
    mesh = builder.assign_global_ids(mesh)
    # mesh.point_data["lambda"] = 0.0
    builder.add_dirichlet(mesh)
    builder.add_energy(Phace.from_pyvista(mesh, id="elastic", clamp_lambda=False))
    model: Model = builder.finalize()
    return model


def main(cfg: Config) -> None:
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.input)
    model: Model = build_model(mesh)
    forward = Forward(
        model=model,
        optimizer=PNCG(
            max_steps=1000,
            timer=True,
            rtol=1e-8,
            max_delta=0.15 * model.edges_length_mean,
            # line_search=PNCG.default_line_search(d_hat=1, line_search_steps=0),
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
                # "path_efficiency": state.path_efficiency,
                # "total_path_length": state.total_path_length,
                # "net_displacement": state.net_displacement,
                # "stagnation_count": state.stagnation_count,
            },
            step=stats.n_steps,
        )

    solution: PNCG.Solution = forward.step(callback=callback)
    mesh.point_data["Displacement"] = solution.params[mesh.point_data[POINT_ID]]
    melon.save(cherries.temp("21-forward.vtu"), mesh)


if __name__ == "__main__":
    cherries.main(main)
