import logging
from pathlib import Path

import jax.numpy as jnp
import pyvista as pv
from liblaf.peach.optim import PNCG

from liblaf import cherries, melon
from liblaf.apple import Forward, Model, ModelBuilder
from liblaf.apple.constants import POINT_ID
from liblaf.apple.warp import Phace

logger: logging.Logger = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    input: Path = cherries.temp(
        "20-inverse-adam-123k.vtu.d/20-inverse-adam-123k_000087.vtu"
    )


def build_model(mesh: pv.UnstructuredGrid) -> Model:
    builder = ModelBuilder()
    mesh = builder.assign_global_ids(mesh)
    builder.add_dirichlet(mesh)
    builder.add_energy(Phace.from_pyvista(mesh, id="elastic"))
    model: Model = builder.finalize()
    return model


def main(cfg: Config) -> None:
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.input)
    model: Model = build_model(mesh)
    forward = Forward(model=model)

    def callback(state: PNCG.State, stats: PNCG.Stats) -> None:
        cherries.log_metrics(
            {
                "relative_decrease": stats.relative_decrease,
                "grad_norm": jnp.linalg.norm(state.grad_flat),
                "grad_max_norm": jnp.linalg.norm(state.grad_flat, ord=jnp.inf),
                "alpha": state.alpha,
                "delta_x_norm": jnp.linalg.norm(
                    state.alpha * state.search_direction_flat
                ),
                "delta_x_max_norm": jnp.linalg.norm(
                    state.alpha * state.search_direction_flat, ord=jnp.inf
                ),
            },
            step=stats.n_steps,
        )

    solution: PNCG.Solution = forward.step(callback=callback)
    mesh.point_data["Displacement"] = solution.params[mesh.point_data[POINT_ID]]
    melon.save(cherries.temp("21-forward.vtu"), mesh)


if __name__ == "__main__":
    cherries.main(main)
