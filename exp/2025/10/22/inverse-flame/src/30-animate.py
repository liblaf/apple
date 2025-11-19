import logging
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pyvista as pv
import warp as wp
from jaxtyping import Array, Float
from liblaf.peach import optim, tree

from liblaf import cherries, grapes, melon
from liblaf.apple import sim
from liblaf.apple.warp import sim as sim_wp
from liblaf.apple.warp import utils as wp_utils
from liblaf.apple.warp.typing import vec6

logger: logging.Logger = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    input: Path = cherries.input("20-inverse-smooth.vtu")

    output: Path = cherries.output("30-animation-smooth.vtu.series")


def build_model(mesh: pv.UnstructuredGrid) -> sim.Model:
    builder = sim.ModelBuilder()
    mesh = builder.assign_dofs(mesh)
    builder.add_dirichlet(mesh)
    builder.add_energy(sim_wp.Phace.from_pyvista(mesh, id="elastic"))
    model: sim.Model = builder.finish()
    return model


@tree.define
class Forward:
    model: sim.Model
    optimizer: optim.Optimizer = tree.field(
        factory=lambda: optim.PNCG(rtol=1e-5, max_steps=500)
    )

    @property
    def energy(self) -> sim_wp.Phace:
        return self.model.model_warp.energies["elastic"]  # pyright: ignore[reportReturnType]

    def solve(self, act: Float[Array, "c 6"]) -> Float[Array, "p 3"]:
        wp.copy(self.energy.params.activations, wp_utils.to_warp(act, vec6))
        objective = optim.Objective(
            fun=self.model.fun,
            grad=self.model.jac,
            hess_diag=self.model.hess_diag,
            hess_prod=self.model.hess_prod,
            hess_quad=self.model.hess_quad,
            value_and_grad=self.model.fun_and_jac,
            grad_and_hess_diag=self.model.jac_and_hess_diag,
        )
        params = jnp.zeros((self.model.n_free,))
        solution: optim.OptimizeSolution = self.optimizer.minimize(
            objective=objective, params=params
        )
        logger.info("Forward time: %g sec", solution.stats.time)
        assert solution.success
        u_free: Float[Array, " free"] = solution.params
        return self.model.to_full(u_free)


def main(cfg: Config) -> None:
    start_idx: str = "000"
    end_idx: str = "001"

    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.input)
    model: sim.Model = build_model(mesh)
    energy: sim_wp.Phace = model.model_warp.energies["elastic"]  # pyright: ignore[reportAssignmentType]
    start_act: Float[Array, "c 6"] = jnp.asarray(
        mesh.cell_data[f"Activations{start_idx}"]
    )
    end_act: Float[Array, "c 6"] = jnp.asarray(mesh.cell_data[f"Activations{end_idx}"])

    forward = Forward(model=model)

    with melon.SeriesWriter(cfg.output) as writer:
        for t in grapes.track(jnp.linspace(0.0, 1.0, num=30)):
            act: Float[Array, "c 6"] = (1.0 - t) * start_act + t * end_act
            wp.copy(energy.params.activations, wp_utils.to_warp(act, vec6))
            u: Float[Array, "p 3"] = forward.solve(act)
            mesh.point_data["Displacement"] = np.asarray(u)
            mesh.cell_data["Activations"] = np.asarray(act)
            writer.append(mesh)


if __name__ == "__main__":
    cherries.main(main)
