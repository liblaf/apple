from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv
from loguru import logger

from liblaf import cherries, melon
from liblaf.apple import sim
from liblaf.apple.jax import optim
from liblaf.apple.jax import sim as sim_jax
from liblaf.apple.jax.typing import Vector

jax.config.update("jax_enable_x64", True)  # noqa: FBT003


class Config(cherries.BaseConfig):
    input: Path = cherries.input("10-input.vtu")
    output: Path = cherries.output("20-target.vtu")


def main(cfg: Config) -> None:
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.input)

    builder = sim.ModelBuilder()
    mesh = builder.assign_dofs(mesh)
    builder.add_dirichlet(mesh)
    builder.add_energy(sim_jax.ARAPActive.from_pyvista(mesh))

    model: sim.Model = builder.finish()
    optimizer: optim.Minimizer = optim.MinimizerScipy(
        method="trust-constr", tol=1e-5, options={"verbose": 3}
    )
    solution: optim.Solution = optimizer.minimize(
        x0=jnp.zeros((model.n_free,)),
        fun=sim.Model.static_fun,
        jac=sim.Model.static_jac,
        hessp=sim.Model.static_hess_prod,
        fun_and_jac=sim.Model.static_fun_and_jac,
        args=(model,),
    )
    logger.info(solution)
    u: Vector = model.to_full(solution["x"])
    mesh.point_data["solution"] = np.asarray(u[mesh.point_data["point-id"]])
    mesh.warp_by_vector("solution", inplace=True)
    melon.save(cfg.output, mesh)


if __name__ == "__main__":
    cherries.run(main)
