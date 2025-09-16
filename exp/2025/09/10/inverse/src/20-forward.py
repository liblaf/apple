from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pyvista as pv
from loguru import logger

from liblaf import cherries, melon
from liblaf.apple import sim
from liblaf.apple.jax import optim
from liblaf.apple.jax.typing import Vector
from liblaf.apple.warp import sim as sim_wp


class Config(cherries.BaseConfig):
    input: Path = cherries.input("10-input.vtu")
    output: Path = cherries.output("20-target.vtu")


def main(cfg: Config) -> None:
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.input)

    builder = sim.ModelBuilder()
    mesh = builder.assign_dofs(mesh)
    builder.add_dirichlet(mesh)
    builder.add_energy(sim_wp.ArapActive.from_pyvista(mesh))

    model: sim.Model = builder.finish()
    optimizer: optim.Minimizer = optim.MinimizerScipy(
        method="trust-constr", tol=1e-5, options={"verbose": 3}
    )
    optimizer: optim.Minimizer = optim.MinimizerPNCG(rtol=1e-5, maxiter=1000)
    solution: optim.Solution = optimizer.minimize(
        x0=jnp.zeros((model.n_free,)),
        fun=sim.fun,
        jac=sim.jac,
        hessp=sim.hess_prod,
        hess_diag=sim.hess_diag,
        hess_quad=sim.hess_quad,
        fun_and_jac=sim.fun_and_jac,
        jac_and_hess_diag=sim.jac_and_hess_diag,
        args=(model,),
    )
    logger.info(solution)
    u: Vector = model.to_full(solution["x"])
    mesh.point_data["solution"] = np.asarray(u[mesh.point_data["point-id"]])
    mesh.warp_by_vector("solution", inplace=True)
    melon.save(cfg.output, mesh)


if __name__ == "__main__":
    cherries.run(main)
