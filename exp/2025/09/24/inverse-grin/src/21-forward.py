from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pyvista as pv
from liblaf.peach import optim
from loguru import logger

from liblaf import cherries, melon
from liblaf.apple import sim
from liblaf.apple.jax.typing import Vector
from liblaf.apple.warp import sim as sim_wp


class Config(cherries.BaseConfig):
    input: Path = cherries.input("11-input.vtu")
    output: Path = cherries.output("21-target.vtu")


def main(cfg: Config) -> None:
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.input)

    builder = sim.ModelBuilder()
    mesh = builder.assign_dofs(mesh)
    builder.add_dirichlet(mesh)
    builder.add_energy(sim_wp.Phace.from_pyvista(mesh))

    model: sim.Model = builder.finish()
    # optimizer: optim.Minimizer = optim.MinimizerScipy(
    #     method="trust-constr", tol=1e-5, options={"verbose": 3}
    # )
    optimizer: optim.Optimizer = optim.PNCG(rtol=1e-10, max_steps=1000)
    solution: optim.OptimizeSolution = optimizer.minimize(
        objective=optim.Objective(
            fun=model.fun,
            grad=model.jac,
            hess_diag=model.hess_diag,
            hess_prod=model.hess_prod,
            hess_quad=model.hess_quad,
            value_and_grad=model.fun_and_jac,
            grad_and_hess_diag=model.jac_and_hess_diag,
        ),
        params=jnp.zeros((model.n_free,)),
    )
    logger.info(solution)
    u: Vector = model.to_full(solution.params)
    mesh.point_data["solution"] = np.asarray(u[mesh.point_data["dof-id"]])
    # mesh.warp_by_vector("solution", inplace=True)
    melon.save(cfg.output, mesh)


if __name__ == "__main__":
    cherries.run(main)
