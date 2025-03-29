import einops
import jax
import jax.numpy as jnp
import logging_tree
import pyvista as pv
import typer
from jaxtyping import Bool, Float, PyTree

import liblaf.apple as apple  # noqa: PLR0402
from liblaf import grapes


def main() -> None:
    grapes.init_logging()
    logging_tree.printout()
    return
    mesh: pv.UnstructuredGrid = pv.read("data/input.vtu")  # pyright: ignore[reportAssignmentType]
    dirichlet_mask: Bool[jax.Array, "V 3"] = einops.repeat(
        jnp.asarray(mesh.point_data["dirichlet-mask"]), "V -> V 3"
    )
    dirichlet_values: Float[jax.Array, "V 3"] = jnp.asarray(
        mesh.point_data["dirichlet-values"]
    )
    problem: apple.RegionTetra = apple.RegionTetra(
        dirichlet_mask=dirichlet_mask,
        dirichlet_values=dirichlet_values,
        material=apple.material.AsRigidAsPossible(),
        mesh=mesh,
    )
    q: PyTree = {"mu": jnp.full((mesh.n_cells,), 1e3)}
    problem.prepare()
    res: apple.MinimizeResult = apple.minimize(
        x0=jnp.zeros((problem.n_dof,)),
        fun=lambda x: problem.fun(x, q),
        jac=lambda x: problem.jac(x, q),
        hess=lambda x: problem.hess(x, q),
        algo=apple.MinimizeScipy(method="trust-constr"),
    )
    ic(res)


if __name__ == "__main__":
    typer.run(main)
