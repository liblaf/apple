import einops
import felupe
import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv
import scipy.optimize
import typer
from jaxtyping import Bool, Float, PyTree

import liblaf.apple as apple  # noqa: PLR0402
import liblaf.grapes as grapes  # noqa: PLR0402


def main() -> None:
    grapes.init_logging()
    mesh: pv.UnstructuredGrid = pv.read("data/input.vtu")  # pyright: ignore[reportAssignmentType]
    mesh_felupe = felupe.Mesh(mesh.points, mesh.cells_dict[pv.CellType.TETRA], "tetra")
    fixed_mask: Bool[np.ndarray, " D"] = einops.repeat(
        mesh.point_data["fixed_mask"], "P -> (P 3)"
    )
    fixed_values: Float[jax.Array, " D"] = jnp.asarray(
        mesh.point_data["fixed_disp"].flatten()
    )
    problem: apple.AbstractPhysicsProblem = apple.Fixed(
        problem=apple.Sum(
            problems=[
                apple.Corotated(
                    mesh=mesh_felupe,
                    lmbda=mesh.cell_data["lmbda"],
                    mu=mesh.cell_data["mu"],
                ),
                apple.Gravity(mass=apple.elem.tetra.mass(mesh), n_points=mesh.n_points),
            ]
        ),
        fixed_mask=fixed_mask,
        fixed_values=fixed_values,
    )
    q: PyTree = {}
    problem.prepare(q)
    result: scipy.optimize.OptimizeResult = problem.solve(q)
    ic(result)
    u: Float[jax.Array, " D"] = problem.fill(result["x"])
    mesh.point_data["solution"] = np.asarray(u)
    mesh.warp_by_vector("solution", inplace=True)
    mesh.save("data/target.vtu")
    q0: PyTree = {
        "corotated": {
            "lmbda": jnp.full((mesh.n_cells,), 3e3),
            "mu": jnp.full((mesh.n_cells,), 1e3),
        }
    }
    problem.prepare(q0)
    result: scipy.optimize.OptimizeResult = problem.solve(q0)
    ic(result)
    u: Float[jax.Array, " D"] = problem.fill(result["x"])
    mesh: pv.UnstructuredGrid = pv.read("data/input.vtu")  # pyright: ignore[reportAssignmentType]
    mesh.point_data["solution"] = np.asarray(u)
    mesh.warp_by_vector("solution", inplace=True)
    mesh.save("data/initial.vtu")


if __name__ == "__main__":
    typer.run(main)
