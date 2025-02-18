import einops
import felupe
import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv
import scipy.optimize
import typer
from jaxtyping import Bool, Float

import liblaf.apple as apple  # noqa: PLR0402
import liblaf.grapes as grapes  # noqa: PLR0402


def main() -> None:
    grapes.init_logging()
    mesh: pv.UnstructuredGrid = pv.read("data/input.vtu")
    mesh_felupe = felupe.Mesh(mesh.points, mesh.cells_dict[pv.CellType.TETRA], "tetra")
    fixed_mask: Bool[np.ndarray, " D"] = einops.repeat(
        mesh.point_data["fixed_mask"], "P -> (P 3)"
    )
    fixed_values: Float[jax.Array, " D"] = jnp.asarray(
        mesh.point_data["fixed_disp"].flatten()
    )
    problem: apple.PhysicsProblem = apple.Fixed(
        problem=apple.Corotated(
            mesh=mesh_felupe, p={"lambda": jnp.asarray(3e3), "mu": jnp.asarray(1e3)}
        ),
        fixed_mask=fixed_mask,
        fixed_values=fixed_values,
    )
    result: scipy.optimize.OptimizeResult = problem.solve()
    u: Float[jax.Array, " D"] = problem.fill(result["x"])
    mesh.point_data["solution"] = np.asarray(u.reshape(mesh.n_points, 3))
    mesh.warp_by_vector("solution", inplace=True)
    mesh.save("data/solution.vtu")


if __name__ == "__main__":
    typer.run(main)
