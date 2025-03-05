import einops
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
    mesh: pv.PolyData = pv.read("data/input.vtp")  # pyright: ignore[reportAssignmentType]
    fixed_mask: Bool[np.ndarray, " D"] = einops.repeat(
        mesh.point_data["fixed_mask"], "P -> (P 3)"
    )
    fixed_values: Float[jax.Array, " D"] = jnp.asarray(
        mesh.point_data["fixed_disp"].flatten()
    )
    problem: apple.AbstractPhysicsProblem = apple.Fixed(
        problem=apple.Sum(
            problems=[
                apple.Koiter(
                    mesh=mesh, lmbda=mesh.cell_data["lmbda"], mu=mesh.cell_data["mu"]
                ),
                apple.Gravity(
                    mass=apple.elem.triangle.mass(mesh), n_points=mesh.n_points
                ),
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
    mesh.save("data/target.vtp")


if __name__ == "__main__":
    typer.run(main)
