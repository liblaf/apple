import jax.numpy as jnp
import pyvista as pv
import scipy
import scipy.optimize
import typer

import liblaf.apple as apple  # noqa: PLR0402
import liblaf.grapes as grapes  # noqa: PLR0402


def main() -> None:
    grapes.init_logging()
    mesh: pv.UnstructuredGrid = pv.read("data/input.vtu")  # pyright: ignore[reportAssignmentType]
    region = apple.RegionTetra(mesh)
    model: apple.Problem = apple.preset.Fixed(
        apple.preset.Sum(
            [
                apple.preset.SaintVenantKirchhoff(region=region, lambda_=3e3, mu=1e3),
                apple.preset.Gravity(
                    region=region, density=1, gravity=[0.0, -9.8, 0.0]
                ),
            ]
        ),
        fixed_mask=jnp.repeat(mesh.point_data["fixed_mask"], 3),
        fixed_values=mesh.point_data["fixed_disp"].flatten(),
    )
    prepared: apple.ProblemPrepared = model.prepare()

    def callback(intermediate_result: scipy.optimize.OptimizeResult) -> None:
        ic(intermediate_result)

    result: scipy.optimize.OptimizeResult = apple.minimize(
        prepared.fun,
        x0=jnp.zeros((prepared.n_dof,)),
        method="trust-constr",
        jac=prepared.jac,
        hess=prepared.hess,
        options={"disp": True, "verbose": 3},
        # callback=callback,
    )
    mesh.point_data["solution"] = model.fill(result["x"]).reshape((mesh.n_points, 3))
    mesh.warp_by_vector("solution", inplace=True)
    mesh.save("data/solution.vtu")


if __name__ == "__main__":
    typer.run(main)
