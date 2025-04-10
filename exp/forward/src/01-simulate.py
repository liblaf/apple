from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv
import rich.traceback
from jaxtyping import Bool, Float

import liblaf.apple as apple  # noqa: PLR0402
from liblaf import cherries, grapes, melon


class Config(cherries.BaseConfig):
    callback: bool = False
    input: Path = grapes.find_project_dir() / "data/bunny/input.vtu"
    method: str = "pncg"
    output_animation: Path = (
        grapes.find_project_dir() / "data/bunny/static/animation.pvd"
    )
    output: Path = grapes.find_project_dir() / "data/bunny/static/output.vtu"


def main(cfg: Config) -> None:
    rich.traceback.install(show_locals=True)
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.input)
    dirichlet_mask: Bool[jax.Array, " N"] = jnp.asarray(
        mesh.point_data["dirichlet-mask"]
    ).ravel()
    dirichlet_values: Float[jax.Array, " N"] = jnp.asarray(
        mesh.point_data["dirichlet-values"]
    ).ravel()
    box = apple.obj.tetra.ObjectTetra(
        name="box",
        params={
            "lambda": jnp.asarray(mesh.cell_data["lambda"]),
            "mu": jnp.asarray(mesh.cell_data["mu"]),
        },
        material=apple.material.tetra.AsRigidAsPossibleFilter(),
        mesh=mesh,
    )
    problem = apple.PhysicsProblem(
        objects=[box], dirichlet_mask=dirichlet_mask, dirichlet_values=dirichlet_values
    )
    problem.prepare()

    def warp_result(result: apple.MinimizeResult) -> pv.UnstructuredGrid:
        solution: Float[jax.Array, " DoF"] = result["x"]
        solution: Float[jax.Array, " F"] = problem.fill(solution)
        solution: Float[jax.Array, " N"] = box.select_dof(solution)
        solution: Float[jax.Array, "V 3"] = box.unravel_u(solution)
        output: pv.UnstructuredGrid = mesh.copy()
        output.point_data["solution"] = np.asarray(solution)
        jac: Float[jax.Array, " DoF"] = result["jac"]  # pyright: ignore[reportAssignmentType]
        jac: Float[jax.Array, " F"] = problem.fill_zeros(jac)
        jac: Float[jax.Array, " N"] = box.select_dof(jac)
        jac: Float[jax.Array, "V 3"] = box.unravel_u(jac)
        output.point_data["jac"] = np.asarray(jac)
        output.warp_by_vector("solution", inplace=True)
        return output

    u0: Float[jax.Array, " F"] = jnp.asarray(mesh.point_data["initial"]).ravel()
    u0: Float[jax.Array, " DoF"] = u0[~dirichlet_mask]
    writer = melon.io.PVDWriter(cfg.output_animation)

    def callback(intermediate_result: apple.MinimizeResult) -> None:
        ic(intermediate_result)
        result: pv.UnstructuredGrid = warp_result(intermediate_result)
        writer.append(result)

    callback(apple.MinimizeResult({"x": u0, "jac": problem.jac(u0)}))  # pyright: ignore[reportAssignmentType]

    if not cfg.callback:
        callback = None  # pyright: ignore[reportAssignmentType]
    if cfg.method == "pncg":
        algo = apple.MinimizePNCG(eps=1e-10, iter_max=150)
    elif cfg.method == "scipy":
        algo = apple.MinimizeScipy(
            method="trust-constr", options={"disp": True, "verbose": 3}
        )
    else:
        raise NotImplementedError(f"Unknown method: {cfg.method}")  # noqa: EM102

    result: apple.MinimizeResult = problem.solve(
        u0=u0, algo=algo, callback=callback if cfg.callback else None
    )
    ic(result)
    writer.end()
    output: pv.UnstructuredGrid = warp_result(result)
    melon.save(cfg.output, output)


if __name__ == "__main__":
    cherries.run(main)
