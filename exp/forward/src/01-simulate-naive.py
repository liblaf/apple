from collections.abc import Callable
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv
import rich.traceback
import scipy
import scipy.optimize
import scipy.sparse.linalg
from jaxtyping import Bool, Float

import liblaf.apple as apple  # noqa: PLR0402
from liblaf import cherries, grapes, melon


class Config(cherries.BaseConfig):
    callback: bool = False
    input: Path = grapes.find_project_dir() / "data/input.vtu"
    method: str = "scipy"
    output_animation: Path = grapes.find_project_dir() / "data/animation.pvd"
    output: Path = grapes.find_project_dir() / "data/output.vtu"


def hess_op(fun: Callable, x: jax.Array) -> scipy.sparse.linalg.LinearOperator:
    def matvec(v: jax.Array) -> jax.Array:
        v = jnp.asarray(v, dtype=float)
        return apple.hvp_op(fun, x)(v)

    return scipy.sparse.linalg.LinearOperator(
        matvec=matvec,  # pyright: ignore[reportCallIssue]
        shape=(x.size, x.size),
        dtype=float,
    )


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
        result: pv.UnstructuredGrid = mesh.copy()
        result.point_data["solution"] = np.asarray(solution)
        result.warp_by_vector("solution", inplace=True)
        return result

    writer = melon.io.PVDWriter(cfg.output_animation)

    def callback(intermediate_result: apple.MinimizeResult) -> None:
        ic(intermediate_result)
        result: pv.UnstructuredGrid = warp_result(intermediate_result)
        writer.append(result)

    if not cfg.callback:
        callback = None  # pyright: ignore[reportAssignmentType]
    if cfg.method == "pncg":
        algo = apple.MinimizePNCG()
    elif cfg.method == "scipy":
        algo = apple.MinimizeScipy(
            method="trust-constr", options={"disp": True, "verbose": 3}
        )
    else:
        raise NotImplementedError(f"Unknown method: {cfg.method}")  # noqa: EM102
    u0: Float[jax.Array, " F"] = jnp.asarray(mesh.point_data["initial"]).ravel()
    u0: Float[jax.Array, " DoF"] = u0[~dirichlet_mask]
    result: apple.MinimizeResult = apple.minimize(
        problem.fun,
        x0=u0,
        algo=algo,
        jac=jax.grad(problem.fun),
        hess=lambda x: hess_op(problem.fun, x),
        callback=callback if cfg.callback else None,
    )
    ic(result)
    writer.end()
    output: pv.UnstructuredGrid = warp_result(result)
    melon.save(cfg.output, output)


if __name__ == "__main__":
    cherries.run(main)
