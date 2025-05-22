from pathlib import Path

import einops
import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv
from jaxtyping import Float, PyTree
from numpy.typing import ArrayLike

import liblaf.apple as apple  # noqa: PLR0402
from liblaf import cherries, melon


class Config(cherries.BaseConfig):
    activation: float = 100

    tetgen: Path = cherries.data("10-tetgen.vtu")

    solution: Path = cherries.data("solution/solution.vtu.series")


def main(cfg: Config) -> None:
    geometry: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.tetgen)
    geometry.cell_data["lambda"] = 3.0
    geometry.cell_data["mu"] = 1.0

    physics = apple.Physics(
        dirichlet_mask=jnp.zeros((geometry.n_points * 3,), dtype=bool)
    )
    physics.add(apple.Object("flesh", apple.jax.energy.tetra.Phace(), geometry))
    q: PyTree = {
        "flesh": {
            "activation": activations(
                geometry,
                np.asarray(
                    [
                        1.0 / cfg.activation,
                        np.sqrt(cfg.activation),
                        np.sqrt(cfg.activation),
                    ]
                ),
            ),
            "active-fraction": geometry.cell_data["muscle-fraction"],
            "lambda": geometry.cell_data["lambda"],
            "mu": geometry.cell_data["mu"],
        },
        "initial": jnp.zeros((geometry.n_points, 3)).ravel(),
    }

    writer = melon.SeriesWriter(cfg.solution)

    def callback(intermediate_result: apple.OptimizeResult) -> None:
        u: Float[jax.Array, " F"] = intermediate_result["x"]
        u: Float[jax.Array, " T"] = physics.fill_dirichlet(u, q)
        u: Float[jax.Array, "P 3"] = u[physics.objects["flesh"].dof_id]
        result: pv.UnstructuredGrid = physics.objects["flesh"].geometry
        result.point_data["solution"] = np.asarray(u)
        result = result.warp_by_vector("solution")
        writer.append(result)

    result: apple.OptimizeResult = physics.solve(
        q,
        method=apple.OptimizerScipy(tol=5e-5, options={"disp": True, "verbose": 3}),
        callback=callback,
    )
    ic(result)


def activations(
    tetmesh: pv.UnstructuredGrid, stretch: Float[ArrayLike, " 3"]
) -> Float[jax.Array, "C 3 3"]:
    muscle_direction: Float[np.ndarray, "C 3"] = tetmesh.cell_data["muscle-direction"]
    muscle_fraction: Float[np.ndarray, " C"] = tetmesh.cell_data["muscle-fraction"]
    stretch: Float[jax.Array, " 3"] = jnp.asarray(stretch)
    orientation: Float[jax.Array, "C 3 3"] = apple.jax.math.orientation_matrix(
        muscle_direction,
        einops.repeat(np.asarray([1.0, 0.0, 0.0]), "i -> C i", C=tetmesh.n_cells),
    )
    activation: Float[jax.Array, "C 3 3"] = einops.einsum(
        orientation,
        jnp.asarray(stretch),
        orientation,
        "C i j, j, C j k -> C i k",
    )
    activation.at[muscle_fraction < 1e-6].set(jnp.identity(3))
    return activation


if __name__ == "__main__":
    cherries.run(main)
