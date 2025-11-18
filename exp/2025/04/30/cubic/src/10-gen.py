from pathlib import Path

import einops
import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv
from jaxtyping import Float, PyTree

import liblaf.apple as apple  # noqa: PLR0402
from liblaf import cherries, melon


class Config(cherries.BaseConfig):
    solution: Path = cherries.data("02-intermediate/solution/solution.vtu.series")
    activation: float = 2.0


def main(cfg: Config) -> None:
    box: pv.PolyData = pv.Box((0, 1, 0, 0.2, 0, 0.01))
    geometry: pv.UnstructuredGrid = melon.tetwild(box)
    geometry.cell_data["lambda"] = 3.0
    geometry.cell_data["mu"] = 1.0

    physics = apple.Physics(
        dirichlet_mask=jnp.zeros((geometry.n_points * 3,), dtype=bool)
    )
    physics.add(apple.Object("box", apple.jax.energy.tetra.Phace(), geometry))
    q: PyTree = {
        "box": {
            "activation": einops.repeat(
                jnp.diagflat(
                    jnp.asarray(
                        [
                            1 / cfg.activation,
                            np.sqrt(cfg.activation),
                            np.sqrt(cfg.activation),
                        ]
                    ),
                ),
                "i j -> C i j",
                C=geometry.n_cells,
            ),
            "active-fraction": jnp.full((geometry.n_cells,), 0.5),
            "lambda": geometry.cell_data["lambda"],
            "mu": geometry.cell_data["mu"],
        },
        "initial": jnp.zeros((geometry.n_points, 3)).ravel(),
    }

    writer = melon.SeriesWriter(cfg.solution)

    def callback(intermediate_result: apple.OptimizeResult) -> None:
        u: Float[jax.Array, " F"] = intermediate_result["x"]
        u: Float[jax.Array, " T"] = physics.fill_dirichlet(u, q)
        u: Float[jax.Array, "P 3"] = u[physics.objects["box"].dof_id]
        result: pv.UnstructuredGrid = physics.objects["box"].geometry
        result.point_data["solution"] = np.asarray(u)
        result = result.warp_by_vector("solution")
        writer.append(result)

    result: apple.OptimizeResult = physics.solve(
        q,
        method=apple.OptimizerScipy(tol=5e-5, options={"disp": True, "verbose": 3}),
        callback=callback,
    )
    ic(result)


if __name__ == "__main__":
    cherries.main(main)
