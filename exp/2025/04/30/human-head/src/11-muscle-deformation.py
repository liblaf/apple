from pathlib import Path

import einops
import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv
from jaxtyping import Float

import liblaf.apple as apple  # noqa: PLR0402
from liblaf import cherries, melon


class Config(cherries.BaseConfig):
    solution: Path = cherries.data("solution/solution_000177.vtu")
    tetmesh: Path = cherries.data("10-tetgen.vtu")


def main(cfg: Config) -> None:
    solution: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.solution)
    tetmesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.tetmesh)

    muscle_direction: Float[jax.Array, "C 3"] = jnp.asarray(
        tetmesh.cell_data["muscle-direction"]
    )
    muscle_fraction: Float[jax.Array, " C"] = jnp.asarray(
        tetmesh.cell_data["muscle-fraction"]
    )
    orientation: Float[jax.Array, "C 3 3"] = apple.jax.math.orientation_matrix(
        muscle_direction,
        einops.repeat(jnp.asarray([1.0, 0.0, 0.0]), "i -> C i", C=tetmesh.n_cells),
    )

    points: Float[jax.Array, "P 3"] = jnp.asarray(tetmesh.points)
    cells: Float[jax.Array, "C 4"] = jnp.asarray(tetmesh.cells_dict[pv.CellType.TETRA])
    dh_dX: Float[jax.Array, "C 4 3"] = apple.jax.elem.tetra.dh_dX(points[cells])
    dV: Float[jax.Array, " C"] = apple.jax.elem.tetra.dV(points[cells])
    dV *= muscle_fraction
    u: Float[jax.Array, "C 3 3"] = jnp.asarray(solution.point_data["solution"])
    F: Float[jax.Array, "C 3 3"] = apple.jax.elem.tetra.deformation_gradient(
        u[cells], dh_dX
    )
    F_aligned: Float[jax.Array, "C 3 3"] = einops.einsum(
        orientation, F, orientation, "C i j, C j k, C l k -> C i l"
    )

    muscles: list[str] = np.unique(tetmesh.cell_data["muscle-name"])
    for muscle in muscles:
        if not muscle:
            continue
        mask: Float[jax.Array, " C"] = tetmesh.cell_data["muscle-name"] == muscle
        F_muscle_aligned: Float[jax.Array, "3 3"] = einops.einsum(
            F_aligned[mask], dV[mask], "C i j, C -> i j"
        ) / jnp.sum(dV[mask])
        ic(muscle, F_muscle_aligned, F_muscle_aligned - jnp.identity(3))
        ic(jnp.linalg.det(F_muscle_aligned))


if __name__ == "__main__":
    cherries.run(main)
