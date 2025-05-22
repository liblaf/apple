from pathlib import Path

import jax
import jax.numpy as jnp
import pyvista as pv
from jaxtyping import Float

import liblaf.apple as apple  # noqa: PLR0402
from liblaf import cherries, melon


class Config(cherries.BaseConfig):
    solution: Path = cherries.data("solution-100/solution_000101.vtu")
    tetmesh: Path = cherries.data("10-tetgen.vtu")


def main(cfg: Config) -> None:
    solution: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.solution)
    tetmesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.tetmesh)

    points: Float[jax.Array, "P 3"] = jnp.asarray(tetmesh.points)
    cells: Float[jax.Array, "C 4"] = jnp.asarray(tetmesh.cells_dict[pv.CellType.TETRA])
    dh_dX: Float[jax.Array, "*C 4 3"] = apple.jax.elem.tetra.dh_dX(points[cells])
    u: Float[jax.Array, "*C 3 3"] = jnp.asarray(solution.point_data["solution"])
    F: Float[jax.Array, "*C 3 3"] = apple.jax.elem.tetra.deformation_gradient(
        u[cells], dh_dX
    )

    ic(F)


if __name__ == "__main__":
    cherries.run(main)
