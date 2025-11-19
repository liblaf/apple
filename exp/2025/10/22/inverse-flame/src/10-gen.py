from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pyvista as pv
from jaxtyping import Array, Bool, Float

from liblaf import cherries, melon


class Config(cherries.BaseConfig):
    raw: Path = cherries.input("00-input.vtu")

    input: Path = cherries.output("10-input.vtu")
    target: Path = cherries.output("10-target.vtu")


def main(cfg: Config) -> None:
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.raw)
    ic(mesh)

    fractions: Float[Array, " c"] = jnp.asarray(mesh.cell_data["MuscleFractions"])
    active_mask: Bool[Array, " c"] = fractions > 1e-3
    ic(jnp.count_nonzero(active_mask))
    ic(jnp.count_nonzero(mesh.point_data["IsFace"]))

    mesh.point_data["IsCranium"] &= mesh.points[:, 1] > 28.0
    mesh.point_data["IsMandible"] &= mesh.points[:, 1] < 24.0
    mesh.point_data["DirichletMask"] = (
        mesh.point_data["IsCranium"] | mesh.point_data["IsMandible"]
    )
    mesh.point_data["DirichletValues"] = np.zeros((mesh.n_points, 3))
    mesh.cell_data["Activations"] = np.zeros((mesh.n_cells, 6))
    mesh.cell_data["lambda"] = np.full((mesh.n_cells,), 3.0)
    mesh.cell_data["mu"] = np.full((mesh.n_cells,), 1.0)

    melon.save(cfg.input, mesh)
    melon.save(cfg.target, mesh)


if __name__ == "__main__":
    cherries.main(main)
