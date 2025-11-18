from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pyvista as pv
from jaxtyping import Array, Bool, Float

import liblaf.apple.jax.sim as sim_jax
from liblaf import cherries, melon


class Config(cherries.BaseConfig):
    raw: Path = cherries.input("00-input.vtu")

    input: Path = cherries.output("10-input.vtu")
    target: Path = cherries.output("10-target.vtu")


def main(cfg: Config) -> None:
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.raw)
    ic(mesh)

    fraction: Float[Array, " c"] = jnp.asarray(mesh.cell_data["muscle-fraction"])
    active_mask: Bool[Array, " c"] = fraction > 1e-3
    ic(jnp.count_nonzero(active_mask))
    ic(jnp.count_nonzero(mesh.point_data["is-face"]))

    mesh.point_data["dirichlet-mask"] = mesh.point_data["is-skull"]
    mesh.point_data["dirichlet-value"] = np.zeros((mesh.n_points, 3))
    mesh.cell_data["activation"] = np.asarray(sim_jax.rest_activation(mesh.n_cells))
    mesh.cell_data["active-fraction"] = mesh.cell_data["muscle-fraction"]
    mesh.cell_data["lambda"] = np.full((mesh.n_cells,), 3.0)
    mesh.cell_data["mu"] = np.full((mesh.n_cells,), 1.0)

    melon.save(cfg.input, mesh)
    melon.save(cfg.target, mesh)


if __name__ == "__main__":
    cherries.main(main)
