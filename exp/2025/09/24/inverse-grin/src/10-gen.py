from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pyvista as pv
from jaxtyping import Array, Bool, Float

import liblaf.apple as apple  # noqa: PLR0402
from liblaf import cherries, melon


class Config(cherries.BaseConfig):
    raw: Path = cherries.input("00-raw.vtu")

    output: Path = cherries.output("10-input.vtu")


def main(cfg: Config) -> None:
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.raw)
    ic(mesh)

    gamma: float = 1e3
    activation_ref: Float[Array, " 6"] = jnp.asarray(
        [jnp.reciprocal(gamma), gamma**0.2, gamma**0.2, 0.0, 0.0, 0.0]
    )
    activation: Float[Array, " c 6"] = apple.jax.sim.rest_activation(
        n_cells=mesh.n_cells
    )
    fraction: Float[Array, " c"] = jnp.asarray(mesh.cell_data["muscle-fraction"])
    active_mask: Bool[Array, " c"] = fraction > 1e-3
    orientation: Float[Array, "ac 3 3"] = jnp.asarray(
        mesh.cell_data["muscle-orientation"]
    ).reshape(mesh.n_cells, 3, 3)[active_mask]
    activation_active: Float[Array, "ac 6"] = apple.jax.sim.transform_activation(
        activation_ref[None], orientation
    )
    activation = activation.at[active_mask].set(activation_active)

    mesh.point_data["dirichlet-mask"] = mesh.point_data["is-skull"]
    mesh.point_data["dirichlet-values"] = np.zeros((mesh.n_points, 3))
    mesh.cell_data["activation"] = np.asarray(activation)
    mesh.cell_data["active-fraction"] = mesh.cell_data["muscle-fraction"]
    mesh.cell_data["lambda"] = np.full((mesh.n_cells,), 3.0)
    mesh.cell_data["mu"] = np.full((mesh.n_cells,), 1.0)

    melon.save(cfg.output, mesh)


if __name__ == "__main__":
    cherries.run(main)
