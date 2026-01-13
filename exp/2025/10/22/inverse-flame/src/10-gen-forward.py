from pathlib import Path

import jax
import jax.numpy as jnp
import pyvista as pv
from jaxtyping import Array, Float, Integer

from liblaf import cherries, melon
from liblaf.apple.constants import ACTIVATION, MUSCLE_FRACTION, POINT_ID
from liblaf.apple.model import Forward, Model, ModelBuilder
from liblaf.apple.warp import Phace


class Config(cherries.BaseConfig):
    raw: Path = cherries.input("20-inverse-adam-Expression002-515k.vtu")
    output: Path = cherries.output("10-input-forward-515k.vtu")


def get_muscle_id_to_cell_neighbors(
    mesh: pv.UnstructuredGrid,
) -> dict[int, Integer[Array, "N 2"]]:
    unique_muscle_id: Integer[Array, " muscles"] = jnp.unique(
        mesh.cell_data["MuscleId"][mesh.cell_data[MUSCLE_FRACTION] > 1e-3]
    )
    muscle_id_to_cell_neighbors: dict[int, Integer[Array, "N 2"]] = {}
    for muscle_id in unique_muscle_id.tolist():
        if muscle_id < 0:
            continue
        muscle_id_to_cell_neighbors[muscle_id] = jnp.asarray(
            mesh.field_data[f"Muscle{muscle_id}CellNeighbors"]
        )
    return muscle_id_to_cell_neighbors


def smooth_activation(
    activation: Float[Array, "cells 6"],
    muscle_id_to_cell_neighbors: dict[int, Integer[Array, "N 2"]],
    iterations: int = 1,
    alpha: float = 0.5,
) -> Float[Array, "cells 6"]:
    @jax.jit
    def _smooth_once(activation: Float[Array, "cells 6"]) -> Float[Array, "cells 6"]:
        for cell_neighbors in muscle_id_to_cell_neighbors.values():
            neighbor_sum: Float[Array, "cells 6"] = jax.ops.segment_sum(
                jnp.concatenate(
                    [activation[cell_neighbors[:, 1]], activation[cell_neighbors[:, 0]]]
                ),
                jnp.concatenate([cell_neighbors[:, 0], cell_neighbors[:, 1]]),
                num_segments=activation.shape[0],
            )
            neighbor_count: Float[Array, " cells"] = jax.ops.segment_sum(
                jnp.ones((cell_neighbors.shape[0] * 2,)),
                jnp.concatenate([cell_neighbors[:, 0], cell_neighbors[:, 1]]),
                num_segments=activation.shape[0],
            )
            neighbor_avg: Float[Array, "cells 6"] = jnp.where(
                neighbor_count[:, jnp.newaxis] > 0,
                neighbor_sum / neighbor_count[:, jnp.newaxis],
                activation,
            )
            activation: Float[Array, "cells 6"] = (
                1.0 - alpha
            ) * activation + alpha * neighbor_avg
        return activation

    for _ in range(iterations):
        activation = _smooth_once(activation)
    return activation


def forward(mesh: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    builder = ModelBuilder()
    mesh = builder.assign_global_ids(mesh)
    builder.add_dirichlet(mesh)
    elastic = Phace.from_pyvista(mesh)
    builder.add_energy(elastic)
    model: Model = builder.finalize()
    forward = Forward(model)
    forward.step()
    mesh.point_data["Expression004"] = model.u_full[mesh.point_data[POINT_ID]]  # pyright: ignore[reportArgumentType]
    return mesh


def main(cfg: Config) -> None:
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.raw)
    muscle_id_to_cell_neighbors: dict[int, Integer[Array, "N 2"]] = (
        get_muscle_id_to_cell_neighbors(mesh)
    )
    activation: Float[Array, " cells"] = jnp.asarray(mesh.cell_data[ACTIVATION])

    activation = smooth_activation(
        activation, muscle_id_to_cell_neighbors, iterations=20
    )
    mesh.cell_data["ActivationOld"] = mesh.cell_data[ACTIVATION]
    mesh.cell_data[ACTIVATION] = activation  # pyright: ignore[reportArgumentType]

    mesh = forward(mesh)

    melon.save(cfg.output, mesh)


if __name__ == "__main__":
    cherries.main(main)
