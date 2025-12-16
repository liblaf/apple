from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pyvista as pv
from environs import env
from jaxtyping import Array, Bool, Float, Integer

from liblaf import cherries, grapes, melon
from liblaf.apple import utils
from liblaf.apple.constants import (
    ACTIVATION,
    DIRICHLET_MASK,
    DIRICHLET_VALUE,
    LAMBDA,
    MU,
    MUSCLE_FRACTION,
)

SUFFIX: str = env.str("SUFFIX", default="-68k-coarse")


class Config(cherries.BaseConfig):
    raw: Path = cherries.input(f"00-input{SUFFIX}.vtu")

    input: Path = cherries.output(f"10-input{SUFFIX}.vtu")


def gen_muscle_id_to_cell_id(
    mesh: pv.UnstructuredGrid,
) -> dict[int, Integer[Array, " cells_per_muscle"]]:
    fractions: Float[Array, " cells"] = jnp.asarray(mesh.cell_data[MUSCLE_FRACTION])
    is_muscle: Bool[Array, " cells"] = fractions > 1e-3

    cell_id_to_muscle_id: Integer[Array, " cells"] = jnp.asarray(
        mesh.cell_data["MuscleId"]
    )
    cell_id_to_muscle_id = jnp.where(is_muscle, cell_id_to_muscle_id, -1)
    muscle_id_to_cell_id: dict[int, Integer[Array, " cells_per_muscle"]] = (
        utils.group_indices(cell_id_to_muscle_id)
    )
    muscle_id_to_cell_id.pop(-1, None)  # remove non-muscle cells
    return muscle_id_to_cell_id


def gen_muscle_cell_neighbors(
    mesh: pv.UnstructuredGrid,
    muscle_id_to_cell_id: dict[int, Integer[Array, " cells_per_muscle"]],
) -> dict[int, Integer[Array, "N 2"]]:
    cells: Integer[Array, "cells 4"] = jnp.asarray(mesh.cells_dict[pv.CellType.TETRA])  # pyright: ignore[reportArgumentType]
    muscle_id_to_cell_neighbors: dict[int, Integer[Array, "N 2"]] = {
        muscle_id: cell_ids[melon.cell_neighbors(cells[cell_ids])]
        for muscle_id, cell_ids in grapes.track(
            muscle_id_to_cell_id.items(),
            total=len(muscle_id_to_cell_id),
            description="Muscle Cell Neighbors",
        )
    }
    return muscle_id_to_cell_neighbors


def main(cfg: Config) -> None:
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.raw)
    mesh = melon.tet.extract_cells(mesh, mesh.cell_data["InFaceConvex"])
    ic(mesh)

    fractions: Float[Array, " cells"] = jnp.asarray(mesh.cell_data[MUSCLE_FRACTION])
    is_muscle: Bool[Array, " cells"] = fractions > 1e-3
    ic(jnp.count_nonzero(is_muscle))
    ic(jnp.count_nonzero(mesh.point_data["IsFace"]))

    mesh = mesh.compute_cell_sizes(length=False, area=False, volume=True)  # pyright: ignore[reportAssignmentType]
    assert np.all(mesh.cell_data["Volume"] > 0.0)
    mesh.cell_data["MuscleVolume"] = (
        mesh.cell_data[MUSCLE_FRACTION] * mesh.cell_data["Volume"]
    )

    muscle_id_to_cell_id: dict[int, Integer[Array, " cells_per_muscle"]] = (
        gen_muscle_id_to_cell_id(mesh)
    )
    for muscle_id, cell_ids in muscle_id_to_cell_id.items():
        mesh.field_data[f"Muscle{muscle_id}CellId"] = cell_ids  # pyright: ignore[reportArgumentType]

    muscle_id_to_cell_neighbors: dict[int, Integer[Array, "N 2"]] = (
        gen_muscle_cell_neighbors(mesh, muscle_id_to_cell_id)
    )
    for muscle_id, cell_neighbors in muscle_id_to_cell_neighbors.items():
        mesh.field_data[f"Muscle{muscle_id}CellNeighbors"] = cell_neighbors  # pyright: ignore[reportArgumentType]

    mesh.point_data[DIRICHLET_MASK] = (
        mesh.point_data["IsCranium"] | mesh.point_data["IsMandible"]
    )
    mesh.point_data[DIRICHLET_VALUE] = np.zeros((mesh.n_points, 3))
    mesh.cell_data[ACTIVATION] = np.zeros((mesh.n_cells, 6))
    mesh.cell_data[LAMBDA] = np.full((mesh.n_cells,), 3.0)
    mesh.cell_data[MU] = np.full((mesh.n_cells,), 1.0)

    melon.save(cfg.input, mesh)


if __name__ == "__main__":
    cherries.main(main)
