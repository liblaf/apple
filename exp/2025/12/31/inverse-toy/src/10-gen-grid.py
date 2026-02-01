import einops
import numpy as np
import pyvista as pv
from environs import env
from jaxtyping import Array, Float, Integer

from liblaf import cherries, melon
from liblaf.apple.consts import (
    ACTIVATION,
    DIRICHLET_MASK,
    DIRICHLET_VALUE,
    LAMBDA,
    MU,
    MUSCLE_FRACTION,
)


class Config(cherries.BaseConfig):
    # lr=0.05 coarsen -> 4k-coarse
    # lr=0.05 -> 7k
    # lr=0.05 conform -> 7k-conform
    # lr=0.02 conform -> 102k-conform
    # lr=0.02 -> 121k
    lambda_: float = env.float("LAMBDA", 3.0)
    tetra_per_cell: int = env.int("TETRA_PER_CELL", 5)


def gen_muscles() -> pv.MultiBlock:
    muscles = pv.MultiBlock()
    muscles.append(pv.Box((0.4, 1.6, 0.2, 0.3, 0.0, 1.0), quads=False), "Muscle000")
    # muscles.append(pv.Box((0, 2, 0.7, 0.8, 0.2, 0.8), quads=False), "muscle-002")
    return muscles


def gen_grid(cfg: Config) -> pv.UnstructuredGrid:
    x = np.linspace(0.0, 2.0, 20 * 4 + 1)
    y = np.linspace(0.0, 0.5, 5 * 4 + 1)
    z = np.linspace(0.0, 1.0, 10 * 4 + 1)
    grid = pv.RectilinearGrid(x, y, z)
    mesh: pv.UnstructuredGrid = grid.to_tetrahedra(tetra_per_cell=cfg.tetra_per_cell)  # pyright: ignore[reportAssignmentType]
    mesh = melon.tet.fix_winding(mesh)
    return mesh


def main(cfg: Config) -> None:
    mesh: pv.UnstructuredGrid = gen_grid(cfg)
    muscles: pv.MultiBlock = gen_muscles()
    ic(mesh)

    mesh.point_data[DIRICHLET_MASK] = mesh.points[:, 1] < 1e-2
    mesh.point_data[DIRICHLET_VALUE] = np.zeros_like(mesh.points)

    mesh.cell_data[LAMBDA] = np.full((mesh.n_cells,), cfg.lambda_)
    mesh.cell_data[MU] = np.full((mesh.n_cells,), 1.0)

    mesh.cell_data[ACTIVATION] = np.zeros((mesh.n_cells, 6))
    mesh.cell_data[MUSCLE_FRACTION] = einops.repeat(
        np.asarray([1.0, 0.0, 0.0]), "i -> c i", c=mesh.n_cells
    )

    mesh.cell_data[MUSCLE_FRACTION] = np.zeros((mesh.n_cells,))
    mesh.cell_data["MuscleId"] = np.full((mesh.n_cells,), -1, np.int32)
    mesh.field_data["MuscleName"] = muscles.keys()
    for muscle_id, muscle in enumerate(muscles):
        fraction: Float[Array, " c"] = melon.tet.compute_volume_fraction(mesh, muscle)
        mesh.cell_data[MUSCLE_FRACTION] += fraction
        mesh.cell_data["MuscleId"][fraction > 0.5] = muscle_id  # pyright: ignore[reportArgumentType]

    muscle_ids: Integer[np.ndarray, " c"] = mesh.cell_data["MuscleId"]
    activation: Float[np.ndarray, "c 6"] = mesh.cell_data[ACTIVATION]
    gamma: float = 15.0
    activation[muscle_ids == 0, :3] = 1.0 - np.asarray(  # pyright: ignore[reportArgumentType]
        [gamma, 1.0 / np.sqrt(gamma), 1.0 / np.sqrt(gamma)]
    )
    mesh.cell_data[ACTIVATION] = activation

    suffix: str = f"-grid{cfg.tetra_per_cell}"
    suffix += f"-{round(mesh.n_cells / 1000)}k"
    melon.save(cherries.output(f"10-input{suffix}.vtu"), mesh)


if __name__ == "__main__":
    cherries.main(main, profile="debug")
