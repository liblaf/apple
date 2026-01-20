import einops
import numpy as np
import pyvista as pv
from environs import env
from jaxtyping import Array, Float, Integer

from liblaf import cherries, melon
from liblaf.apple.constants import (
    ACTIVATION,
    DIRICHLET_MASK,
    DIRICHLET_VALUE,
    LAMBDA,
    MU,
)


class Config(cherries.BaseConfig):
    # lr=0.05 coarsen -> 4k-coarse
    # lr=0.05 -> 7k
    # lr=0.05 conform -> 7k-conform
    # lr=0.02 conform -> 102k-conform
    # lr=0.02 -> 121k
    coarsen: bool = env.bool("COARSEN", False)
    conform: bool = env.bool("CONFORM", False)
    lr: float = env.float("LR", 0.02)

    lambda_: float = env.float("LAMBDA", 3.0)


def gen_muscles() -> pv.MultiBlock:
    muscles = pv.MultiBlock()
    muscles.append(pv.Box((0.4, 1.6, 0.2, 0.3, 0.0, 1.0), quads=False), "Muscle000")
    # muscles.append(pv.Box((0, 2, 0.7, 0.8, 0.2, 0.8), quads=False), "muscle-002")
    return muscles


def main(cfg: Config) -> None:
    surface: pv.PolyData = pv.Box((0.0, 2.0, 0.0, 0.5, 0.0, 1.0))
    muscles: pv.MultiBlock = gen_muscles()
    muscles.save(cherries.output("10-muscles.vtm"))

    tetwild_input: pv.PolyData
    if cfg.conform:
        muscle_combined: pv.PolyData = muscles.combine().extract_surface()
        tetwild_input = pv.merge([surface, muscle_combined])
    else:
        tetwild_input = surface
    mesh: pv.UnstructuredGrid = melon.tetwild(
        tetwild_input, lr=cfg.lr, coarsen=cfg.coarsen
    )
    surface = mesh.extract_surface()  # pyright: ignore[reportAssignmentType]
    ic(mesh, surface)

    mesh.point_data[DIRICHLET_MASK] = mesh.points[:, 1] < 1e-2
    mesh.point_data[DIRICHLET_VALUE] = np.zeros_like(mesh.points)

    mesh.cell_data[LAMBDA] = np.full((mesh.n_cells,), cfg.lambda_)
    mesh.cell_data[MU] = np.full((mesh.n_cells,), 1.0)

    mesh.cell_data[ACTIVATION] = np.zeros((mesh.n_cells, 6))
    mesh.cell_data["MuscleDirection"] = einops.repeat(
        np.asarray([1.0, 0.0, 0.0]), "i -> c i", c=mesh.n_cells
    )

    mesh.cell_data["MuscleFraction"] = np.zeros((mesh.n_cells,))
    mesh.cell_data["MuscleId"] = np.full((mesh.n_cells,), -1, np.int32)
    mesh.field_data["MuscleName"] = muscles.keys()
    for muscle_id, muscle in enumerate(muscles):
        fraction: Float[Array, " c"] = melon.tet.compute_volume_fraction(mesh, muscle)
        mesh.cell_data["MuscleFraction"] += fraction
        mesh.cell_data["MuscleId"][fraction > 0.5] = muscle_id  # pyright: ignore[reportArgumentType]

    muscle_ids: Integer[np.ndarray, " c"] = mesh.cell_data["MuscleId"]
    activation: Float[np.ndarray, "c 6"] = mesh.cell_data[ACTIVATION]
    gamma: float = 15.0
    activation[muscle_ids == 0, :3] = 1.0 - np.asarray(  # pyright: ignore[reportArgumentType]
        [gamma, 1.0 / np.sqrt(gamma), 1.0 / np.sqrt(gamma)]
    )
    mesh.cell_data[ACTIVATION] = activation

    suffix: str = f"-{round(mesh.n_cells / 1000)}k"
    if cfg.coarsen:
        suffix += "-coarse"
    if cfg.conform:
        suffix += "-conform"
    melon.save(cherries.output(f"10-input{suffix}.vtu"), mesh)


if __name__ == "__main__":
    cherries.main(main)
