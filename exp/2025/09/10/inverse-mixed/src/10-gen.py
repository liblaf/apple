import itertools
from pathlib import Path

import einops
import joblib
import numpy as np
import pyvista as pv
from jaxtyping import Bool, Float, Integer

from liblaf import cherries, grapes, melon


class Config(cherries.BaseConfig):
    output: Path = cherries.output("10-input.vtu")

    coarsen: bool = True
    lr: float = 0.05

    samples_per_cell: int = 100


def gen_muscles() -> pv.MultiBlock:
    muscles = pv.MultiBlock()
    muscles.append(pv.Box((0, 2, 0.2, 0.8, 0.2, 0.3), quads=False), "muscle-001")
    muscles.append(pv.Box((0, 2, 0.2, 0.8, 0.7, 0.8), quads=False), "muscle-002")
    return muscles


def compute_muscle_fraction(
    cell: pv.Cell, muscles: pv.MultiBlock, samples_per_cell: int
) -> tuple[int, float]:
    muscle_id: int = -1
    max_muscle_fraction: float = 0.0
    muscle_fraction: float = 0.0
    for mid, muscle in enumerate(muscles):
        barycentric: Float[np.ndarray, "S 4"] = melon.sample_barycentric_coords(
            (samples_per_cell, 4)
        )
        samples: Float[np.ndarray, "S 3"] = melon.barycentric_to_points(
            cell.points[np.newaxis, ...], barycentric
        )
        contains: Bool[np.ndarray, " S"] = melon.tri.contains(muscle, samples)
        fraction: float = float(np.count_nonzero(contains) / samples_per_cell)
        muscle_fraction += fraction
        if fraction > max_muscle_fraction:
            max_muscle_fraction = fraction
            muscle_id = mid
    return muscle_id, muscle_fraction


def main(cfg: Config) -> None:
    surface: pv.PolyData = pv.Box((0, 2, 0, 1, 0, 1))
    mesh: pv.UnstructuredGrid = melon.tetwild(surface, lr=cfg.lr, coarsen=cfg.coarsen)
    surface = mesh.extract_surface()  # pyright: ignore[reportAssignmentType]
    ic(mesh, surface)

    mesh.point_data["dirichlet-mask"] = mesh.points[:, 0] < 1e-2
    mesh.point_data["dirichlet-values"] = np.zeros_like(mesh.points)

    mesh.cell_data["lambda"] = 3.0  # pyright: ignore[reportArgumentType]
    mesh.cell_data["mu"] = 1.0  # pyright: ignore[reportArgumentType]

    muscles: pv.MultiBlock = gen_muscles()
    muscles.save("muscles.vtm")
    mesh.cell_data["activation"] = einops.repeat(
        np.asarray([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]), "i -> c i", c=mesh.n_cells
    )
    mesh.cell_data["muscle-direction"] = einops.repeat(
        np.asarray([1.0, 0.0, 0.0]), "i -> c i", c=mesh.n_cells
    )
    mesh.cell_data["active-fraction"] = np.zeros((mesh.n_cells,))
    mesh.cell_data["muscle-ids"] = np.full((mesh.n_cells,), -1, np.int32)
    mesh.field_data["muscle-names"] = muscles.keys()

    with joblib.parallel_config(prefer="processes"):
        for cid, (muscle_id, muscle_fraction) in enumerate(
            grapes.parallel(
                compute_muscle_fraction,
                mesh.cell,
                itertools.repeat(muscles),
                itertools.repeat(cfg.samples_per_cell),
                return_as="generator",
                total=mesh.n_cells,
            )
        ):
            mesh.cell_data["muscle-ids"][cid] = muscle_id
            mesh.cell_data["active-fraction"][cid] += muscle_fraction

    muscle_ids: Integer[np.ndarray, " c"] = mesh.cell_data["muscle-ids"]
    activation: Float[np.ndarray, " c 6"] = mesh.cell_data["activation"]
    activation[muscle_ids == 0, :3] = [1.0 / 0.5, np.sqrt(0.5), np.sqrt(0.5)]  # pyright: ignore[reportArgumentType]
    activation[muscle_ids == 1, :3] = [1.0 / 2.0, np.sqrt(2.0), np.sqrt(2.0)]  # pyright: ignore[reportArgumentType]
    mesh.cell_data["activation"] = activation
    melon.save(cfg.output, mesh)


if __name__ == "__main__":
    cherries.main(main)
