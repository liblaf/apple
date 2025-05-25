from pathlib import Path

import numpy as np
import pyvista as pv
from jaxtyping import Float

from liblaf import cherries, melon


class Config(cherries.BaseConfig):
    muscle: Path = cherries.data("10-muscle.vtp")
    tetgen: Path = cherries.data("10-tetgen.vtu")


def main(cfg: Config) -> None:
    cherries.log_input(cfg.muscle)
    muscle: pv.PolyData = melon.load_poly_data(cfg.muscle)
    cherries.log_output(cfg.tetgen)
    tetmesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.tetgen)

    ic(muscle.volume / tetmesh.volume)

    tetmesh = tetmesh.compute_cell_sizes()
    muscle_fraction: Float[np.ndarray, " C"] = tetmesh.cell_data["muscle-fraction"]
    volume: Float[np.ndarray, " C"] = tetmesh.cell_data["Volume"]
    occupied_volume: Float[np.ndarray, " C"] = np.sum(volume[muscle_fraction > 1e-9])
    ic(occupied_volume / tetmesh.volume)


if __name__ == "__main__":
    cherries.run(main)
