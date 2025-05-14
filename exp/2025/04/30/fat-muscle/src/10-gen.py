from pathlib import Path

import einops
import numpy as np
import pyvista as pv
from jaxtyping import Bool, Float

import liblaf.melon as melon  # noqa: PLR0402
from liblaf import cherries


class Config(cherries.BaseConfig):
    n_samples: int = 100

    muscle: Path = cherries.data("10-muscle.vtp")
    tetgen: Path = cherries.data("10-tetgen.vtu")


def main(cfg: Config) -> None:
    surface: pv.PolyData = pv.Box((-1, 1, -0.4, 0.4, 0, 0.2))
    muscle: pv.PolyData = pv.Box((-1, 1, -0.3, 0.3, 0.08, 0.12))
    tetmesh: pv.UnstructuredGrid = melon.tetwild(surface)
    tetmesh.cell_data["muscle-fraction"] = 0.0
    for cid, cell in enumerate(tetmesh.cell):
        cell: pv.Cell
        barycentric: Float[np.ndarray, "N 3"] = melon.sample_barycentric_coords(
            (cfg.n_samples, 4)
        )
        samples: Float[np.ndarray, "N 3"] = melon.barycentric_to_points(
            einops.repeat(cell.points, "B D -> N B D", N=cfg.n_samples), barycentric
        )
        is_in: Bool[np.ndarray, " N"] = melon.triangle.contains(muscle, samples)
        tetmesh.cell_data["muscle-fraction"][cid] = (
            np.count_nonzero(is_in) / cfg.n_samples
        )

    melon.save(cfg.muscle, muscle)
    cherries.log_output(cfg.muscle)
    melon.save(cfg.tetgen, tetmesh)
    cherries.log_output(cfg.tetgen)


if __name__ == "__main__":
    cherries.run(main)
