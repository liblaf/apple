from pathlib import Path

import einops
import numpy as np
import pyvista as pv

from liblaf import cherries, melon


class Config(cherries.BaseConfig):
    output: Path = cherries.output("10-input.vtu")

    lr: float = 0.2


def main(cfg: Config) -> None:
    surface: pv.PolyData = pv.Box(bounds=(0, 2, 0, 1, 0, 1))
    mesh: pv.UnstructuredGrid = melon.tetwild(surface, lr=cfg.lr)
    mesh.cell_data["activation"] = einops.repeat(
        np.diagflat([0.5, 1.0, 1.0]), "i j -> c i j", c=mesh.n_cells
    )
    mesh.cell_data["lambda"] = np.full((mesh.n_cells,), 3.0)
    mesh.cell_data["mu"] = np.full((mesh.n_cells,), 1.0)
    melon.save(cfg.output, mesh)


if __name__ == "__main__":
    cherries.run(main)
