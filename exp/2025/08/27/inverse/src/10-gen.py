from pathlib import Path

import einops
import numpy as np
import pyvista as pv
from jaxtyping import Bool, Float

from liblaf import cherries, melon


class Config(cherries.BaseConfig):
    output: Path = cherries.output("10-input.vtu")

    lr: float = 0.2


def main(cfg: Config) -> None:
    surface: pv.PolyData = pv.Box(bounds=(0, 2, 0, 1, 0, 1))
    mesh: pv.UnstructuredGrid = melon.tetwild(surface, lr=cfg.lr, coarsen=True)

    dirichlet_mask: Bool[np.ndarray, "p J"] = np.zeros_like(mesh.points, dtype=np.bool_)
    dirichlet_values: Float[np.ndarray, "p J"] = np.zeros_like(
        mesh.points, dtype=np.float32
    )
    dirichlet_mask[mesh.points[:, 0] < 1e-3, :] = True
    mesh.point_data["dirichlet-mask"] = dirichlet_mask
    mesh.point_data["dirichlet-values"] = dirichlet_values

    mesh.cell_data["activation"] = einops.repeat(
        np.diagflat([0.5, 1.0, 1.0]), "i j -> c i j", c=mesh.n_cells
    )
    mesh.cell_data["lambda"] = np.full((mesh.n_cells,), 3.0)
    mesh.cell_data["mu"] = np.full((mesh.n_cells,), 1.0)
    ic(mesh)
    melon.save(cfg.output, mesh)


if __name__ == "__main__":
    cherries.run(main)
