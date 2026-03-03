from pathlib import Path

import numpy as np
import pyvista as pv
from environs import env
from liblaf.peach.optim import Optimizer

from liblaf import cherries, melon
from liblaf.apple import scene
from liblaf.apple.consts import ACTIVATION, GLOBAL_POINT_ID, LAMBDA, MU
from liblaf.apple.model import Forward, Model


class Config(cherries.BaseConfig):
    activation: float = env.float("ACTIVATION", 2.0)
    input: Path = cherries.input("10-input.vtu")


def load_mesh(cfg: Config) -> pv.UnstructuredGrid:
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.input)

    mesh.cell_data[ACTIVATION] = np.zeros((mesh.n_cells, 6))
    mesh.cell_data[ACTIVATION][:, 0] = cfg.activation  # pyright: ignore[reportArgumentType]
    mesh.cell_data[LAMBDA] = np.full((mesh.n_cells,), 3.0)
    mesh.cell_data[MU] = np.full((mesh.n_cells,), 1.0)

    return mesh


def main(cfg: Config) -> None:
    mesh: pv.UnstructuredGrid = load_mesh(cfg)
    ic(mesh)
    model: Model = scene.build_phace_v3(mesh)
    forward: Forward = Forward(model)

    solution: Optimizer.Solution = forward.step()
    ic(solution)
    mesh.point_data["Solution"] = np.asarray(
        forward.u_full[mesh.point_data[GLOBAL_POINT_ID]]
    )
    melon.save(cherries.output(f"20-forward-whole-act{cfg.activation:.0f}.vtu"), mesh)


if __name__ == "__main__":
    cherries.main(main, profile="debug")
