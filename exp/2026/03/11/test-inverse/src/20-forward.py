from pathlib import Path

import numpy as np
import pyvista as pv
from liblaf.peach.optim import Optimizer

from liblaf import cherries, melon
from liblaf.apple.consts import ACTIVATION, GLOBAL_POINT_ID, LAMBDA, MU
from liblaf.apple.model import Forward, Model, ModelBuilder
from liblaf.apple.warp import (
    WarpArapMuscleOld,
    WarpVolumePreservationDeterminant,
)


class Config(cherries.BaseConfig):
    input: Path = cherries.input("10-input.vtu")


def load_mesh(cfg: Config) -> pv.UnstructuredGrid:
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.input)
    return mesh


def build_model(mesh: pv.UnstructuredGrid) -> Model:
    builder = ModelBuilder()

    mesh: pv.UnstructuredGrid = builder.add_points(mesh)
    mesh.cell_data[ACTIVATION] = np.zeros((mesh.n_cells, 6))
    mesh.cell_data[ACTIVATION] = np.tile(
        np.asarray([-0.5, 0.0, 0.0, 0.0, 0.0, 0.0]), (mesh.n_cells, 1)
    )
    builder.add_dirichlet(mesh)

    mesh.cell_data[MU] = np.full((mesh.n_cells,), 1.0)
    energy_muscle: WarpArapMuscleOld = WarpArapMuscleOld.from_pyvista(mesh)
    builder.add_energy(energy_muscle)

    mesh.cell_data[LAMBDA] = np.full((mesh.n_cells,), 3.0)
    energy_vol: WarpVolumePreservationDeterminant = (
        WarpVolumePreservationDeterminant.from_pyvista(mesh)
    )
    builder.add_energy(energy_vol)

    model: Model = builder.finalize()
    return model


def main(cfg: Config) -> None:
    mesh: pv.UnstructuredGrid = load_mesh(cfg)
    ic(mesh)
    model: Model = build_model(mesh)
    forward: Forward = Forward(model)

    solution: Optimizer.Solution = forward.step()
    ic(solution)
    mesh.point_data["Solution"] = np.asarray(
        forward.u_full[mesh.point_data[GLOBAL_POINT_ID]]
    )
    melon.save(cherries.output("20-forward.vtu"), mesh)


if __name__ == "__main__":
    cherries.main(main, profile="debug")
