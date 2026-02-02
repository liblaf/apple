from pathlib import Path

import numpy as np
import pyvista as pv
from liblaf.peach.optim import Optimizer, ScipyOptimizer

from liblaf import cherries, melon
from liblaf.apple.consts import (
    ACTIVATION,
    DIRICHLET_MASK,
    DIRICHLET_VALUE,
    GLOBAL_POINT_ID,
    LAMBDA,
    MU,
)
from liblaf.apple.model import Forward, Model, ModelBuilder
from liblaf.apple.warp import WarpPhaceV2


class Config(cherries.BaseConfig):
    input: Path = cherries.input("10-input.vtu")
    subface: Path = cherries.input("20-forward-subface-base-act2.vtp")


def load_mesh(cfg: Config) -> pv.UnstructuredGrid:
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.input)
    subface: pv.PolyData = melon.load_polydata(cfg.subface)

    mesh.cell_data[ACTIVATION] = np.zeros((mesh.n_cells, 6))
    mesh.cell_data[LAMBDA] = np.full((mesh.n_cells,), 3.0)
    mesh.cell_data[MU] = np.full((mesh.n_cells,), 1.0)
    mesh.point_data[DIRICHLET_MASK][subface.point_data[GLOBAL_POINT_ID]] = True
    mesh.point_data[DIRICHLET_VALUE][subface.point_data[GLOBAL_POINT_ID]] = (
        subface.point_data["Solution"]
    )

    return mesh


def build(mesh: pv.UnstructuredGrid) -> Forward:
    builder = ModelBuilder()

    # mesh: pv.UnstructuredGrid = builder.add_points(mesh)
    builder.add_dirichlet(mesh)
    elastic: WarpPhaceV2 = WarpPhaceV2.from_pyvista(mesh)
    builder.add_energy(elastic)

    model: Model = builder.finalize()
    forward: Forward = Forward(model, optimizer=ScipyOptimizer(method="L-BFGS-B"))
    return forward


def main(cfg: Config) -> None:
    mesh: pv.UnstructuredGrid = load_mesh(cfg)
    forward: Forward = build(mesh)
    solution: Optimizer.Solution = forward.step()
    ic(solution)
    mesh.point_data["Solution"] = np.asarray(
        forward.u_full[mesh.point_data[GLOBAL_POINT_ID]]
    )
    melon.save(cherries.output("21-forward-whole-act2.vtu"), mesh)


if __name__ == "__main__":
    cherries.main(main, profile="debug")
