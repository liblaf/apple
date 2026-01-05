from pathlib import Path

import numpy as np
import pyvista as pv
from environs import env
from liblaf.peach.optim import PNCG

from liblaf import cherries, melon
from liblaf.apple.constants import LAMBDA, POINT_ID
from liblaf.apple.model import Forward, Model, ModelBuilder
from liblaf.apple.warp import Phace

SUFFIX: str = env.str("SUFFIX", default="-4k-coarsen")
LAMBDA_VALUE: float = env.float("LAMBDA", default=3.0)


class Config(cherries.BaseConfig):
    lambda_: float = LAMBDA_VALUE
    suffix: str = SUFFIX
    input: Path = cherries.input(f"10-input{SUFFIX}.vtu")
    output: Path = cherries.output(f"20-forward{SUFFIX}-{LAMBDA_VALUE}.vtu")


def build_model(mesh: pv.UnstructuredGrid) -> Forward:
    builder = ModelBuilder()
    mesh = builder.assign_global_ids(mesh)
    elastic: Phace = Phace.from_pyvista(mesh)
    builder.add_energy(elastic)
    model: Model = builder.finalize()
    forward = Forward(
        model,
        optimizer=PNCG(
            max_delta=0.15 * model.edges_length_mean,
            max_steps=2000,
            rtol=1e-5,
            rtol_primary=1e-5,
            stagnation_max_restarts=100,
        ),
    )
    return forward


def main(cfg: Config) -> None:
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.input)
    mesh.cell_data[LAMBDA] = np.full((mesh.n_cells,), cfg.lambda_)
    forward: Forward = build_model(mesh)
    forward.step()
    mesh.point_data["Solution"] = forward.u_full[mesh.point_data[POINT_ID]]  # pyright: ignore[reportArgumentType]
    melon.save(cfg.output, mesh)


if __name__ == "__main__":
    cherries.main(main)
