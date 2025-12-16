from pathlib import Path

import numpy as np
import pyvista as pv
from jaxtyping import Integer

from liblaf import cherries, melon
from liblaf.apple import Forward, Model, ModelBuilder
from liblaf.apple.constants import ACTIVATION, POINT_ID
from liblaf.apple.warp import Phace


class Config(cherries.BaseConfig):
    low_res: Path = cherries.temp("20-low-res.vtu")
    high_res: Path = cherries.input("10-input.vtu")


def forward_mesh(mesh: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    builder: ModelBuilder = ModelBuilder()
    mesh = builder.assign_global_ids(mesh)
    builder.add_dirichlet(mesh)
    elastic: Phace = Phace.from_pyvista(
        mesh, id="elastic", requires_grad=["activation"]
    )
    builder.add_energy(elastic)
    model: Model = builder.finalize()

    forward: Forward = Forward(model=model)
    forward.step()

    mesh.point_data["Solution"] = forward.u_full[mesh.point_data[POINT_ID]]  # pyright: ignore[reportArgumentType]
    return mesh


def main(cfg: Config) -> None:
    low_res: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.low_res)
    high_res: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.high_res)
    low_res.cell_data[ACTIVATION] = low_res.cell_data["Activation000"]

    # ic(low_res.find_containing_cell(high_res.points), short_arrays=False)
    cell_id: Integer[np.ndarray] = low_res.find_closest_cell(high_res.points)  # pyright: ignore[reportAssignmentType]
    # ic(cell_id, short_arrays=False)
    high_res.point_data[ACTIVATION] = low_res.cell_data[ACTIVATION][cell_id]
    high_res = high_res.point_data_to_cell_data(pass_point_data=True)  # pyright: ignore[reportAssignmentType]

    high_res_sample: pv.UnstructuredGrid = high_res.sample(
        low_res, tolerance=1e-3, snap_to_closest_point=True
    )  # pyright: ignore[reportAssignmentType]
    ic(high_res, high_res_sample)
    high_res_sample = high_res_sample.point_data_to_cell_data(pass_point_data=True)  # pyright: ignore[reportAssignmentType]
    high_res.cell_data[ACTIVATION] = high_res_sample.cell_data[ACTIVATION]  # pyright: ignore[reportArgumentType]
    melon.save(cherries.temp("20-sampled.vtu"), high_res)

    # return

    low_res = forward_mesh(low_res)
    melon.save(cherries.temp("20-forward-low-res.vtu"), low_res)
    high_res = forward_mesh(high_res)
    melon.save(cherries.temp("20-forward-high-res.vtu"), high_res)


if __name__ == "__main__":
    cherries.main(main)
