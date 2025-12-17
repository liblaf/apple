from pathlib import Path

import pyvista as pv

from liblaf import cherries, melon
from liblaf.apple import Forward, Model, ModelBuilder
from liblaf.apple.constants import ACTIVATION, POINT_ID
from liblaf.apple.warp import Phace


class Config(cherries.BaseConfig):
    low_res: Path = cherries.temp(
        "20-inverse-adam-123k.vtu.d/20-inverse-adam-123k_000150.vtu"
    )
    high_res: Path = cherries.input("10-input-515k.vtu")


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
    low_res.cell_data[ACTIVATION] = low_res.cell_data[ACTIVATION]

    high_res = melon.transfer_tet_cell(low_res, high_res, data=[ACTIVATION])

    low_res = forward_mesh(low_res)
    melon.save(cherries.temp("20-coarse.vtu"), low_res)
    high_res = forward_mesh(high_res)
    melon.save(cherries.temp("20-coarse-to-fine.vtu"), high_res)

    target_tet: pv.UnstructuredGrid = high_res.extract_surface()  # pyright: ignore[reportAssignmentType]
    target: pv.PolyData = melon.tri.extract_points(
        target_tet, target_tet.point_data["IsFace"]
    )
    target.warp_by_vector("Expression000", inplace=True)
    melon.save(cherries.temp("20-coarse-to-fine-target.vtp"), target)


if __name__ == "__main__":
    cherries.main(main)
