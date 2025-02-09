import felupe
import numpy as np
import pytetwild
import pyvista as pv
import typer
from jaxtyping import Bool, Float, Integer

from liblaf import grapes


def main() -> None:
    grapes.init_logging()
    boundary: pv.PolyData = pv.Box(bounds=(0, 5, 0, 1, 0, 1), quads=False)  # pyright: ignore[reportAssignmentType]
    mesh: pv.UnstructuredGrid = pytetwild.tetrahedralize_pv(
        boundary, edge_length_fac=0.1
    )
    mesh = mesh.compute_cell_sizes(volume=True)  # pyright: ignore[reportAssignmentType]
    points: Float[np.ndarray, "P 3"]
    cells: Integer[np.ndarray, "C 4"]
    points, cells, _ = felupe.mesh.flip(
        mesh.points,
        mesh.cells_dict[pv.CellType.TETRA],
        cell_type="tetra",
        mask=mesh.cell_data["Volume"] < 0,
    )
    mesh = pv.UnstructuredGrid(
        pv.CellArray.from_regular_cells(cells).cells,
        np.full((cells.shape[0],), pv.CellType.TETRA),
        points,
    )
    fixed_mask: Bool[np.ndarray, " P"] = mesh.points[:, 0] < 1e-3
    fixed_disp: Float[np.ndarray, "P 3"] = np.zeros((mesh.n_points, 3))
    mesh.point_data["fixed_mask"] = fixed_mask
    mesh.point_data["fixed_disp"] = fixed_disp
    mesh.save("data/input.vtu")


if __name__ == "__main__":
    typer.run(main)
