import numpy as np
import pyvista as pv
from jaxtyping import Bool, Integer


def fix_winding(mesh: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    mesh = mesh.compute_cell_sizes(length=False, area=False, volume=True)  # pyright: ignore[reportAssignmentType]
    mask: Bool[np.ndarray, " C"] = mesh.cell_data["Volume"] < 0
    cells: Integer[np.ndarray, "C 4"] = mesh.cells_dict[pv.CellType.TETRA]
    cells[mask] = cells[mask][:, [2, 1, 0, 3]]
    result = pv.UnstructuredGrid({pv.CellType.TETRA: cells}, mesh.points)
    result.copy_attributes(mesh)
    return result
