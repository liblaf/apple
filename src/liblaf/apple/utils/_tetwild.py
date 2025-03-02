import felupe
import pytetwild
import pyvista as pv


def tetwild(
    surface: pv.PolyData, *, edge_length_fac: float = 0.05, optimize: bool = True
) -> pv.UnstructuredGrid:
    mesh: pv.UnstructuredGrid = pytetwild.tetrahedralize_pv(
        surface, edge_length_fac=edge_length_fac, optimize=optimize
    )
    mesh = mesh.compute_cell_sizes(length=False, area=False, volume=True)  # pyright: ignore[reportAssignmentType]
    mesh_felupe = felupe.Mesh(mesh.points, mesh.cells_dict[pv.CellType.TETRA], "tetra")
    mesh_felupe = mesh_felupe.flip(mesh.cell_data["Volume"] < 0)
    return pv.UnstructuredGrid(
        {pv.CellType.TETRA: mesh_felupe.cells}, mesh_felupe.points
    )
