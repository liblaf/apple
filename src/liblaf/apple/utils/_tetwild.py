import pytetwild
import pyvista as pv

from liblaf import grapes

from . import fix_winding


@grapes.timer()
def tetwild(
    surface: pv.PolyData, *, edge_length_fac: float = 0.05, optimize: bool = True
) -> pv.UnstructuredGrid:
    mesh: pv.UnstructuredGrid = pytetwild.tetrahedralize_pv(
        surface, edge_length_fac=edge_length_fac, optimize=optimize
    )
    mesh = fix_winding(mesh)
    return mesh
