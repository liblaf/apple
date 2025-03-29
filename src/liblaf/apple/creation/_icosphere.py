import pyvista as pv

from liblaf import apple


def icosphere() -> pv.UnstructuredGrid:
    surface: pv.PolyData = pv.Icosphere()
    mesh: pv.UnstructuredGrid = apple.tetwild(surface)
    return mesh
