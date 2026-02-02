import numpy as np
import pyvista as pv

from liblaf import cherries, melon
from liblaf.apple.consts import DIRICHLET_MASK, DIRICHLET_VALUE, MUSCLE_FRACTION
from liblaf.apple.consts._array_names import GLOBAL_POINT_ID


class Config(cherries.BaseConfig):
    pass


def make_surface() -> pv.PolyData:
    surface: pv.PolyData = pv.Box((0.0, 10.0, 0.0, 1.0, 0.0, 10.0), quads=False)
    return surface


def make_muscle() -> pv.PolyData:
    muscle: pv.PolyData = pv.Box((0.0, 5.0, 0.0, 0.2, 4.0, 6.0), quads=False)
    return muscle


def make_tetmesh() -> pv.UnstructuredGrid:
    surface = make_surface()
    muscle = make_muscle()
    tetmesh: pv.UnstructuredGrid = melon.tetwild(surface, lr=0.01)
    tetmesh.cell_data[MUSCLE_FRACTION] = np.asarray(
        melon.tet.compute_volume_fraction(tetmesh, muscle)
    )

    EPS = 1e-3
    tetmesh.point_data[DIRICHLET_MASK] = np.broadcast_to(
        (
            (tetmesh.points[:, 0] < EPS)
            | (tetmesh.points[:, 0] > 10.0 - EPS)
            | (tetmesh.points[:, 2] < EPS)
            | (tetmesh.points[:, 2] > 10.0 - EPS)
        )[:, np.newaxis],
        (tetmesh.n_points, 3),
    )
    tetmesh.point_data[DIRICHLET_MASK][:, 1] |= tetmesh.points[:, 1] < EPS  # pyright: ignore[reportArgumentType]
    tetmesh.point_data[DIRICHLET_VALUE] = np.zeros((tetmesh.n_points, 3))

    tetmesh.point_data[GLOBAL_POINT_ID] = np.arange(tetmesh.n_points)

    return tetmesh


def make_subface(mesh: pv.UnstructuredGrid) -> pv.PolyData:
    surface: pv.PolyData = mesh.extract_surface()  # pyright: ignore[reportAssignmentType]
    subface: pv.PolyData = melon.tri.extract_points(
        surface, surface.points[:, 1] < 1e-3, adjacent_cells=False
    )
    return subface


def main(_cfg: Config) -> None:
    tetmesh: pv.UnstructuredGrid = make_tetmesh()
    subface: pv.PolyData = make_subface(tetmesh)
    melon.save(cherries.output("10-input.vtu"), tetmesh)
    melon.save(cherries.output("10-subface.vtp"), subface)


if __name__ == "__main__":
    cherries.main(main, profile="debug")
