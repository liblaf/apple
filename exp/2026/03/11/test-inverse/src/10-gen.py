import numpy as np
import pyvista as pv

from liblaf import cherries, melon
from liblaf.apple.consts._array_names import DIRICHLET_MASK, DIRICHLET_VALUE


class Config(cherries.BaseConfig):
    pass


def main(_cfg: Config) -> None:
    surface: pv.PolyData = pv.Box((0.0, 1.0, 0.0, 1.0, 0.0, 1.0), quads=False)
    tetmesh: pv.UnstructuredGrid = melon.tetwild(surface, coarsen=True)
    tetmesh.point_data[DIRICHLET_MASK] = tetmesh.points[:, 0] < 1e-3
    tetmesh.point_data[DIRICHLET_VALUE] = np.zeros((tetmesh.n_points, 3))
    melon.save(cherries.output("10-input.vtu"), tetmesh)


if __name__ == "__main__":
    cherries.main(main, profile="debug")
