import numpy as np
import pyvista as pv
from environs import env

from liblaf import cherries, melon
from liblaf.apple.consts import (
    DIRICHLET_MASK,
    DIRICHLET_VALUE,
    GLOBAL_POINT_ID,
    MUSCLE_FRACTION,
    SMAS_FRACTION,
)


class Config(cherries.BaseConfig):
    coarsen: bool = env.bool("COARSEN", False)
    lr: float = env.float("LR", 0.01)


def make_surface() -> pv.PolyData:
    surface: pv.PolyData = pv.Box((0.0, 10.0, 0.0, 1.0, 0.0, 10.0), quads=False)
    return surface


def make_muscle() -> pv.PolyData:
    muscle: pv.PolyData = pv.Box((0.0, 5.0, 0.4, 0.6, 4.0, 6.0), quads=False)
    return muscle


def make_smas() -> pv.PolyData:
    smas: pv.PolyData = pv.Box((0.0, 10.0, 0.4, 0.6, 0.0, 10.0), quads=False)
    return smas


def make_tetwild_input(
    surface: pv.PolyData, smas: pv.PolyData, muscle: pv.PolyData
) -> pv.PolyData:
    del muscle
    tetwild_input: pv.PolyData = pv.merge([surface, smas])
    return tetwild_input


def make_tetmesh(cfg: Config) -> pv.UnstructuredGrid:
    surface = make_surface()
    muscle = make_muscle()
    smas = make_smas()
    tetwild_input = make_tetwild_input(surface, smas, muscle)
    tetmesh: pv.UnstructuredGrid = melon.tetwild(
        tetwild_input, lr=cfg.lr, coarsen=cfg.coarsen
    )
    tetmesh.cell_data[MUSCLE_FRACTION] = np.asarray(
        melon.tet.compute_volume_fraction(tetmesh, muscle)
    )
    tetmesh.cell_data[SMAS_FRACTION] = np.asarray(
        melon.tet.compute_volume_fraction(tetmesh, smas)
    )
    # binarize the fractions to make the mesh conforming
    tetmesh.cell_data[SMAS_FRACTION] = np.where(
        tetmesh.cell_data[SMAS_FRACTION] > 0.5, 1.0, 0.0
    )

    eps = 1e-2
    tetmesh.point_data[DIRICHLET_MASK] = np.broadcast_to(
        (
            (tetmesh.points[:, 0] < eps)
            | (tetmesh.points[:, 0] > 10.0 - eps)
            | (tetmesh.points[:, 2] < eps)
            | (tetmesh.points[:, 2] > 10.0 - eps)
        )[:, np.newaxis],
        (tetmesh.n_points, 3),
    )
    tetmesh.point_data[DIRICHLET_MASK][:, 1] |= tetmesh.points[:, 1] < eps  # pyright: ignore[reportArgumentType]
    tetmesh.point_data[DIRICHLET_VALUE] = np.zeros((tetmesh.n_points, 3))

    tetmesh.point_data[GLOBAL_POINT_ID] = np.arange(tetmesh.n_points)

    return tetmesh


def make_subface(mesh: pv.UnstructuredGrid) -> pv.PolyData:
    surface: pv.PolyData = mesh.extract_surface()  # pyright: ignore[reportAssignmentType]
    subface: pv.PolyData = melon.tri.extract_points(
        surface, surface.points[:, 1] < 1e-3, adjacent_cells=False
    )
    return subface


def make_output_path(cfg: Config) -> str:
    suffix = "-smas46-muscle46"
    if cfg.coarsen:
        suffix += "-coarse"
    suffix += "-conform"
    return f"10-input{suffix}.vtu"


def main(cfg: Config) -> None:
    tetmesh: pv.UnstructuredGrid = make_tetmesh(cfg)
    # subface: pv.PolyData = make_subface(tetmesh)
    melon.save(cherries.output(make_output_path(cfg)), tetmesh)
    # melon.save(cherries.output("10-subface-conform.vtp"), subface)


if __name__ == "__main__":
    cherries.main(main, profile="debug")
