import itertools
from pathlib import Path

import numpy as np
import pyvista as pv
import scipy.spatial
from jaxtyping import Bool, Float, Integer

import liblaf.apple as apple  # noqa: PLR0402
from liblaf import cherries, grapes, melon


def gen_dirichlet(mesh: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    kdtree = scipy.spatial.KDTree(mesh.points)
    corners: Float[np.ndarray, "8 3"] = np.asarray(
        list(itertools.product([xmin, xmax], [ymin, ymax], [zmin, zmax]))
    )
    indices: Integer[np.ndarray, " 8"]
    _distance, indices = kdtree.query(corners)  # pyright: ignore[reportAssignmentType]
    dirichlet_mask: Bool[np.ndarray, "V 3"] = np.zeros((mesh.n_points, 3), bool)
    dirichlet_mask[indices, :] = True
    mesh.point_data["dirichlet-mask"] = dirichlet_mask
    mesh.point_data["dirichlet-values"] = np.zeros((mesh.n_points, 3))
    return mesh


def gen_initial_values(mesh: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    rng: np.random.Generator = np.random.default_rng()
    initial: Float[np.ndarray, "V 3"] = rng.uniform(
        -mesh.length, mesh.length, (mesh.n_points, 3)
    )
    dirichlet_mask: Bool[np.ndarray, "V 3"] = mesh.point_data["dirichlet-mask"]
    initial[dirichlet_mask] = mesh.point_data["dirichlet-values"][dirichlet_mask]
    mesh.point_data["initial"] = initial
    return mesh


def gen_params(mesh: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    mesh.cell_data["E"] = np.full((mesh.n_cells,), 1e3)
    mesh.cell_data["nu"] = np.full((mesh.n_cells,), 0.3)
    mesh.cell_data["lambda"], mesh.cell_data["mu"] = apple.constitution.E_nu_to_lame(
        mesh.cell_data["E"], mesh.cell_data["nu"]
    )
    mesh.cell_data["density"] = 1e3  # pyright: ignore[reportArgumentType]
    return mesh


def gen_surface_mask(mesh: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    mesh.point_data["PointIds"] = np.arange(mesh.n_points)
    surface: pv.PolyData = mesh.extract_surface()  # pyright: ignore[reportAssignmentType]
    mesh.point_data["is-surface"] = np.zeros((mesh.n_points,), bool)
    mesh.point_data["is-surface"][surface.point_data["PointIds"]] = True
    return mesh


DATASET_DIR: Path = Path(
    "~/mnt/SeaDrive/My Libraries/dataset/2024-09-13-template-head/sculptor/"
).expanduser()


class Config(cherries.BaseConfig):
    face: Path = DATASET_DIR / "face.ply"
    cranium: Path = DATASET_DIR / "cranium.ply"
    mandible: Path = DATASET_DIR / "mandible.ply"
    output: Path = grapes.find_project_dir() / "data/input.vtu"
    edge_length_fac: float = 0.05


def main(cfg: Config) -> None:
    face: pv.PolyData = melon.load_poly_data(cfg.face)
    cranium: pv.PolyData = melon.load_poly_data(cfg.cranium)
    mandible: pv.PolyData = melon.load_poly_data(cfg.mandible)
    skull: pv.PolyData = pv.merge([cranium, mandible])
    skull.flip_normals()
    surface: pv.PolyData = pv.merge([face, skull])
    surface.triangulate(inplace=True)
    mesh: pv.UnstructuredGrid = apple.tetwild(
        surface, edge_length_fac=cfg.edge_length_fac
    )
    mesh = gen_dirichlet(mesh)
    mesh = gen_initial_values(mesh)
    mesh = gen_params(mesh)
    mesh = gen_surface_mask(mesh)
    ic(mesh)
    melon.save(cfg.output, mesh)


if __name__ == "__main__":
    cherries.run(main)
