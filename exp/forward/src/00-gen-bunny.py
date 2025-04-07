from pathlib import Path

import numpy as np
import pyvista as pv
from jaxtyping import Bool, Float
from pyvista import examples

import liblaf.apple as apple  # noqa: PLR0402
from liblaf import cherries, grapes, melon


def gen_params(mesh: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    mesh.cell_data["E"] = np.full((mesh.n_cells,), 5e3)
    mesh.cell_data["nu"] = np.full((mesh.n_cells,), 0.4)
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


def gen_dirichlet(mesh: pv.UnstructuredGrid, y_threshold: float) -> pv.UnstructuredGrid:
    dirichlet_mask: Bool[np.ndarray, "V 3"] = np.zeros((mesh.n_points, 3), bool)
    dirichlet_mask[mesh.points[:, 1] < y_threshold, :] = True
    dirichlet_mask &= mesh.point_data["is-surface"][:, None]
    mesh.point_data["dirichlet-mask"] = dirichlet_mask
    mesh.point_data["dirichlet-values"] = np.zeros((mesh.n_points, 3))
    return mesh


def gen_initial_values(mesh: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    rng: np.random.Generator = np.random.default_rng()
    initial: Float[np.ndarray, "V 3"] = rng.uniform(
        -mesh.length / 3, mesh.length / 3, (mesh.n_points, 3)
    )
    dirichlet_mask: Bool[np.ndarray, "V 3"] = mesh.point_data["dirichlet-mask"]
    initial[dirichlet_mask] = mesh.point_data["dirichlet-values"][dirichlet_mask]
    mesh.point_data["initial"] = initial
    return mesh


class Config(cherries.BaseConfig):
    output: Path = grapes.find_project_dir() / "data/bunny/input.vtu"
    edge_length_fac: float = 0.05


def main(cfg: Config) -> None:
    surface: pv.PolyData = examples.download_bunny(load=True)  # pyright: ignore[reportAssignmentType]
    surface.triangulate(inplace=True)
    _, _, ymin, ymax, _, _ = surface.bounds
    y_threshold: float = ymin + 0.02 * (ymax - ymin)
    mesh: pv.UnstructuredGrid = apple.tetwild(
        surface, edge_length_fac=cfg.edge_length_fac
    )
    mesh = gen_params(mesh)
    mesh = gen_surface_mask(mesh)
    mesh = gen_dirichlet(mesh, y_threshold)
    mesh = gen_initial_values(mesh)
    melon.save(cfg.output, mesh)


if __name__ == "__main__":
    cherries.run(main)
