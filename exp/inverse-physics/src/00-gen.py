import numpy as np
import pyvista as pv
import typer
from jaxtyping import Bool, Float

import liblaf.grapes as grapes  # noqa: PLR0402
from liblaf import apple


def gen_surface_mask(mesh: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    mesh.point_data["PointID"] = np.arange(mesh.n_points)
    surface: pv.PolyData = mesh.extract_surface()  # pyright: ignore[reportAssignmentType]
    mesh.point_data["is_surface"] = np.zeros((mesh.n_points,), bool)
    mesh.point_data["is_surface"][surface.point_data["PointID"]] = True
    return mesh


def gen_params(mesh: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    centers: pv.PolyData = mesh.cell_centers()  # pyright: ignore[reportAssignmentType]
    xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
    mesh.cell_data["E"] = np.exp(
        np.interp(centers.points[:, 0], [xmin, xmax], [np.log(1), np.log(1e5)])
    )
    mesh.cell_data["nu"] = np.interp(centers.points[:, 1], [ymin, ymax], [0.0, 0.49])
    mesh.cell_data["lambda"], mesh.cell_data["mu"] = apple.constitution.E_nu_to_lame(
        mesh.cell_data["E"], mesh.cell_data["nu"]
    )
    mesh.cell_data["density"] = 1e3  # pyright: ignore[reportArgumentType]
    return mesh


def gen_fixed(mesh: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    xmin: float
    xmax: float
    xmin, xmax, _, _, _, _ = mesh.bounds
    left_mask: Bool[np.ndarray, " P"] = mesh.points[:, 0] < xmin + 1e-2
    right_mask: Bool[np.ndarray, " P"] = mesh.points[:, 0] > xmax - 1e-2
    fixed_disp: Float[np.ndarray, "P 3"] = np.zeros((mesh.n_points, 3))
    fixed_disp[left_mask] = [-0.1, 0.0, 0.0]
    fixed_disp[right_mask] = [0.1, 0.0, 0.0]
    mesh.point_data["fixed_mask"] = left_mask | right_mask
    mesh.point_data["fixed_disp"] = fixed_disp
    return mesh


def main() -> None:
    grapes.init_logging()
    surface: pv.PolyData = pv.Cylinder(radius=0.1, height=0.2)  # pyright: ignore[reportAssignmentType]
    mesh: pv.UnstructuredGrid = apple.tetwild(surface, edge_length_fac=0.1)
    mesh = gen_fixed(mesh)
    mesh = gen_params(mesh)
    mesh = gen_surface_mask(mesh)
    mesh.save("data/input.vtu")


if __name__ == "__main__":
    typer.run(main)
