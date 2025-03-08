import numpy as np
import pyvista as pv
from jaxtyping import Bool, Float

import liblaf.apple as apple  # noqa: PLR0402
import liblaf.grapes as grapes  # noqa: PLR0402


def gen_mesh() -> pv.PolyData:
    mesh: pv.PolyData = pv.Plane(direction=[0, 1, 0])  # pyright: ignore[reportAssignmentType]
    mesh.triangulate(inplace=True)
    return mesh
    # clustering = pyacvd.Clustering(mesh)
    # clustering.subdivide(3)
    # clustering.fast_cluster(100)
    # return clustering.create_mesh()


def gen_fixed(mesh: pv.PolyData, eps: float = 5e-2) -> pv.PolyData:
    xmin: float
    xmax: float
    zmin: float
    zmax: float
    xmin, xmax, _, _, zmin, zmax = mesh.bounds
    fixed_mask: Bool[np.ndarray, " P"] = (
        (mesh.points[:, 0] < xmin + eps * mesh.length)
        | (mesh.points[:, 0] > xmax - eps * mesh.length)
        | (mesh.points[:, 2] < zmin + eps * mesh.length)
        | (mesh.points[:, 2] > zmax - eps * mesh.length)
    )
    fixed_disp: Float[np.ndarray, "P 3"] = np.zeros((mesh.n_points, 3))
    mesh.point_data["fixed_mask"] = fixed_mask
    mesh.point_data["fixed_disp"] = fixed_disp
    return mesh


def gen_material(mesh: pv.PolyData) -> pv.PolyData:
    centers: pv.PolyData = mesh.cell_centers()  # pyright: ignore[reportAssignmentType]
    xmin, xmax, _, _, zmin, zmax = mesh.bounds
    mesh.cell_data["E"] = np.exp(
        np.interp(centers.points[:, 0], [xmin, xmax], [np.log(1e3), np.log(1e5)])
    )
    # mesh.cell_data["E"] = 1e3  # pyright: ignore[reportArgumentType]
    mesh.cell_data["nu"] = np.interp(centers.points[:, 2], [zmin, zmax], [0.0, 0.49])
    # mesh.cell_data["nu"] = 0.4  # pyright: ignore[reportArgumentType]
    mesh.cell_data["lmbda"], mesh.cell_data["mu"] = apple.constitution.E_nu_to_lame(
        mesh.cell_data["E"], mesh.cell_data["nu"]
    )
    mesh.cell_data["density"] = 1e3  # pyright: ignore[reportArgumentType]
    mesh.cell_data["thickness"] = 1e-3  # pyright: ignore[reportArgumentType]
    mesh.cell_data["pre_strain"] = 1.0  # pyright: ignore[reportArgumentType]
    return mesh


def main() -> None:
    grapes.init_logging()
    mesh: pv.PolyData = gen_mesh()
    mesh = gen_fixed(mesh)
    mesh = gen_material(mesh)
    mesh.save("data/input.vtp")


if __name__ == "__main__":
    main()
