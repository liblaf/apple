from pathlib import Path

import numpy as np
import pyvista as pv
from environs import env

from liblaf import cherries, melon
from liblaf.apple.consts import GLOBAL_POINT_ID, SMAS_FRACTION

FORCE = "Force"
IS_SMAS_TET = "IsSmasTet"
IS_SMAS_BOTTOM_FORCE_POINT = "IsSmasBottomForcePoint"
SMAS_BOTTOM_PROJECTED_POINT_AREA = "SmasBottomProjectedPointArea"
SUFFIX: str = "-smas46-muscle46"


class Config(cherries.BaseConfig):
    input: Path = cherries.input(f"10-input{SUFFIX}.vtu")
    output: Path = cherries.output(f"11-input{SUFFIX}-smas-bottom-force.vtu")
    smas_threshold: float = env.float("SMAS_THRESHOLD", 1.0e-2)
    y_threshold: float = env.float("Y_THRESHOLD", 0.5)
    boundary_tolerance: float = env.float("BOUNDARY_TOLERANCE", 1.0e-6)


def load_mesh(cfg: Config) -> pv.UnstructuredGrid:
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.input)
    return mesh


def extract_smas_tets(
    mesh: pv.UnstructuredGrid, smas_threshold: float
) -> pv.UnstructuredGrid:
    smas_mask: np.ndarray = np.asarray(mesh.cell_data[SMAS_FRACTION] > smas_threshold)
    if not np.any(smas_mask):
        raise ValueError(
            f"no SMAS tetrahedra found with {SMAS_FRACTION} > {smas_threshold:g}"
        )
    smas_tets: pv.UnstructuredGrid = mesh.extract_cells(smas_mask)
    smas_tets.cell_data[IS_SMAS_TET] = np.ones(smas_tets.n_cells, dtype=bool)
    smas_tets.point_data[GLOBAL_POINT_ID] = np.arange(smas_tets.n_points)
    return smas_tets


def compute_bottom_force(
    smas_tets: pv.UnstructuredGrid, y_threshold: float, boundary_tolerance: float
) -> np.ndarray:
    surface: pv.PolyData = smas_tets.extract_surface(algorithm=None)
    faces: np.ndarray = surface.regular_faces
    face_points: np.ndarray = np.asarray(surface.points[faces])
    face_x: np.ndarray = face_points[:, :, 0]
    face_y: np.ndarray = face_points[:, :, 1]
    face_z: np.ndarray = face_points[:, :, 2]

    x_min, x_max, _, _, z_min, z_max = smas_tets.bounds
    is_below_y_threshold: np.ndarray = np.all(face_y < y_threshold, axis=1)
    is_on_x_boundary: np.ndarray = np.all(
        np.isclose(face_x, x_min, atol=boundary_tolerance), axis=1
    ) | np.all(
        np.isclose(face_x, x_max, atol=boundary_tolerance), axis=1
    )
    is_on_z_boundary: np.ndarray = np.all(
        np.isclose(face_z, z_min, atol=boundary_tolerance), axis=1
    ) | np.all(
        np.isclose(face_z, z_max, atol=boundary_tolerance), axis=1
    )
    bottom_cell_mask: np.ndarray = (
        is_below_y_threshold & ~is_on_x_boundary & ~is_on_z_boundary
    )
    if not np.any(bottom_cell_mask):
        raise ValueError("no bottom SMAS surface triangles found")

    edge_01_x: np.ndarray = face_x[:, 1] - face_x[:, 0]
    edge_01_z: np.ndarray = face_z[:, 1] - face_z[:, 0]
    edge_02_x: np.ndarray = face_x[:, 2] - face_x[:, 0]
    edge_02_z: np.ndarray = face_z[:, 2] - face_z[:, 0]
    projected_cell_area: np.ndarray = 0.5 * np.abs(
        edge_01_x * edge_02_z - edge_01_z * edge_02_x
    )
    bottom_faces: np.ndarray = faces[bottom_cell_mask]
    bottom_projected_cell_area: np.ndarray = projected_cell_area[bottom_cell_mask]

    point_area: np.ndarray = np.zeros(surface.n_points)
    np.add.at(
        point_area,
        bottom_faces.reshape(-1),
        np.repeat(bottom_projected_cell_area / 3.0, 3),
    )

    surface_point_id: np.ndarray = np.asarray(surface.point_data[GLOBAL_POINT_ID])
    force: np.ndarray = np.zeros((smas_tets.n_points, 3), dtype=smas_tets.points.dtype)
    force[surface_point_id, 1] = point_area
    return force


def apply_bottom_force(
    mesh: pv.UnstructuredGrid, force: np.ndarray
) -> pv.UnstructuredGrid:
    mesh.point_data[FORCE] = force
    mesh.point_data[SMAS_BOTTOM_PROJECTED_POINT_AREA] = force[:, 1].copy()
    mesh.point_data[IS_SMAS_BOTTOM_FORCE_POINT] = force[:, 1] > 0.0
    return mesh


def main(cfg: Config) -> None:
    mesh: pv.UnstructuredGrid = load_mesh(cfg)
    smas_tets = extract_smas_tets(mesh, cfg.smas_threshold)
    force = compute_bottom_force(
        smas_tets, cfg.y_threshold, cfg.boundary_tolerance
    )
    smas_tets = apply_bottom_force(smas_tets, force)
    melon.save(cfg.output, smas_tets)


if __name__ == "__main__":
    cherries.main(main, profile="debug")
