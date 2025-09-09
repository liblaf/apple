from pathlib import Path

import einops
import numpy as np
import pyvista as pv
from jaxtyping import Bool, Float

from liblaf import cherries, melon


class Config(cherries.BaseConfig):
    output: Path = cherries.output("10-input.vtu")

    lr: float = 0.02


def main(cfg: Config) -> None:
    surface: pv.PolyData = pv.Box(bounds=(0, 2, 0, 1, 0, 1))
    mesh: pv.UnstructuredGrid = melon.tetwild(surface, lr=cfg.lr)
    mesh.point_data["point-id"] = np.arange(mesh.n_points, dtype=np.int32)
    surface = mesh.extract_surface()  # pyright: ignore[reportAssignmentType]
    mesh.point_data["surface-mask"] = np.zeros((mesh.n_points,), dtype=np.bool_)
    mesh.point_data["surface-mask"][surface.point_data["point-id"]] = True

    edges: pv.PolyData = mesh.extract_all_edges()  # pyright: ignore[reportAssignmentType]
    edges = edges.compute_cell_sizes()  # pyright: ignore[reportAssignmentType]
    edge_length_min: float = np.min(edges.cell_data["Length"])

    dirichlet_mask: Bool[np.ndarray, "p J"] = np.zeros_like(mesh.points, dtype=np.bool_)
    dirichlet_values: Float[np.ndarray, "p J"] = np.zeros_like(mesh.points)
    dirichlet_mask[mesh.points[:, 0] < edge_length_min, :] = True
    mesh.point_data["dirichlet-mask"] = dirichlet_mask
    mesh.point_data["dirichlet-values"] = dirichlet_values

    cell_centers: pv.PolyData = mesh.cell_centers()  # pyright: ignore[reportAssignmentType]
    mesh.cell_data["muscle-0-mask"] = cell_centers.points[:, 2] < 0.5
    mesh.cell_data["muscle-1-mask"] = cell_centers.points[:, 2] > 0.5

    mesh.cell_data["activation"] = einops.repeat(
        np.asarray([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]),
        "i -> c i",
        c=mesh.n_cells,
    )
    mesh.cell_data["activation"][mesh.cell_data["muscle-1-mask"], 0] = 0.5  # pyright: ignore[reportArgumentType]
    mesh.cell_data["lambda"] = np.full((mesh.n_cells,), 3.0)
    mesh.cell_data["mu"] = np.full((mesh.n_cells,), 1.0)
    ic(mesh)
    melon.save(cfg.output, mesh)


if __name__ == "__main__":
    cherries.run(main)
