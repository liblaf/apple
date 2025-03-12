from pathlib import Path

import einops
import felupe
import numpy as np
import pyvista as pv
import trimesh as tm
from jaxtyping import Float

from liblaf import cherries, melon


class Config(cherries.BaseConfig):
    activation: float = 3
    input: Path = Path("./data/00-raw.vtu")
    output: Path = Path("./data/01-input.vtu")


@cherries.main()
def main(cfg: Config) -> None:
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.input)
    mesh = fix_winding(mesh)
    muscle_fraction: Float[np.ndarray, "C M"] = einops.reduce(
        mesh.cell_data["muscle-fraction"], "C M -> C", "sum"
    )
    is_muscle: Float[np.ndarray, " C"] = muscle_fraction > 1e-3
    activation: Float[np.ndarray, "C 3 3"] = einops.repeat(
        np.identity(3), "i j -> C i j", C=mesh.n_cells
    )
    orientation: Float[np.ndarray, "C 3"] = mesh.cell_data["orientation"]
    Q: Float[np.ndarray, "C 3 3"] = einops.repeat(
        np.identity(3), "i j -> C i j", C=mesh.n_cells
    )
    for cell_id in range(mesh.n_cells):
        if is_muscle[cell_id]:
            rotation: Float[np.ndarray, "4 4"] = tm.geometry.align_vectors(
                orientation[cell_id], [1.0, 0.0, 0.0]
            )
            Q[cell_id] = rotation[0:3, 0:3]
            assert np.allclose(np.linalg.det(Q[cell_id]), 1)
            # assert np.allclose(Q[cell_id] @ [1.0, 0.0, 0.0], orientation[cell_id])
            assert np.allclose(Q[cell_id] @ orientation[cell_id], [1.0, 0.0, 0.0])
    stretch: Float[np.ndarray, " 3"] = np.asarray(
        [1.0 / cfg.activation, np.sqrt(cfg.activation), np.sqrt(cfg.activation)]
    )
    for cell_id in range(mesh.n_cells):
        if is_muscle[cell_id]:
            activation[cell_id] = Q[cell_id].T @ np.diagflat(stretch) @ Q[cell_id]
            # activation[cell_id] = np.diagflat(orientation[cell_id]) / cfg.activation
            if cell_id == 0:
                ic(orientation[cell_id])
                ic(Q[cell_id] @ orientation[cell_id])
                ic(np.diagflat(stretch) @ Q[cell_id] @ orientation[cell_id])
                ic(activation[cell_id] @ orientation[cell_id])
                ic(activation[cell_id] @ np.asarray([1.0, 0.0, 0.0]))
                ic(activation[cell_id] @ np.asarray([0.0, 1.0, 0.0]))
                ic(activation[cell_id] @ np.asarray([0.0, 0.0, 1.0]))
    mesh.cell_data["activation"] = activation
    melon.save(cfg.output, mesh)


def fix_winding(mesh: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    mesh = mesh.compute_cell_sizes(length=False, area=False, volume=True)  # pyright: ignore[reportAssignmentType]
    mesh_felupe = felupe.Mesh(mesh.points, mesh.cells_dict[pv.CellType.TETRA], "tetra")
    mesh_felupe = mesh_felupe.flip(mesh.cell_data["Volume"] < 0)
    mesh_new = pv.UnstructuredGrid(
        {pv.CellType.TETRA: mesh_felupe.cells}, mesh_felupe.points
    )
    mesh_new.copy_attributes(mesh)
    return mesh_new


def rotation_from_vectors(
    from_: Float[np.ndarray, "3"], to: Float[np.ndarray, "3"]
) -> Float[np.ndarray, "3 3"]:
    from_ = from_ / np.linalg.norm(from_)
    to = to / np.linalg.norm(to)
    v = np.cross(from_, to)
    c = np.dot(from_, to)
    s = np.linalg.norm(v)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.identity(3) + k + np.dot(k, k) * (1 - c) / (s**2)


if __name__ == "__main__":
    main(Config())
