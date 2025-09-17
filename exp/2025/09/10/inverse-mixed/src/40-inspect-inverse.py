import numpy as np
import pyvista as pv

from liblaf import melon


def main() -> None:
    # mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(
    #     "data/30-inverse.vtu/30-inverse_000021.vtu"
    # )
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid("data/10-input.vtu")
    activation = mesh.cell_data["activation"]
    muscle_ids = mesh.cell_data["muscle-ids"]
    muscle_0_mask = muscle_ids == 0
    muscle_1_mask = muscle_ids == 1
    muscle_0_activation = np.mean(activation[muscle_0_mask], axis=0)
    muscle_1_activation = np.mean(activation[muscle_1_mask], axis=0)
    ic(muscle_0_activation, muscle_1_activation)


if __name__ == "__main__":
    main()
