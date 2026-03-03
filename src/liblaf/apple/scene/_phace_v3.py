import numpy as np
import pyvista as pv

from liblaf.apple.consts import LAMBDA, MU, MUSCLE_FRACTION, SMAS_FRACTION
from liblaf.apple.model import Model, ModelBuilder
from liblaf.apple.warp import (
    WarpArap,
    WarpArapMuscle,
    WarpVolumePreservationDeterminant,
)


def build_phace_v3(mesh: pv.UnstructuredGrid) -> Model:
    builder = ModelBuilder()

    muscle_frac: np.ndarray = mesh.cell_data[MUSCLE_FRACTION]
    smas_frac: np.ndarray = mesh.cell_data[SMAS_FRACTION]
    aponeurosis_frac: np.ndarray = smas_frac - muscle_frac
    fat_frac: np.ndarray = 1.0 - smas_frac

    mesh: pv.UnstructuredGrid = builder.add_points(mesh)
    builder.add_dirichlet(mesh)

    mesh.cell_data["Fraction"] = fat_frac
    mesh.cell_data[MU] = np.full((mesh.n_cells,), 1.0)
    energy_fat: WarpArap = WarpArap.from_pyvista(mesh)
    builder.add_energy(energy_fat)

    mesh.cell_data["Fraction"] = aponeurosis_frac
    mesh.cell_data[MU] = np.full((mesh.n_cells,), 1.0e2)
    energy_aponeurosis: WarpArap = WarpArap.from_pyvista(mesh)
    builder.add_energy(energy_aponeurosis)

    mesh.cell_data["Fraction"] = muscle_frac
    mesh.cell_data[MU] = np.full((mesh.n_cells,), 1.0e2)
    energy_muscle: WarpArapMuscle = WarpArapMuscle.from_pyvista(mesh)
    builder.add_energy(energy_muscle)

    mesh.cell_data[LAMBDA] = fat_frac * 3.0 + smas_frac * 3.0e2
    energy_vol: WarpVolumePreservationDeterminant = (
        WarpVolumePreservationDeterminant.from_pyvista(mesh)
    )
    builder.add_energy(energy_vol)

    model: Model = builder.finalize()
    return model
