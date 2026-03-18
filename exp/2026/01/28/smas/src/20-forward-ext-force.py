from pathlib import Path

import numpy as np
import pyvista as pv
from environs import env
from liblaf.peach.optim import Optimizer

from liblaf import cherries, melon
from liblaf.apple.consts import (
    ACTIVATION,
    DIRICHLET_MASK,
    GLOBAL_POINT_ID,
    LAMBDA,
    MU,
    MUSCLE_FRACTION,
    SMAS_FRACTION,
)
from liblaf.apple.jax import JaxPointForce
from liblaf.apple.model import Forward, Model, ModelBuilder
from liblaf.apple.warp import (
    WarpArap,
    WarpArapMuscle,
    WarpVolumePreservationDeterminant,
)

SUFFIX: str = "-smas46-muscle46"


class Config(cherries.BaseConfig):
    activation: float = env.float("ACTIVATION", 2.0)
    input: Path = cherries.input(f"10-input{SUFFIX}.vtu")


def load_mesh(cfg: Config) -> pv.UnstructuredGrid:
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.input)
    return mesh


def build_phace_v3(mesh: pv.UnstructuredGrid) -> Model:
    builder = ModelBuilder()
    mesh: pv.UnstructuredGrid = builder.add_points(mesh)

    mesh.point_data[DIRICHLET_MASK][mesh.points[:, 1] < 1e-3, :] = False
    surface: pv.PolyData = mesh.extract_surface()
    surface = melon.tri.compute_point_area(surface)
    bottom_indices: np.ndarray = surface.point_data[GLOBAL_POINT_ID][
        surface.points[:, 1] < 1e-3
    ]
    mesh.point_data["Force"] = np.zeros_like(mesh.points)
    mesh.point_data["Force"][bottom_indices, 1] = (
        1.0 * surface.point_data["Area"][surface.points[:, 1] < 1e-3]
    )

    muscle_frac: np.ndarray = mesh.cell_data[MUSCLE_FRACTION]
    smas_frac: np.ndarray = mesh.cell_data[SMAS_FRACTION]
    aponeurosis_frac: np.ndarray = smas_frac - muscle_frac
    fat_frac: np.ndarray = 1.0 - smas_frac

    mesh.cell_data[ACTIVATION] = np.zeros((mesh.n_cells, 6))
    mesh.cell_data[ACTIVATION][smas_frac > 1e-3] = np.asarray(
        [2.0 - 1.0, 0.25 - 1.0, 2.0 - 1.0, 0.0, 0.0, 0.0]
    )
    # mesh.cell_data[ACTIVATION][muscle_frac > 1e-3] = np.asarray(
    #     [5.0 - 1.0, 0.25 - 1.0, 2.0 - 1.0, 0.0, 0.0, 0.0]
    # )
    builder.add_dirichlet(mesh)

    mesh.cell_data["Fraction"] = fat_frac
    mesh.cell_data[MU] = np.full((mesh.n_cells,), 1.0)
    energy_fat: WarpArap = WarpArap.from_pyvista(mesh)
    builder.add_energy(energy_fat)

    mesh.cell_data["Fraction"] = aponeurosis_frac
    mesh.cell_data[MU] = np.full((mesh.n_cells,), 1.0e2)
    energy_aponeurosis: WarpArapMuscle = WarpArapMuscle.from_pyvista(mesh)
    builder.add_energy(energy_aponeurosis)

    mesh.cell_data["Fraction"] = muscle_frac
    mesh.cell_data[MU] = np.full((mesh.n_cells,), 1.0e2)
    energy_muscle: WarpArapMuscle = WarpArapMuscle.from_pyvista(
        mesh, requires_grad=("activation",), name="muscle"
    )
    builder.add_energy(energy_muscle)

    mesh.cell_data[LAMBDA] = fat_frac * 3.0 + smas_frac * 3.0e2
    energy_vol: WarpVolumePreservationDeterminant = (
        WarpVolumePreservationDeterminant.from_pyvista(mesh)
    )
    builder.add_energy(energy_vol)

    ext_force: JaxPointForce = JaxPointForce.from_pyvista(mesh)
    builder.add_energy(ext_force)

    model: Model = builder.finalize()
    return model


def main(cfg: Config) -> None:
    mesh: pv.UnstructuredGrid = load_mesh(cfg)
    ic(mesh)
    model: Model = build_phace_v3(mesh)
    forward: Forward = Forward(model)

    solution: Optimizer.Solution = forward.step()
    ic(solution)
    mesh.point_data["Solution"] = np.asarray(
        forward.u_full[mesh.point_data[GLOBAL_POINT_ID]]
    )
    melon.save(cherries.output(f"20-forward{SUFFIX}-prestrain-ext-force-1e0.vtu"), mesh)


if __name__ == "__main__":
    cherries.main(main, profile="debug")
