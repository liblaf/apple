from pathlib import Path
from typing import cast

import jax.numpy as jnp
import numpy as np
import pyvista as pv
from environs import env
from liblaf.peach.optim import PNCG, Optimizer

from liblaf import cherries, melon
from liblaf.apple.consts import (
    ACTIVATION,
    DIRICHLET_MASK,
    DIRICHLET_VALUE,
    GLOBAL_POINT_ID,
    LAMBDA,
    MU,
    MUSCLE_FRACTION,
    SMAS_FRACTION,
)
from liblaf.apple.model import Forward, Model, ModelBuilder
from liblaf.apple.warp import (
    WarpArap,
    WarpArapMuscle,
    WarpVolumePreservationDeterminant,
)

SUFFIX: str = "-smas46-muscle46"


class Config(cherries.BaseConfig):
    activation: float = env.float("ACTIVATION", 2.0)
    arch_height: float = env.float("ARCH_HEIGHT", 1.0)
    input: Path = cherries.input(f"10-input{SUFFIX}.vtu")


def load_mesh(cfg: Config) -> pv.UnstructuredGrid:
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.input)
    return mesh


def apply_bottom_arch_dirichlet(
    mesh: pv.UnstructuredGrid, arch_height: float
) -> pv.UnstructuredGrid:
    bottom_mask: np.ndarray = mesh.points[:, 1] < 1e-3
    bottom_points: np.ndarray = mesh.points[bottom_mask]

    x: np.ndarray = bottom_points[:, 0]
    z: np.ndarray = bottom_points[:, 2]
    x_min, x_max = x.min(), x.max()
    z_min, z_max = z.min(), z.max()
    x_center = 0.5 * (x_min + x_max)
    z_center = 0.5 * (z_min + z_max)
    x_hat: np.ndarray = 2.0 * (x - x_center) / (x_max - x_min)
    z_hat: np.ndarray = 2.0 * (z - z_center) / (z_max - z_min)
    arch: np.ndarray = arch_height * (1.0 - x_hat**2) * (1.0 - z_hat**2)

    mesh.point_data[DIRICHLET_MASK][bottom_mask] = True
    mesh.point_data[DIRICHLET_VALUE][bottom_mask] = 0.0
    mesh.point_data[DIRICHLET_VALUE][bottom_mask, 1] = arch
    return mesh


def build_phace_v3(mesh: pv.UnstructuredGrid, arch_height: float) -> Model:
    builder = ModelBuilder()
    mesh: pv.UnstructuredGrid = builder.add_points(mesh)
    mesh = apply_bottom_arch_dirichlet(mesh, arch_height)

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

    model: Model = builder.finalize()
    return model


def main(cfg: Config) -> None:
    mesh: pv.UnstructuredGrid = load_mesh(cfg)
    ic(mesh)
    model: Model = build_phace_v3(mesh, cfg.arch_height)
    forward: Forward = Forward(model)
    optimizer = cast("PNCG", forward.optimizer)
    optimizer.rtol = jnp.asarray(1e-5)
    optimizer.rtol_primary = jnp.asarray(1e-10)

    solution: Optimizer.Solution = forward.step()
    ic(solution)
    mesh.point_data["Solution"] = np.asarray(
        forward.u_full[mesh.point_data[GLOBAL_POINT_ID]]
    )
    melon.save(
        cherries.output(f"20-forward{SUFFIX}-prestrain-bottom-dirichlet-arch.vtu"),
        mesh,
    )


if __name__ == "__main__":
    cherries.main(main, profile="debug")
