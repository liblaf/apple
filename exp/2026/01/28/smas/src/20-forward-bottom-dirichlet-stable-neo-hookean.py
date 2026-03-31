from pathlib import Path
from typing import cast

import jax.numpy as jnp
import numpy as np
import pyvista as pv
import warp as wp
from environs import env
from liblaf.peach.optim import Optimizer

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
from liblaf.apple.optim import PNCG
from liblaf.apple.warp import WarpStableNeoHookean, WarpStableNeoHookeanMuscle

SUFFIX: str = "-smas46-muscle46"


class Config(cherries.BaseConfig):
    activation: float = env.float("ACTIVATION", 2.0)
    arch_height: float = env.float("ARCH_HEIGHT", 2.0)
    input: Path = cherries.input(f"10-input{SUFFIX}.vtu")


def load_mesh(cfg: Config) -> pv.UnstructuredGrid:
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.input)
    return mesh


def compute_arch_profile(
    mesh: pv.UnstructuredGrid,
    arch_height: float,
    points: np.ndarray | None = None,
) -> np.ndarray:
    if points is None:
        points = mesh.points
    bottom_mask: np.ndarray = mesh.points[:, 1] < 1e-3
    bottom_points: np.ndarray = mesh.points[bottom_mask]

    x_min, x_max = bottom_points[:, 0].min(), bottom_points[:, 0].max()
    z_min, z_max = bottom_points[:, 2].min(), bottom_points[:, 2].max()
    x_center = 0.5 * (x_min + x_max)
    z_center = 0.5 * (z_min + z_max)
    x_hat: np.ndarray = 2.0 * (points[:, 0] - x_center) / (x_max - x_min)
    z_hat: np.ndarray = 2.0 * (points[:, 2] - z_center) / (z_max - z_min)
    arch: np.ndarray = arch_height * (1.0 - x_hat**2) * (1.0 - z_hat**2)
    return np.clip(arch, 0.0, None)


def apply_bottom_arch_dirichlet(
    mesh: pv.UnstructuredGrid, arch_height: float
) -> pv.UnstructuredGrid:
    bottom_mask: np.ndarray = mesh.points[:, 1] < 1e-3
    arch: np.ndarray = compute_arch_profile(mesh, arch_height, mesh.points[bottom_mask])

    mesh.point_data[DIRICHLET_MASK][bottom_mask] = True
    mesh.point_data[DIRICHLET_VALUE][bottom_mask] = 0.0
    mesh.point_data[DIRICHLET_VALUE][bottom_mask, 1] = arch
    return mesh


def build_phace_v3(mesh: pv.UnstructuredGrid, arch_height: float) -> Model:
    builder = ModelBuilder()
    mesh = builder.add_points(mesh)
    mesh = apply_bottom_arch_dirichlet(mesh, arch_height)

    muscle_frac: np.ndarray = mesh.cell_data[MUSCLE_FRACTION]
    smas_frac: np.ndarray = mesh.cell_data[SMAS_FRACTION]
    aponeurosis_frac: np.ndarray = smas_frac - muscle_frac
    fat_frac: np.ndarray = 1.0 - smas_frac

    mesh.cell_data[ACTIVATION] = np.zeros((mesh.n_cells, 6))
    mesh.cell_data[ACTIVATION][smas_frac > 1e-3] = np.asarray(
        [2.0 - 1.0, 0.25 - 1.0, 2.0 - 1.0, 0.0, 0.0, 0.0]
    )
    builder.add_dirichlet(mesh)

    mesh.cell_data["Fraction"] = fat_frac
    mesh.cell_data[MU] = np.full((mesh.n_cells,), 1.0)
    mesh.cell_data[LAMBDA] = np.full((mesh.n_cells,), 3.0)
    builder.add_energy(WarpStableNeoHookean.from_pyvista(mesh))

    mesh.cell_data["Fraction"] = aponeurosis_frac
    mesh.cell_data[MU] = np.full((mesh.n_cells,), 1.0e2)
    mesh.cell_data[LAMBDA] = np.full((mesh.n_cells,), 3.0e2)
    builder.add_energy(WarpStableNeoHookeanMuscle.from_pyvista(mesh))

    mesh.cell_data["Fraction"] = muscle_frac
    mesh.cell_data[MU] = np.full((mesh.n_cells,), 1.0e2)
    mesh.cell_data[LAMBDA] = np.full((mesh.n_cells,), 3.0e2)
    builder.add_energy(
        WarpStableNeoHookeanMuscle.from_pyvista(
            mesh, requires_grad=("activation",), name="muscle"
        )
    )

    return builder.finalize()


def initialize_parabolic_guess(
    model: Model, mesh: pv.UnstructuredGrid, arch_height: float
) -> None:
    u_full: np.ndarray = np.zeros((model.n_points, model.dim), dtype=mesh.points.dtype)
    u_full[mesh.point_data[GLOBAL_POINT_ID], 1] = compute_arch_profile(
        mesh, arch_height
    )
    model.u_free = model.dirichlet.get_free(jnp.asarray(u_full))


def main(cfg: Config) -> None:
    wp.init()
    mesh: pv.UnstructuredGrid = load_mesh(cfg)
    ic(mesh)
    model: Model = build_phace_v3(mesh, cfg.arch_height)
    initialize_parabolic_guess(model, mesh, cfg.arch_height)
    forward = Forward(model)
    optimizer = cast("PNCG", forward.optimizer)
    optimizer.rtol = jnp.asarray(1e-5)
    optimizer.rtol_primary = jnp.asarray(1e-6)

    solution: Optimizer.Solution = forward.step()
    ic(solution)
    mesh.point_data["Solution"] = np.asarray(
        forward.u_full[mesh.point_data[GLOBAL_POINT_ID]]
    )
    melon.save(
        cherries.output(
            f"20-forward{SUFFIX}-prestrain-bottom-dirichlet-arch-stable-neo-hookean.vtu"
        ),
        mesh,
    )


if __name__ == "__main__":
    cherries.main(main, profile="debug")
