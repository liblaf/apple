from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pyvista as pv
from environs import env

from liblaf import cherries, melon
from liblaf.apple.consts import ACTIVATION, LAMBDA, MU, MUSCLE_FRACTION, SMAS_FRACTION
from liblaf.apple.model import Forward, Model, ModelBuilder
from liblaf.apple.warp import (
    WarpArap,
    WarpArapMuscle,
    WarpVolumePreservationDeterminant,
)


class Config(cherries.BaseConfig):
    activation: float = env.float("ACTIVATION", 2.0)
    input: Path = cherries.input("10-input-coarse.vtu")


def load_mesh(cfg: Config) -> pv.UnstructuredGrid:
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.input)

    mesh.cell_data[ACTIVATION] = np.zeros((mesh.n_cells, 6))
    mesh.cell_data[ACTIVATION][:, 0] = cfg.activation  # pyright: ignore[reportArgumentType]
    mesh.cell_data[LAMBDA] = np.full((mesh.n_cells,), 3.0)
    mesh.cell_data[MU] = np.full((mesh.n_cells,), 1.0)

    return mesh


def build(mesh: pv.UnstructuredGrid) -> Forward:
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
    mesh.cell_data[MU] = np.full((mesh.n_cells,), 1.0e0)
    energy_aponeurosis: WarpArap = WarpArap.from_pyvista(mesh)
    builder.add_energy(energy_aponeurosis)

    mesh.cell_data["Fraction"] = muscle_frac
    mesh.cell_data[MU] = np.full((mesh.n_cells,), 1.0e0)
    energy_muscle: WarpArapMuscle = WarpArapMuscle.from_pyvista(mesh)
    builder.add_energy(energy_muscle)

    mesh.cell_data[LAMBDA] = fat_frac * 3.0 + smas_frac * 3.0e0
    energy_vol: WarpVolumePreservationDeterminant = (
        WarpVolumePreservationDeterminant.from_pyvista(mesh)
    )
    builder.add_energy(energy_vol)

    model: Model = builder.finalize()
    # forward: Forward = Forward(model, optimizer=ScipyOptimizer(method="L-BFGS-B"))
    forward: Forward = Forward(model)
    return forward


def main(cfg: Config) -> None:
    mesh: pv.UnstructuredGrid = load_mesh(cfg)
    ic(mesh)
    forward: Forward = build(mesh)

    H_diag = forward.model.hess_diag(forward.state)
    print(H_diag)

    def matvec(v: np.ndarray) -> np.ndarray:
        v_full = forward.model.dirichlet.to_full(jnp.asarray(v))
        output = forward.model.hess_prod(forward.state, v_full)
        output /= H_diag
        output = forward.model.dirichlet.get_free(output)
        return np.asarray(output)

    import pylops

    hess_op = pylops.FunctionOperator(
        matvec, matvec, forward.model.n_free, forward.model.n_free, dtype=np.float64
    )
    hess: np.ndarray = hess_op.todense()
    ic(np.linalg.cond(hess))
    # print(hess)
    # print(np.linalg.eigvals(hess))
    # print(np.linalg.eigvals(hess).max())
    # print(np.linalg.eigvals(hess).min())
    # ic(np.max(np.abs(hess)) / np.min(np.abs(hess)))


if __name__ == "__main__":
    cherries.main(main, profile="debug")
