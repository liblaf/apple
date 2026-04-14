from pathlib import Path
from typing import cast

import attrs
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
    GLOBAL_POINT_ID,
    LAMBDA,
    MU,
    MUSCLE_FRACTION,
    SMAS_FRACTION,
)
from liblaf.apple.jax import JaxPointForce
from liblaf.apple.model import Forward, Model, ModelBuilder
from liblaf.apple.optim import PNCG
from liblaf.apple.warp import WarpNeoHookean, WarpNeoHookeanMuscle

SUFFIX: str = "-smas46-muscle46"


class Config(cherries.BaseConfig):
    activation: float = env.float("ACTIVATION", 2.0)
    force_scale: float = env.float("FORCE_SCALE", 1.0)
    lambda_value: float = env.float("LAMBDA_VALUE", 3.0)
    input: Path = cherries.input(f"10-input{SUFFIX}.vtu")


def load_mesh(cfg: Config) -> pv.UnstructuredGrid:
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.input)
    return mesh


def format_force_scale(force_scale: float) -> str:
    mantissa, exponent = f"{force_scale:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def format_lambda_value(lambda_value: float) -> str:
    if np.isclose(lambda_value, round(lambda_value)):
        return str(int(round(lambda_value)))
    mantissa, exponent = f"{lambda_value:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def apply_bottom_ext_force(
    mesh: pv.UnstructuredGrid, force_scale: float
) -> pv.UnstructuredGrid:
    bottom_mask: np.ndarray = mesh.points[:, 1] < 1e-3
    mesh.point_data[DIRICHLET_MASK][bottom_mask, :] = False

    surface: pv.PolyData = mesh.extract_surface(algorithm=None)
    surface = melon.tri.compute_point_area(surface)
    surface_bottom_mask: np.ndarray = surface.points[:, 1] < 1e-3
    bottom_indices: np.ndarray = surface.point_data[GLOBAL_POINT_ID][
        surface_bottom_mask
    ]
    mesh.point_data["Force"] = np.zeros_like(mesh.points)
    mesh.point_data["Force"][bottom_indices, 1] = (
        force_scale * surface.point_data["Area"][surface_bottom_mask]
    )
    return mesh


def build_phace_v3(
    mesh: pv.UnstructuredGrid,
    activation: float,
    force_scale: float,
    lambda_value: float,
) -> Model:
    builder = ModelBuilder()
    mesh = builder.add_points(mesh)
    mesh = apply_bottom_ext_force(mesh, force_scale)

    muscle_frac: np.ndarray = mesh.cell_data[MUSCLE_FRACTION]
    smas_frac: np.ndarray = mesh.cell_data[SMAS_FRACTION]
    aponeurosis_frac: np.ndarray = smas_frac - muscle_frac
    fat_frac: np.ndarray = 1.0 - smas_frac

    mesh.cell_data[ACTIVATION] = np.zeros((mesh.n_cells, 6))
    mesh.cell_data[ACTIVATION][smas_frac > 1e-3] = np.asarray(
        [activation - 1.0, 0.25 - 1.0, activation - 1.0, 0.0, 0.0, 0.0]
    )
    builder.add_dirichlet(mesh)

    mesh.cell_data["Fraction"] = fat_frac
    mesh.cell_data[MU] = np.full((mesh.n_cells,), 1.0)
    mesh.cell_data[LAMBDA] = np.full((mesh.n_cells,), lambda_value)
    builder.add_energy(WarpNeoHookean.from_pyvista(mesh))

    mesh.cell_data["Fraction"] = aponeurosis_frac
    mesh.cell_data[MU] = np.full((mesh.n_cells,), 1.0e2)
    mesh.cell_data[LAMBDA] = np.full((mesh.n_cells,), lambda_value * 1.0e2)
    builder.add_energy(WarpNeoHookeanMuscle.from_pyvista(mesh))

    mesh.cell_data["Fraction"] = muscle_frac
    mesh.cell_data[MU] = np.full((mesh.n_cells,), 1.0e2)
    mesh.cell_data[LAMBDA] = np.full((mesh.n_cells,), lambda_value * 1.0e2)
    builder.add_energy(
        WarpNeoHookeanMuscle.from_pyvista(
            mesh, requires_grad=("activation",), name="muscle"
        )
    )

    ext_force = JaxPointForce.from_pyvista(mesh)
    ext_force.name = "force"
    builder.add_energy(ext_force)
    return builder.finalize()


def main(cfg: Config) -> None:
    wp.init()
    mesh: pv.UnstructuredGrid = load_mesh(cfg)
    ic(mesh)
    model: Model = build_phace_v3(
        mesh, cfg.activation, cfg.force_scale, cfg.lambda_value
    )
    forward: Forward = Forward(model)
    optimizer = cast("PNCG", forward.optimizer)
    optimizer.convergence = attrs.evolve(
        optimizer.convergence,
        acceptable_relative_gradient_norm=jnp.asarray(1e-5),
        target_relative_gradient_norm=jnp.asarray(1e-6),
    )
    solution: Optimizer.Solution = forward.step()
    ic(solution)
    mesh.point_data["Solution"] = np.asarray(
        forward.u_full[mesh.point_data[GLOBAL_POINT_ID]]
    )
    melon.save(
        cherries.output(
            "20-forward"
            f"{SUFFIX}-prestrain-ext-force-neo-hookean-lambda-"
            f"{format_lambda_value(cfg.lambda_value)}-force-"
            f"{format_force_scale(cfg.force_scale)}.vtu"
        ),
        mesh,
    )


if __name__ == "__main__":
    cherries.main(main, profile="debug")
