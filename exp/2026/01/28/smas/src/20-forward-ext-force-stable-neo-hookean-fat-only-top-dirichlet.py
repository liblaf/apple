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
    DIRICHLET_MASK,
    DIRICHLET_VALUE,
    GLOBAL_POINT_ID,
    LAMBDA,
    MU,
)
from liblaf.apple.jax import JaxPointForce, Region
from liblaf.apple.model import Forward, Model, ModelBuilder
from liblaf.apple.optim import PNCG
from liblaf.apple.warp import WarpStableNeoHookean

SUFFIX: str = "-smas46-muscle46-coarse"


class Config(cherries.BaseConfig):
    force_scale: float = env.float("FORCE_SCALE", 1.0)
    input: Path = cherries.input(f"10-input{SUFFIX}.vtu")
    lambda_value: float = env.float("LAMBDA_VALUE", 3.0)
    mu_value: float = env.float("MU_VALUE", 1.0)
    surface_tolerance: float = env.float("SURFACE_TOLERANCE", 1.0e-2)


def load_mesh(cfg: Config) -> pv.UnstructuredGrid:
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.input)
    return mesh


def format_force_scale(force_scale: float) -> str:
    mantissa, exponent = f"{force_scale:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def format_material_value(value: float) -> str:
    if np.isclose(value, round(value)):
        return str(round(value))
    mantissa, exponent = f"{value:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def add_volume_change_ratio(
    mesh: pv.UnstructuredGrid, u_full: jnp.ndarray
) -> pv.UnstructuredGrid:
    region = Region.from_pyvista(mesh, grad=True)
    deformation_gradient: np.ndarray = np.asarray(region.deformation_gradient(u_full))
    mesh.cell_data["VolumeChangeRatio"] = np.linalg.det(deformation_gradient[:, 0])
    return mesh


def apply_top_dirichlet(
    mesh: pv.UnstructuredGrid, surface_tolerance: float
) -> pv.UnstructuredGrid:
    top_y = mesh.bounds[3]
    top_mask: np.ndarray = mesh.points[:, 1] >= top_y - surface_tolerance

    mesh.point_data[DIRICHLET_MASK] = np.zeros((mesh.n_points, 3), dtype=bool)
    mesh.point_data[DIRICHLET_MASK][top_mask] = True
    mesh.point_data[DIRICHLET_VALUE] = np.zeros(
        (mesh.n_points, 3), dtype=mesh.points.dtype
    )
    return mesh


def apply_bottom_ext_force(
    mesh: pv.UnstructuredGrid, force_scale: float, surface_tolerance: float
) -> pv.UnstructuredGrid:
    bottom_y = mesh.bounds[2]
    surface: pv.PolyData = mesh.extract_surface(algorithm=None)
    surface = melon.tri.compute_point_area(surface)
    surface_bottom_mask: np.ndarray = surface.points[:, 1] <= (
        bottom_y + surface_tolerance
    )
    bottom_indices: np.ndarray = surface.point_data[GLOBAL_POINT_ID][
        surface_bottom_mask
    ]

    mesh.point_data["Force"] = np.zeros_like(mesh.points)
    mesh.point_data["Force"][bottom_indices, 1] = (
        force_scale * surface.point_data["Area"][surface_bottom_mask]
    )
    return mesh


def build_pure_fat_model(
    mesh: pv.UnstructuredGrid,
    force_scale: float,
    lambda_value: float,
    mu_value: float,
    surface_tolerance: float,
) -> Model:
    builder = ModelBuilder()
    mesh = builder.add_points(mesh)
    mesh = apply_top_dirichlet(mesh, surface_tolerance)
    mesh = apply_bottom_ext_force(mesh, force_scale, surface_tolerance)
    builder.add_dirichlet(mesh)

    mesh.cell_data["Fraction"] = np.ones((mesh.n_cells,))
    mesh.cell_data[MU] = np.full((mesh.n_cells,), mu_value)
    mesh.cell_data[LAMBDA] = np.full((mesh.n_cells,), lambda_value)
    builder.add_energy(WarpStableNeoHookean.from_pyvista(mesh))

    ext_force = JaxPointForce.from_pyvista(mesh)
    ext_force.name = "force"
    builder.add_energy(ext_force)
    return builder.finalize()


def main(cfg: Config) -> None:
    wp.init()
    mesh: pv.UnstructuredGrid = load_mesh(cfg)
    ic(mesh)
    model: Model = build_pure_fat_model(
        mesh,
        cfg.force_scale,
        cfg.lambda_value,
        cfg.mu_value,
        cfg.surface_tolerance,
    )
    forward: Forward = Forward(model)
    optimizer = cast("PNCG", forward.optimizer)
    optimizer.convergence = attrs.evolve(
        optimizer.convergence,
        acceptable_relative_gradient_norm=jnp.asarray(1e-3),
        target_relative_gradient_norm=jnp.asarray(1e-5),
    )

    solution: Optimizer.Solution = forward.step()
    ic(solution)
    mesh.point_data["Solution"] = np.asarray(
        forward.u_full[mesh.point_data[GLOBAL_POINT_ID]]
    )
    mesh = add_volume_change_ratio(mesh, forward.u_full)
    melon.save(
        cherries.output(
            "20-forward"
            f"{SUFFIX}-fat-only-top-dirichlet-ext-force-stable-neo-hookean-mu-"
            f"{format_material_value(cfg.mu_value)}-lambda-"
            f"{format_material_value(cfg.lambda_value)}-force-"
            f"{format_force_scale(cfg.force_scale)}.vtu"
        ),
        mesh,
    )


if __name__ == "__main__":
    cherries.main(main, profile="debug")
