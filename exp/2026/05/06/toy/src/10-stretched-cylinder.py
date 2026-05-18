from pathlib import Path

import jax
import numpy as np
import pyvista as pv
import warp as wp

from liblaf import cherries, melon


class Config(cherries.BaseConfig):
    output: Path = cherries.output("10-stretched-cylinder.vtu")

    length: float = 1.0
    radius: float = 0.5
    stretch: float = 1.0
    lr: float = 0.12
    coarsen: bool = False
    E: float = 1.0
    nu: float = 0.49


def make_cylinder_surface(cfg: Config) -> pv.PolyData:
    surface: pv.PolyData = pv.Cylinder(
        center=(0.0, 0.0, 0.0),
        direction=(1.0, 0.0, 0.0),
        radius=cfg.radius,
        height=cfg.length,
    )
    surface.triangulate(inplace=True)
    surface.compute_normals(auto_orient_normals=True, inplace=True)
    return surface


def make_cylinder_mesh(cfg: Config) -> pv.UnstructuredGrid:
    surface = make_cylinder_surface(cfg)
    mesh: pv.UnstructuredGrid = melon.tetwild(surface, lr=cfg.lr, coarsen=cfg.coarsen)
    return melon.tet.fix_winding(mesh)


def add_boundary_conditions(mesh: pv.UnstructuredGrid, stretch: float) -> None:
    from liblaf.apple.common import FIXED_MASK, FIXED_VALUE

    x = mesh.points[:, 0]
    left = np.isclose(x, x.min())
    right = np.isclose(x, x.max())

    fixed_mask = np.zeros((mesh.n_points, 3), dtype=bool)
    fixed_value = np.zeros((mesh.n_points, 3), dtype=np.float64)
    fixed_mask[left | right, :] = True
    fixed_value[left, 0] = -0.5 * stretch
    fixed_value[right, 0] = 0.5 * stretch

    mesh.point_data[FIXED_MASK.vtk] = fixed_mask
    mesh.point_data[FIXED_VALUE.vtk] = fixed_value
    mesh.point_data["FixedSide"] = np.select([left, right], [-1, 1], default=0)


def solve(
    mesh: pv.UnstructuredGrid, E: float, nu: float
) -> tuple[pv.UnstructuredGrid, dict[str, float]]:
    from liblaf.apple.common import (
        GLOBAL_POINT_ID,
        LAMBDA,
        MU,
        NU,
        lame_converter,
    )
    from liblaf.apple.common import (
        E as YOUNG_MODULUS,
    )
    from liblaf.apple.forward import Forward, ModelBuilder
    from liblaf.apple.warp.fem import StableNeoHookean

    lambda_, mu = lame_converter(E, nu)
    mesh.cell_data[YOUNG_MODULUS.vtk] = np.full(mesh.n_cells, E, dtype=np.float64)
    mesh.cell_data[NU.vtk] = np.full(mesh.n_cells, nu, dtype=np.float64)
    mesh.cell_data[LAMBDA.vtk] = np.full(mesh.n_cells, float(lambda_), dtype=np.float64)
    mesh.cell_data[MU.vtk] = np.full(mesh.n_cells, float(mu), dtype=np.float64)

    builder = ModelBuilder()
    builder.add_vertices(mesh)
    builder.add_fixed(mesh)
    builder.add_potential(StableNeoHookean.from_pyvista(mesh))
    model = builder.finalize()
    forward = Forward(model)

    initial_energy = float(forward.problem.fun(forward.state))
    solution = forward.step()
    final_energy = float(forward.problem.fun(forward.state))

    displacement = np.asarray(forward.state.u)[mesh.point_data[GLOBAL_POINT_ID.vtk]]
    result = mesh.copy(deep=True)
    result.point_data["Displacement"] = displacement
    result.field_data["InitialEnergy"] = np.asarray([initial_energy])
    result.field_data["FinalEnergy"] = np.asarray([final_energy])
    result.field_data["OptimizerResult"] = np.asarray([solution.result.name])

    metrics = {
        "initial_energy": initial_energy,
        "final_energy": final_energy,
        "E": E,
        "nu": nu,
        "lambda": float(lambda_),
        "mu": float(mu),
        "n_points": float(mesh.n_points),
        "n_cells": float(mesh.n_cells),
    }
    return result, metrics


def main(cfg: Config) -> None:
    if jax.default_backend() != "gpu":
        msg = "This experiment uses Warp JAX FFI and needs JAX's active backend to be GPU."
        raise RuntimeError(msg)

    wp.config.mode = "release"
    wp.init()

    mesh = make_cylinder_mesh(cfg)
    add_boundary_conditions(mesh, cfg.stretch)
    result, metrics = solve(mesh, cfg.E, cfg.nu)

    melon.save(cfg.output, result)
    cherries.log_metrics(metrics)
    print(f"saved: {cfg.output}")
    print(
        "energy:",
        f"{metrics['initial_energy']:.6g}",
        "->",
        f"{metrics['final_energy']:.6g}",
    )
    print("mesh:", int(metrics["n_points"]), "points,", int(metrics["n_cells"]), "tets")


if __name__ == "__main__":
    cherries.main(main, profile="debug")
