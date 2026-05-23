from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
import torch
import warp as wp

from liblaf import cherries, melon

OUTPUT_STEM = "10-stretched-cylinder-no-collision"


class Config(cherries.BaseConfig):
    output_input: Path = cherries.output(f"{OUTPUT_STEM}-input.vtu")
    output: Path = cherries.output(f"{OUTPUT_STEM}.vtu")

    E: float = 1.0
    nu: float = 0.49
    lr: float = 0.01
    coarsen: bool = False
    x_min_displacement: float = -0.25
    x_max_displacement: float = 0.25


def configure_runtime() -> None:
    if not torch.cuda.is_available():
        msg = "This experiment uses Warp kernels through Torch and needs CUDA."
        raise RuntimeError(msg)
    torch.set_default_dtype(torch.float64)
    torch.set_default_device("cuda")
    wp.config.mode = "release"
    wp.init()


def make_cylinder_surface() -> pv.PolyData:
    surface: pv.PolyData = pv.Cylinder()
    surface.triangulate(inplace=True)
    surface.compute_normals(auto_orient_normals=True, inplace=True)
    return surface


def make_cylinder_mesh(cfg: Config) -> pv.UnstructuredGrid:
    surface = make_cylinder_surface()
    mesh: pv.UnstructuredGrid = melon.tetwild(surface, lr=cfg.lr, coarsen=cfg.coarsen)
    return melon.tet.fix_winding(mesh)


def add_boundary_conditions(mesh: pv.UnstructuredGrid, cfg: Config) -> None:
    from liblaf.apple.common import FIXED_MASK, FIXED_VALUE

    x = mesh.points[:, 0]
    x_min = np.isclose(x, x.min())
    x_max = np.isclose(x, x.max())

    fixed_mask = np.zeros((mesh.n_points, 3), dtype=bool)
    fixed_value = np.zeros((mesh.n_points, 3), dtype=np.float64)
    fixed_mask[x_min | x_max, :] = True
    fixed_value[x_min, 0] = cfg.x_min_displacement
    fixed_value[x_max, 0] = cfg.x_max_displacement

    mesh.point_data[FIXED_MASK.vtk] = fixed_mask
    mesh.point_data[FIXED_VALUE.vtk] = fixed_value
    mesh.point_data["FixedSide"] = np.select([x_min, x_max], [-1, 1], default=0)


def add_material(mesh: pv.UnstructuredGrid, cfg: Config) -> None:
    from liblaf.apple.common import LAMBDA, MU, NU, lame_converter
    from liblaf.apple.common import E as YOUNG_MODULUS

    lambda_, mu = lame_converter(cfg.E, cfg.nu)
    mesh.cell_data[YOUNG_MODULUS.vtk] = np.full(mesh.n_cells, cfg.E, dtype=np.float64)
    mesh.cell_data[NU.vtk] = np.full(mesh.n_cells, cfg.nu, dtype=np.float64)
    mesh.cell_data[LAMBDA.vtk] = np.full(mesh.n_cells, float(lambda_), dtype=np.float64)
    mesh.cell_data[MU.vtk] = np.full(mesh.n_cells, float(mu), dtype=np.float64)
    mesh.field_data["E"] = np.asarray([cfg.E])
    mesh.field_data["Nu"] = np.asarray([cfg.nu])
    mesh.field_data["Lambda"] = np.asarray([float(lambda_)])
    mesh.field_data["Mu"] = np.asarray([float(mu)])
    mesh.field_data["XMinDisplacement"] = np.asarray([cfg.x_min_displacement])
    mesh.field_data["XMaxDisplacement"] = np.asarray([cfg.x_max_displacement])
    mesh.field_data["WithCollision"] = np.asarray([0])


def build_model(mesh: pv.UnstructuredGrid):
    from liblaf.apple.forward import Forward, ModelBuilder
    from liblaf.apple.warp.fem import StableNeoHookean

    builder = ModelBuilder()
    builder.add_vertices(mesh)
    builder.add_fixed(mesh)
    builder.add_potential(StableNeoHookean.from_pyvista(mesh))
    return Forward(builder.finalize())


def to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def tensor_scalar(value: Any) -> float:
    return float(to_numpy(value).reshape(-1)[0])


def solve(
    mesh: pv.UnstructuredGrid,
) -> tuple[pv.UnstructuredGrid, dict[str, float | str]]:
    from liblaf.apple.common import GLOBAL_POINT_ID

    forward = build_model(mesh)

    initial_energy = tensor_scalar(forward.problem.fun(forward.state))
    solution = forward.step()
    final_energy = tensor_scalar(forward.problem.fun(forward.state))

    displacement = to_numpy(forward.state.u)[mesh.point_data[GLOBAL_POINT_ID.vtk]]
    result = mesh.copy(deep=True)
    result.point_data["Displacement"] = displacement
    result.point_data["DeformedPoint"] = result.points + displacement

    metrics: dict[str, float | str] = {
        "initial_energy": initial_energy,
        "final_energy": final_energy,
        "optimizer_steps": float(solution.state.step),
        "optimizer_result": solution.result.name,
        "n_points": float(mesh.n_points),
        "n_cells": float(mesh.n_cells),
        "n_fixed": float(forward.model.n_fixed),
        "n_free": float(forward.model.n_free),
    }
    for name, value in metrics.items():
        if isinstance(value, str):
            continue
        result.field_data[name] = np.asarray([value])
    return result, metrics


def main(cfg: Config) -> None:
    configure_runtime()

    mesh = make_cylinder_mesh(cfg)
    add_boundary_conditions(mesh, cfg)
    add_material(mesh, cfg)

    melon.save(cfg.output_input, mesh)
    result, metrics = solve(mesh)
    melon.save(cfg.output, result)

    cherries.log_metrics(
        {name: value for name, value in metrics.items() if not isinstance(value, str)}
    )
    print(metrics)
    print(f"saved: {cfg.output}")


if __name__ == "__main__":
    cherries.main(main, profile="debug")
