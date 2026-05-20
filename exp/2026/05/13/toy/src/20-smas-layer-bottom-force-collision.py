from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
import pyvista as pv
import torch
import warp as wp

from liblaf import cherries, melon

OUTPUT_STEM = "20-smas-layer-bottom-force-collision"
BODY_BOUNDS = (0.0, 1.0, 0.0, 0.1, 0.0, 1.0)
SMAS_BOUNDS = (0.0, 1.0, 0.04, 0.06, 0.0, 1.0)
FAT_FRACTION = "FatFraction"
SMAS_FRACTION = "SmasFraction"
SMAS_ACTIVE_TOL = 1.0e-6


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    output_input: Path = cherries.output(f"{OUTPUT_STEM}-input.vtu")
    output: Path = cherries.output(f"{OUTPUT_STEM}.vtu")

    E_fat: float = 1.0
    nu: float = 0.49
    smas_stiffness_ratio: float = 1.0e3
    lr: float = 0.02
    coarsen: bool = False
    bottom_pressure: float = 2.6
    smas_prestrain: tuple[float, float, float, float, float, float] = (
        0.8,
        1.0,
        0.8,
        0.0,
        0.0,
        0.0,
    )
    collision_stiffness_scale: float = 0.1
    optimizer_max_steps: int = 3000
    rtol_primary: float = 5.0e-4
    rtol_secondary: float = 5.0e-4
    load_steps: int = 4
    boundary_atol: float = 1.0e-2
    bottom_atol: float = 1.0e-6

    @property
    def E_smas(self) -> float:
        return self.smas_stiffness_ratio * self.E_fat

    @property
    def collision_stiffness(self) -> float:
        return self.collision_stiffness_scale * self.E_fat


def configure_runtime() -> None:
    if not torch.cuda.is_available():
        msg = "This experiment uses Warp kernels through Torch and needs CUDA."
        raise RuntimeError(msg)
    torch.set_default_dtype(torch.float64)
    torch.set_default_device("cuda")
    wp.config.mode = "release"
    wp.init()


def make_box(bounds: tuple[float, float, float, float, float, float]) -> pv.PolyData:
    box = pv.Box(bounds, quads=False)
    box.triangulate(inplace=True)
    box.compute_normals(auto_orient_normals=True, inplace=True)
    return box


def make_body_mesh(cfg: Config) -> pv.UnstructuredGrid:
    surface = make_box(BODY_BOUNDS)
    smas_layer = make_box(SMAS_BOUNDS)
    mesh: pv.UnstructuredGrid = melon.tetwild(surface, lr=cfg.lr, coarsen=cfg.coarsen)
    mesh = melon.tet.fix_winding(mesh)

    smas_fraction = np.asarray(melon.tet.compute_volume_fraction(mesh, smas_layer))
    mesh.cell_data[SMAS_FRACTION] = smas_fraction
    mesh.cell_data[FAT_FRACTION] = np.clip(1.0 - smas_fraction, 0.0, 1.0)
    mesh.cell_data["SmasActive"] = (smas_fraction > SMAS_ACTIVE_TOL).astype(np.int8)
    return mesh


def add_boundary_conditions(mesh: pv.UnstructuredGrid, cfg: Config) -> None:
    from liblaf.apple.common import FIXED_MASK, FIXED_VALUE

    x = mesh.points[:, 0]
    z = mesh.points[:, 2]
    side = (
        np.isclose(x, BODY_BOUNDS[0], atol=cfg.boundary_atol)
        | np.isclose(x, BODY_BOUNDS[1], atol=cfg.boundary_atol)
        | np.isclose(z, BODY_BOUNDS[4], atol=cfg.boundary_atol)
        | np.isclose(z, BODY_BOUNDS[5], atol=cfg.boundary_atol)
    )

    fixed_mask = np.zeros((mesh.n_points, 3), dtype=bool)
    fixed_value = np.zeros((mesh.n_points, 3), dtype=np.float64)
    fixed_mask[side, :] = True

    mesh.point_data[FIXED_MASK.vtk] = fixed_mask
    mesh.point_data[FIXED_VALUE.vtk] = fixed_value
    mesh.point_data["FixedSide"] = side.astype(np.int8)


def ensure_global_point_ids(mesh: pv.UnstructuredGrid) -> None:
    from liblaf.apple.common import GLOBAL_POINT_ID

    if GLOBAL_POINT_ID.vtk not in mesh.point_data:
        mesh.point_data[GLOBAL_POINT_ID.vtk] = np.arange(mesh.n_points)


def add_bottom_force(mesh: pv.UnstructuredGrid, cfg: Config) -> pv.PolyData:
    from liblaf.apple.common import FIXED_MASK, FORCE, GLOBAL_POINT_ID

    ensure_global_point_ids(mesh)

    surface: pv.PolyData = mesh.extract_surface(algorithm=None)
    surface = melon.tri.compute_point_area(surface)

    bottom = surface.points[:, 1] <= BODY_BOUNDS[2] + cfg.bottom_atol
    free_bottom = bottom & ~surface.point_data[FIXED_MASK.vtk][:, 0]
    force = np.zeros((surface.n_points, 3), dtype=np.float64)
    force[free_bottom, 1] = (
        cfg.bottom_pressure * surface.point_data["Area"][free_bottom]
    )
    surface.point_data[FORCE.vtk] = force
    surface.point_data["LoadedBottom"] = free_bottom.astype(np.int8)

    mesh.point_data[FORCE.vtk] = np.zeros((mesh.n_points, 3), dtype=np.float64)
    global_ids = surface.point_data[GLOBAL_POINT_ID.vtk][free_bottom]
    mesh.point_data[FORCE.vtk][global_ids] = force[free_bottom]
    mesh.point_data["LoadedBottom"] = np.zeros(mesh.n_points, dtype=np.int8)
    mesh.point_data["LoadedBottom"][global_ids] = 1

    return surface


def add_reference_fields(mesh: pv.UnstructuredGrid, cfg: Config) -> None:
    from liblaf.apple.common import FORCE

    smas_fraction = np.asarray(mesh.cell_data[SMAS_FRACTION])
    fat_fraction = np.asarray(mesh.cell_data[FAT_FRACTION])
    mesh.cell_data["EffectiveYoungModulus"] = (
        cfg.E_fat * fat_fraction + cfg.E_smas * smas_fraction
    )
    mesh.cell_data["Nu"] = np.full(mesh.n_cells, cfg.nu, dtype=np.float64)

    total_force = float(np.asarray(mesh.point_data[FORCE.vtk])[:, 1].sum())
    mesh.field_data["Efat"] = np.asarray([cfg.E_fat])
    mesh.field_data["Esmas"] = np.asarray([cfg.E_smas])
    mesh.field_data["Nu"] = np.asarray([cfg.nu])
    mesh.field_data["TetWildLr"] = np.asarray([cfg.lr])
    mesh.field_data["BottomPressure"] = np.asarray([cfg.bottom_pressure])
    mesh.field_data["TotalBottomForce"] = np.asarray([total_force])
    mesh.field_data["CollisionStiffness"] = np.asarray([cfg.collision_stiffness])
    mesh.field_data["OptimizerMaxSteps"] = np.asarray([cfg.optimizer_max_steps])
    mesh.field_data["RtolPrimary"] = np.asarray([cfg.rtol_primary])
    mesh.field_data["RtolSecondary"] = np.asarray([cfg.rtol_secondary])
    mesh.field_data["LoadSteps"] = np.asarray([cfg.load_steps])
    mesh.field_data["SmasLayerBounds"] = np.asarray([SMAS_BOUNDS])
    mesh.field_data["SmasPrestrain"] = np.asarray([cfg.smas_prestrain])
    mesh.field_data["SmasActivation"] = activation_from_prestrain(cfg.smas_prestrain)[
        np.newaxis, :
    ]


def add_material(
    mesh: pv.UnstructuredGrid, *, E: float, nu: float, fraction: np.ndarray
) -> None:
    from liblaf.apple.common import FRACTION, LAMBDA, MU, NU, lame_converter
    from liblaf.apple.common import E as YOUNG_MODULUS

    lambda_, mu = lame_converter(E, nu)
    mesh.cell_data[YOUNG_MODULUS.vtk] = np.full(mesh.n_cells, E, dtype=np.float64)
    mesh.cell_data[NU.vtk] = np.full(mesh.n_cells, nu, dtype=np.float64)
    mesh.cell_data[LAMBDA.vtk] = np.full(mesh.n_cells, float(lambda_), dtype=np.float64)
    mesh.cell_data[MU.vtk] = np.full(mesh.n_cells, float(mu), dtype=np.float64)
    mesh.cell_data[FRACTION.vtk] = fraction


def activation_from_prestrain(
    prestrain: tuple[float, float, float, float, float, float],
) -> np.ndarray:
    prestrain_array = np.asarray(prestrain, dtype=np.float64)
    activation = prestrain_array.copy()
    activation[:3] -= 1.0
    return activation


def add_smas_activation(mesh: pv.UnstructuredGrid, cfg: Config) -> np.ndarray:
    from liblaf.apple.common import ACTIVATION

    smas = np.asarray(mesh.cell_data[SMAS_FRACTION]) > SMAS_ACTIVE_TOL
    activation_value = activation_from_prestrain(cfg.smas_prestrain)

    activation = np.zeros((mesh.n_cells, 6), dtype=np.float64)
    activation[smas] = activation_value
    mesh.cell_data[ACTIVATION.vtk] = activation
    mesh.cell_data["SmasActivation"] = activation
    return activation_value


def build_model(body: pv.UnstructuredGrid, cfg: Config):
    from liblaf.apple.collision import CollisionBuilder
    from liblaf.apple.forward import Forward, ModelBuilder
    from liblaf.apple.warp.fem import StableNeoHookean, StableNeoHookeanActive
    from liblaf.apple.warp.potential import ExternalForce

    if not np.isclose(cfg.rtol_primary, cfg.rtol_secondary):
        msg = (
            "Forward.default_optimizer currently exposes one rtol; keep "
            f"rtol_primary and rtol_secondary equal, got {cfg.rtol_primary} and "
            f"{cfg.rtol_secondary}."
        )
        raise ValueError(msg)

    builder = ModelBuilder()
    builder.add_vertices(body)
    add_boundary_conditions(body, cfg)
    force_surface = add_bottom_force(body, cfg)
    builder.add_fixed(body)

    add_material(
        body,
        E=cfg.E_fat,
        nu=cfg.nu,
        fraction=np.asarray(body.cell_data[FAT_FRACTION]),
    )
    builder.add_potential(StableNeoHookean.from_pyvista(body, name="fat"))

    add_material(
        body,
        E=cfg.E_smas,
        nu=cfg.nu,
        fraction=np.asarray(body.cell_data[SMAS_FRACTION]),
    )
    add_smas_activation(body, cfg)
    builder.add_potential(StableNeoHookeanActive.from_pyvista(body, name="smas"))
    builder.add_potential(ExternalForce.from_pyvista(force_surface, name="force"))

    collision_builder = CollisionBuilder(stiffness=cfg.collision_stiffness)
    collision_builder.add_tetmesh(body)
    builder.collision = collision_builder

    forward = Forward(builder.finalize())
    forward.optimizer = forward.default_optimizer(
        max_steps=cfg.optimizer_max_steps,
        rtol=cfg.rtol_primary,
    )
    return forward


def to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def tensor_scalar(value: Any) -> float:
    return float(to_numpy(value).reshape(-1)[0])


def pressure_schedule(cfg: Config) -> np.ndarray:
    if cfg.load_steps < 1:
        msg = f"load_steps must be >= 1, got {cfg.load_steps}."
        raise ValueError(msg)
    return np.linspace(
        cfg.bottom_pressure / cfg.load_steps, cfg.bottom_pressure, cfg.load_steps
    )


def add_metric_fields(
    mesh: pv.UnstructuredGrid, metrics: dict[str, float | str]
) -> None:
    for name, value in metrics.items():
        if isinstance(value, str):
            mesh.field_data["".join(part.title() for part in name.split("_"))] = (
                np.asarray([value])
            )
        else:
            mesh.field_data["".join(part.title() for part in name.split("_"))] = (
                np.asarray([value])
            )


def solve(
    base_body: pv.UnstructuredGrid, cfg: Config
) -> tuple[pv.UnstructuredGrid, dict[str, float | str]]:
    from liblaf.apple.common import GLOBAL_POINT_ID

    body: pv.UnstructuredGrid | None = None
    forward = None
    solution = None
    initial_energy: float | None = None
    total_optimizer_steps = 0

    for pressure in pressure_schedule(cfg):
        step_cfg = cfg.model_copy(update={"bottom_pressure": float(pressure)})
        previous_u = None if forward is None else forward.state.u.detach().clone()
        body = base_body.copy(deep=True)
        forward = build_model(body, step_cfg)
        if previous_u is not None:
            forward.model.update(forward.state, previous_u)
        if initial_energy is None:
            initial_energy = tensor_scalar(forward.problem.fun(forward.state))
        solution = forward.step()
        total_optimizer_steps += int(solution.state.step)

    assert body is not None
    assert forward is not None
    assert solution is not None
    assert initial_energy is not None
    final_energy = tensor_scalar(forward.problem.fun(forward.state))

    global_ids = body.point_data[GLOBAL_POINT_ID.vtk]
    displacement = to_numpy(forward.state.u)[global_ids]
    deformed_points = body.points + displacement
    metrics: dict[str, float | str] = {
        "initial_energy": initial_energy,
        "final_energy": final_energy,
        "optimizer_steps": float(total_optimizer_steps),
        "optimizer_result": solution.result.name,
        "max_displacement": float(np.linalg.norm(displacement, axis=1).max()),
        "bottom_max_displacement_y": float(
            displacement[
                np.asarray(body.point_data["LoadedBottom"], dtype=bool), 1
            ].max()
        ),
        "bottom_mean_displacement_y": float(
            displacement[
                np.asarray(body.point_data["LoadedBottom"], dtype=bool), 1
            ].mean()
        ),
        "min_y": float(deformed_points[:, 1].min()),
        "max_y": float(deformed_points[:, 1].max()),
        "n_points": float(body.n_points),
        "n_cells": float(body.n_cells),
        "n_fixed": float(forward.model.n_fixed),
        "n_free": float(forward.model.n_free),
        "smas_active_cells": float(
            np.count_nonzero(
                np.asarray(body.cell_data[SMAS_FRACTION]) > SMAS_ACTIVE_TOL
            )
        ),
    }

    result = body.copy(deep=True)
    result.point_data["Displacement"] = displacement
    result.point_data["DisplacementNorm"] = np.linalg.norm(displacement, axis=1)
    result.point_data["DeformedPoint"] = result.points + displacement
    add_reference_fields(result, cfg)
    add_metric_fields(result, metrics)
    return result, metrics


def make_input_mesh(base_body: pv.UnstructuredGrid, cfg: Config) -> pv.UnstructuredGrid:
    mesh = base_body.copy(deep=True)
    add_boundary_conditions(mesh, cfg)
    add_bottom_force(mesh, cfg)
    add_smas_activation(mesh, cfg)
    add_reference_fields(mesh, cfg)
    return mesh


def log_metrics(metrics: dict[str, float | str], cfg: Config) -> None:
    scalars: dict[str, float] = {
        "E_fat": cfg.E_fat,
        "E_smas": cfg.E_smas,
        "nu": cfg.nu,
        "tetwild_lr": cfg.lr,
        "bottom_pressure": cfg.bottom_pressure,
        "collision_stiffness": cfg.collision_stiffness,
        "optimizer_max_steps": float(cfg.optimizer_max_steps),
        "rtol_primary": cfg.rtol_primary,
        "rtol_secondary": cfg.rtol_secondary,
        "load_steps": float(cfg.load_steps),
        "smas_prestrain_x": cfg.smas_prestrain[0],
        "smas_prestrain_y": cfg.smas_prestrain[1],
        "smas_prestrain_z": cfg.smas_prestrain[2],
        "smas_activation_x": activation_from_prestrain(cfg.smas_prestrain)[0],
        "smas_activation_y": activation_from_prestrain(cfg.smas_prestrain)[1],
        "smas_activation_z": activation_from_prestrain(cfg.smas_prestrain)[2],
    }
    scalars.update(
        {name: value for name, value in metrics.items() if not isinstance(value, str)}
    )
    cherries.log_metrics(scalars)


def main(cfg: Config) -> None:
    configure_runtime()

    body = make_body_mesh(cfg)
    input_mesh = make_input_mesh(body, cfg)
    melon.save(cfg.output_input, input_mesh)

    result, metrics = solve(body, cfg)
    melon.save(cfg.output, result)

    log_metrics(metrics, cfg)
    print(f"saved: {cfg.output_input}")
    print(f"saved: {cfg.output}")
    print(
        f"optimizer: {metrics['optimizer_result']} in {metrics['optimizer_steps']:.0f} steps"
    )
    print(
        "bottom max displacement y:",
        f"{metrics['bottom_max_displacement_y']:.6g}",
    )
    print(
        "bottom mean displacement y:",
        f"{metrics['bottom_mean_displacement_y']:.6g}",
    )
    print("max displacement:", f"{metrics['max_displacement']:.6g}")


if __name__ == "__main__":
    cherries.main(main, profile="debug")
