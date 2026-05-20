from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
import pyvista as pv
import torch
import warp as wp

from liblaf import cherries, melon

OUTPUT_STEM = "20-activated-boxes-collision"
LOWER_BOX = (0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
UPPER_BOX = (0.0, 1.0, 1.5, 2.5, 0.0, 1.0)
LOWER_ACTIVE_BOX = (0.0, 1.0, 0.0, 0.5, 0.0, 1.0)
UPPER_ACTIVE_BOX = (0.0, 1.0, 2.0, 2.5, 0.0, 1.0)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)
    output_input: Path = cherries.output(f"{OUTPUT_STEM}-input.vtu")
    output_no_collision: Path = cherries.output(f"{OUTPUT_STEM}-no-collision.vtu")
    output_collision: Path = cherries.output(f"{OUTPUT_STEM}-collision.vtu")
    output_comparison: Path = cherries.output(f"{OUTPUT_STEM}-comparison.vtu")

    E: float = 1.0
    nu: float = 0.49
    lr: float = 0.05
    coarsen: bool = False
    y_stretch: float = 2.0
    collision_stiffness_scale: float = 0.1
    optimizer_rtol: float = 5e-4
    active_fraction_tol: float = 1.0e-6
    fixed_atol: float = 1.0e-6


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


def make_scene_surface() -> pv.PolyData:
    surface = pv.merge([make_box(LOWER_BOX), make_box(UPPER_BOX)])
    surface.clean(inplace=True)
    return surface


def make_scene_mesh(cfg: Config) -> pv.UnstructuredGrid:
    surface = make_scene_surface()
    mesh: pv.UnstructuredGrid = melon.tetwild(surface, lr=cfg.lr, coarsen=cfg.coarsen)
    mesh = melon.tet.fix_winding(mesh)
    add_scene_tags(mesh)
    add_activation_fractions(mesh)
    return mesh


def add_scene_tags(mesh: pv.UnstructuredGrid) -> None:
    point_box_id = np.where(mesh.points[:, 1] < 1.25, 0, 1).astype(np.int8)
    cell_centers = mesh.cell_centers().points
    cell_box_id = np.where(cell_centers[:, 1] < 1.25, 0, 1).astype(np.int8)
    mesh.point_data["BoxId"] = point_box_id
    mesh.cell_data["BoxId"] = cell_box_id


def add_activation_fractions(mesh: pv.UnstructuredGrid) -> None:
    lower_active = make_box(LOWER_ACTIVE_BOX)
    upper_active = make_box(UPPER_ACTIVE_BOX)
    lower_fraction = np.asarray(melon.tet.compute_volume_fraction(mesh, lower_active))
    upper_fraction = np.asarray(melon.tet.compute_volume_fraction(mesh, upper_active))
    active_fraction = np.clip(lower_fraction + upper_fraction, 0.0, 1.0)
    passive_fraction = np.clip(1.0 - active_fraction, 0.0, 1.0)

    mesh.cell_data["LowerActivationFraction"] = lower_fraction
    mesh.cell_data["UpperActivationFraction"] = upper_fraction
    mesh.cell_data["ActiveFraction"] = active_fraction
    mesh.cell_data["PassiveFraction"] = passive_fraction


def activation_from_y_stretch(y_stretch: float) -> np.ndarray:
    if y_stretch <= 0.0:
        msg = f"y_stretch must be positive, got {y_stretch}"
        raise ValueError(msg)
    activation = np.zeros(6, dtype=np.float64)
    activation[1] = y_stretch - 1.0
    return activation


def activation_inv_from_activation(activation: np.ndarray) -> np.ndarray:
    stretch = 1.0 + activation[:3]
    activation_inv = np.zeros(6, dtype=np.float64)
    activation_inv[:3] = (1.0 / stretch) - 1.0
    return activation_inv


def add_boundary_conditions(mesh: pv.UnstructuredGrid, cfg: Config) -> None:
    from liblaf.apple.common import FIXED_MASK, FIXED_VALUE

    y = mesh.points[:, 1]
    lower_fixed = np.isclose(y, LOWER_BOX[2], atol=cfg.fixed_atol)
    upper_fixed = np.isclose(y, UPPER_BOX[3], atol=cfg.fixed_atol)
    fixed = lower_fixed | upper_fixed

    fixed_mask = np.zeros((mesh.n_points, 3), dtype=bool)
    fixed_value = np.zeros((mesh.n_points, 3), dtype=np.float64)
    fixed_mask[fixed, :] = True

    mesh.point_data[FIXED_MASK.vtk] = fixed_mask
    mesh.point_data[FIXED_VALUE.vtk] = fixed_value
    mesh.point_data["FixedLowerYMin"] = lower_fixed.astype(np.int8)
    mesh.point_data["FixedUpperYMax"] = upper_fixed.astype(np.int8)


def add_material_fields(mesh: pv.UnstructuredGrid, cfg: Config) -> None:
    from liblaf.apple.common import ACTIVATION, FRACTION, LAMBDA, MU, NU, lame_converter
    from liblaf.apple.common import E as YOUNG_MODULUS

    lambda_, mu = lame_converter(cfg.E, cfg.nu)
    activation_value = activation_from_y_stretch(cfg.y_stretch)
    active = np.asarray(mesh.cell_data["ActiveFraction"]) > cfg.active_fraction_tol
    activation = np.zeros((mesh.n_cells, 6), dtype=np.float64)
    activation[active] = activation_value

    mesh.cell_data[YOUNG_MODULUS.vtk] = np.full(mesh.n_cells, cfg.E, dtype=np.float64)
    mesh.cell_data[NU.vtk] = np.full(mesh.n_cells, cfg.nu, dtype=np.float64)
    mesh.cell_data[LAMBDA.vtk] = np.full(mesh.n_cells, float(lambda_), dtype=np.float64)
    mesh.cell_data[MU.vtk] = np.full(mesh.n_cells, float(mu), dtype=np.float64)
    mesh.cell_data[ACTIVATION.vtk] = activation
    mesh.cell_data[FRACTION.vtk] = np.asarray(mesh.cell_data["ActiveFraction"])
    mesh.cell_data["ActivationMask"] = active.astype(np.int8)

    mesh.field_data["E"] = np.asarray([cfg.E])
    mesh.field_data["Nu"] = np.asarray([cfg.nu])
    mesh.field_data["Lambda"] = np.asarray([float(lambda_)])
    mesh.field_data["Mu"] = np.asarray([float(mu)])
    mesh.field_data["ActivationStretch"] = np.asarray([[1.0, cfg.y_stretch, 1.0]])
    mesh.field_data["Activation"] = activation_value[np.newaxis, :]
    mesh.field_data["ActivationInv"] = activation_inv_from_activation(activation_value)[
        np.newaxis, :
    ]
    mesh.field_data["CollisionStiffness"] = np.asarray(
        [cfg.collision_stiffness_scale * cfg.E]
    )
    mesh.field_data["OptimizerRtol"] = np.asarray([cfg.optimizer_rtol])
    mesh.field_data["TetWildLr"] = np.asarray([cfg.lr])


def set_fraction(mesh: pv.UnstructuredGrid, name: str) -> None:
    from liblaf.apple.common import FRACTION

    mesh.cell_data[FRACTION.vtk] = np.asarray(mesh.cell_data[name])


def build_model(mesh: pv.UnstructuredGrid, cfg: Config, *, with_collision: bool):
    from liblaf.apple.collision import CollisionBuilder
    from liblaf.apple.forward import Forward, ModelBuilder
    from liblaf.apple.warp.fem import StableNeoHookean, StableNeoHookeanActive

    builder = ModelBuilder()
    builder.add_vertices(mesh)
    builder.add_fixed(mesh)

    set_fraction(mesh, "PassiveFraction")
    builder.add_potential(StableNeoHookean.from_pyvista(mesh, name="passive"))

    set_fraction(mesh, "ActiveFraction")
    builder.add_potential(StableNeoHookeanActive.from_pyvista(mesh, name="active"))

    if with_collision:
        collision_builder = CollisionBuilder(
            stiffness=cfg.collision_stiffness_scale * cfg.E
        )
        collision_builder.add_tetmesh(mesh)
        builder.collision = collision_builder

    forward = Forward(builder.finalize())
    forward.optimizer = forward.default_optimizer(rtol=cfg.optimizer_rtol)
    return forward


def to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def tensor_scalar(value: Any) -> float:
    return float(to_numpy(value).reshape(-1)[0])


def gap_metrics(
    mesh: pv.UnstructuredGrid, displacement: np.ndarray, prefix: str
) -> dict[str, float]:
    box_id = np.asarray(mesh.point_data["BoxId"])
    deformed_y = mesh.points[:, 1] + displacement[:, 1]
    lower_y_max = float(deformed_y[box_id == 0].max())
    upper_y_min = float(deformed_y[box_id == 1].min())
    gap = upper_y_min - lower_y_max
    return {
        f"{prefix}_lower_y_max": lower_y_max,
        f"{prefix}_upper_y_min": upper_y_min,
        f"{prefix}_gap": gap,
        f"{prefix}_overlap": max(0.0, -gap),
    }


def add_metric_fields(
    mesh: pv.UnstructuredGrid, metrics: dict[str, float | str]
) -> None:
    for name, value in metrics.items():
        field_name = "".join(part.title() for part in name.split("_"))
        mesh.field_data[field_name] = np.asarray([value])


def make_result_mesh(
    mesh: pv.UnstructuredGrid,
    displacement: np.ndarray,
    metrics: dict[str, float | str],
    *,
    with_collision: bool,
) -> pv.UnstructuredGrid:
    result = mesh.copy(deep=True)
    result.point_data["Displacement"] = displacement
    result.point_data["DisplacementNorm"] = np.linalg.norm(displacement, axis=1)
    result.point_data["DeformedPoint"] = result.points + displacement
    result.field_data["WithCollision"] = np.asarray([int(with_collision)])
    add_metric_fields(result, metrics)
    return result


def solve(
    base_mesh: pv.UnstructuredGrid,
    cfg: Config,
    *,
    with_collision: bool,
) -> tuple[pv.UnstructuredGrid, np.ndarray, dict[str, float | str]]:
    from liblaf.apple.common import GLOBAL_POINT_ID

    mesh = base_mesh.copy(deep=True)
    forward = build_model(mesh, cfg, with_collision=with_collision)

    initial_energy = tensor_scalar(forward.problem.fun(forward.state))
    solution = forward.step()
    final_energy = tensor_scalar(forward.problem.fun(forward.state))

    global_ids = mesh.point_data[GLOBAL_POINT_ID.vtk]
    displacement = to_numpy(forward.state.u)[global_ids]
    metrics: dict[str, float | str] = {
        "initial_energy": initial_energy,
        "final_energy": final_energy,
        "optimizer_steps": float(solution.state.step),
        "optimizer_result": solution.result.name,
        "max_displacement": float(np.linalg.norm(displacement, axis=1).max()),
        "n_points": float(mesh.n_points),
        "n_cells": float(mesh.n_cells),
        "n_fixed": float(forward.model.n_fixed),
        "n_free": float(forward.model.n_free),
    }
    metrics.update(gap_metrics(mesh, displacement, "final"))
    return (
        make_result_mesh(mesh, displacement, metrics, with_collision=with_collision),
        displacement,
        metrics,
    )


def make_comparison(
    base_mesh: pv.UnstructuredGrid,
    no_collision: np.ndarray,
    collision: np.ndarray,
    no_collision_metrics: dict[str, float | str],
    collision_metrics: dict[str, float | str],
) -> pv.UnstructuredGrid:
    comparison = base_mesh.copy(deep=True)
    comparison.point_data["DisplacementNoCollision"] = no_collision
    comparison.point_data["DisplacementCollision"] = collision
    comparison.point_data["CollisionDelta"] = collision - no_collision
    comparison.point_data["Displacement"] = collision
    comparison.point_data["DeformedPoint"] = comparison.points + collision
    comparison.field_data["NoCollisionFinalGap"] = np.asarray(
        [no_collision_metrics["final_gap"]]
    )
    comparison.field_data["CollisionFinalGap"] = np.asarray(
        [collision_metrics["final_gap"]]
    )
    comparison.field_data["NoCollisionFinalOverlap"] = np.asarray(
        [no_collision_metrics["final_overlap"]]
    )
    comparison.field_data["CollisionFinalOverlap"] = np.asarray(
        [collision_metrics["final_overlap"]]
    )
    comparison.field_data["CollisionDeltaMax"] = np.asarray(
        [float(np.linalg.norm(collision - no_collision, axis=1).max())]
    )
    return comparison


def log_metrics(
    no_collision_metrics: dict[str, float | str],
    collision_metrics: dict[str, float | str],
    cfg: Config,
) -> None:
    metrics: dict[str, float] = {
        "E": cfg.E,
        "nu": cfg.nu,
        "tetwild_lr": cfg.lr,
        "collision_stiffness": cfg.collision_stiffness_scale * cfg.E,
        "optimizer_rtol": cfg.optimizer_rtol,
    }
    metrics.update(
        {
            f"no_collision/{name}": value
            for name, value in no_collision_metrics.items()
            if not isinstance(value, str)
        }
    )
    metrics.update(
        {
            f"collision/{name}": value
            for name, value in collision_metrics.items()
            if not isinstance(value, str)
        }
    )
    cherries.log_metrics(metrics)


def main(cfg: Config) -> None:
    configure_runtime()

    mesh = make_scene_mesh(cfg)
    add_boundary_conditions(mesh, cfg)
    add_material_fields(mesh, cfg)
    melon.save(cfg.output_input, mesh)

    no_collision_result, no_collision_u, no_collision_metrics = solve(
        mesh, cfg, with_collision=False
    )
    melon.save(cfg.output_no_collision, no_collision_result)

    collision_result, collision_u, collision_metrics = solve(
        mesh, cfg, with_collision=True
    )
    melon.save(cfg.output_collision, collision_result)

    comparison = make_comparison(
        mesh,
        no_collision_u,
        collision_u,
        no_collision_metrics,
        collision_metrics,
    )
    melon.save(cfg.output_comparison, comparison)

    log_metrics(no_collision_metrics, collision_metrics, cfg)
    print(f"saved: {cfg.output_input}")
    print(f"saved: {cfg.output_no_collision}")
    print(f"saved: {cfg.output_collision}")
    print(f"saved: {cfg.output_comparison}")
    print(
        "final gap:",
        f"{no_collision_metrics['final_gap']:.6g}",
        "without collision,",
        f"{collision_metrics['final_gap']:.6g}",
        "with collision",
    )
    print(
        "final overlap:",
        f"{no_collision_metrics['final_overlap']:.6g}",
        "without collision,",
        f"{collision_metrics['final_overlap']:.6g}",
        "with collision",
    )


if __name__ == "__main__":
    cherries.main(main, profile="debug")
