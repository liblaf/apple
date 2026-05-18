from pathlib import Path
from typing import Any

import attrs
import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv
import warp as wp
from icecream import ic

from liblaf import cherries, melon

SMAS_FRACTION = "SmasFraction"
MUSCLE_FRACTION = "MuscleFraction"
FAT_FRACTION = "FatFraction"
STIFF_MID_FRACTION = "StiffMidLayerFraction"


class Config(cherries.BaseConfig):
    output_no_collision: Path = cherries.output(
        "20-ext-force-stable-neo-hookean-no-collision.vtu"
    )
    output_collision: Path = cherries.output(
        "20-ext-force-stable-neo-hookean-collision.vtu"
    )
    output_comparison: Path = cherries.output(
        "20-ext-force-stable-neo-hookean-collision-comparison.vtu"
    )

    width: float = 1.0
    thickness: float = 0.1
    depth: float = 1.0
    body_y_min: float = 0.0
    lr: float = 0.02
    coarsen: bool = False

    E: float = 1.0
    nu: float = 0.375
    stiffness_ratio: float = 1e4
    force_scale: float = 1.0


def body_bounds(cfg: Config) -> tuple[float, float, float, float, float, float]:
    return (
        0.0,
        cfg.width,
        cfg.body_y_min,
        cfg.body_y_min + cfg.thickness,
        0.0,
        cfg.depth,
    )


def make_body_surface(cfg: Config) -> pv.PolyData:
    return pv.Box(body_bounds(cfg), quads=False)


def make_stiff_mid_layer(cfg: Config) -> pv.PolyData:
    xmin, xmax, ymin, _ymax, zmin, zmax = body_bounds(cfg)
    return pv.Box(
        (
            xmin,
            xmax,
            ymin + 0.4 * cfg.thickness,
            ymin + 0.6 * cfg.thickness,
            zmin,
            zmax,
        ),
        quads=False,
    )


def make_muscle_patch(cfg: Config) -> pv.PolyData:
    xmin, xmax, ymin, _ymax, zmin, zmax = body_bounds(cfg)
    return pv.Box(
        (
            xmin,
            xmin + 0.5 * (xmax - xmin),
            ymin + 0.4 * cfg.thickness,
            ymin + 0.6 * cfg.thickness,
            zmin + 0.4 * (zmax - zmin),
            zmin + 0.6 * (zmax - zmin),
        ),
        quads=False,
    )


def make_body_mesh(cfg: Config) -> pv.UnstructuredGrid:
    surface = make_body_surface(cfg)
    stiff_mid_layer = make_stiff_mid_layer(cfg)
    muscle_patch = make_muscle_patch(cfg)
    mesh: pv.UnstructuredGrid = melon.tetwild(surface, lr=cfg.lr, coarsen=cfg.coarsen)
    mesh = melon.tet.fix_winding(mesh)

    smas_frac = np.asarray(melon.tet.compute_volume_fraction(mesh, stiff_mid_layer))
    muscle_frac = np.asarray(melon.tet.compute_volume_fraction(mesh, muscle_patch))
    mesh.cell_data[SMAS_FRACTION] = smas_frac
    mesh.cell_data[MUSCLE_FRACTION] = muscle_frac
    mesh.cell_data[FAT_FRACTION] = np.clip(1.0 - smas_frac, 0.0, 1.0)
    mesh.cell_data[STIFF_MID_FRACTION] = smas_frac
    return mesh


def add_material(
    mesh: pv.UnstructuredGrid, E: float, nu: float, fraction: np.ndarray
) -> None:
    from liblaf.apple.common import FRACTION, LAMBDA, MU, NU, lame_converter
    from liblaf.apple.common import E as YOUNG_MODULUS

    lambda_, mu = lame_converter(E, nu)
    mesh.cell_data[YOUNG_MODULUS.vtk] = np.full(mesh.n_cells, E, dtype=np.float64)
    mesh.cell_data[NU.vtk] = np.full(mesh.n_cells, nu, dtype=np.float64)
    mesh.cell_data[LAMBDA.vtk] = np.full(mesh.n_cells, float(lambda_), dtype=np.float64)
    mesh.cell_data[MU.vtk] = np.full(mesh.n_cells, float(mu), dtype=np.float64)
    mesh.cell_data[FRACTION.vtk] = fraction


def add_body_boundary_conditions(mesh: pv.UnstructuredGrid) -> None:
    from liblaf.apple.common import FIXED_MASK, FIXED_VALUE

    x = mesh.points[:, 0]
    z = mesh.points[:, 2]
    side = (
        np.isclose(x, x.min(), atol=1e-2)
        | np.isclose(x, x.max(), atol=1e-2)
        | np.isclose(z, z.min(), atol=1e-2)
        | np.isclose(z, z.max(), atol=1e-2)
    )

    fixed_mask = np.zeros((mesh.n_points, 3), dtype=bool)
    fixed_value = np.zeros((mesh.n_points, 3), dtype=np.float64)
    fixed_mask[side, :] = True

    mesh.point_data[FIXED_MASK.vtk] = fixed_mask
    mesh.point_data[FIXED_VALUE.vtk] = fixed_value
    mesh.point_data["FixedSide"] = side.astype(np.int8)


def add_bottom_force(mesh: pv.UnstructuredGrid, force_scale: float) -> pv.PolyData:
    from liblaf.apple.common import FIXED_MASK, FORCE, GLOBAL_POINT_ID

    surface: pv.PolyData = mesh.extract_surface(algorithm=None)
    surface = melon.tri.compute_point_area(surface)

    thickness = mesh.points[:, 1].max() - mesh.points[:, 1].min()
    bottom_tol = max(1.0e-6, 0.1 * thickness)
    bottom = surface.points[:, 1] <= mesh.points[:, 1].min() + bottom_tol
    free_bottom = bottom & ~surface.point_data[FIXED_MASK.vtk][:, 0]
    force = np.zeros((surface.n_points, 3), dtype=np.float64)
    force[free_bottom, 1] = force_scale * surface.point_data["Area"][free_bottom]
    surface.point_data[FORCE.vtk] = force

    mesh.point_data[FORCE.vtk] = np.zeros((mesh.n_points, 3), dtype=np.float64)
    global_ids = surface.point_data[GLOBAL_POINT_ID.vtk][free_bottom]
    mesh.point_data[FORCE.vtk][global_ids] = force[free_bottom]
    mesh.point_data["LoadedBottom"] = np.zeros(mesh.n_points, dtype=np.int8)
    mesh.point_data["LoadedBottom"][global_ids] = 1

    return surface


def build_model(
    body: pv.UnstructuredGrid,
    *,
    E: float,
    nu: float,
    stiffness_ratio: float,
    force_scale: float,
    with_collision: bool,
):
    from liblaf.apple.collision import CollisionBuilder
    from liblaf.apple.forward import ModelBuilder
    from liblaf.apple.warp.fem import StableNeoHookean
    from liblaf.apple.warp.potential import ExternalForce

    builder = ModelBuilder()
    builder.add_vertices(body)

    add_body_boundary_conditions(body)
    force_surface = add_bottom_force(body, force_scale)

    builder.add_fixed(body)

    add_material(body, E, nu, np.asarray(body.cell_data[FAT_FRACTION]))
    builder.add_potential(StableNeoHookean.from_pyvista(body, name="fat"))

    add_material(
        body,
        stiffness_ratio * E,
        nu,
        np.asarray(body.cell_data[STIFF_MID_FRACTION]),
    )
    builder.add_potential(StableNeoHookean.from_pyvista(body, name="stiff_mid_layer"))

    builder.add_potential(
        attrs.evolve(ExternalForce.from_pyvista(force_surface), name="force")
    )

    if with_collision:
        collision_builder = CollisionBuilder(stiffness=1.0 * E)
        collision_builder.add_tetmesh(body)
        builder.collision = collision_builder
    return builder.finalize()


def log_optimizer_progress(case: str, opt_state: Any) -> None:
    line_search = opt_state.line_search_state
    convergence = opt_state.convergence_state
    hess_damping = opt_state.hess_damping_state
    relative_grad_norm = convergence.grad_norm / convergence.grad_norm_first
    step_direction = line_search.alpha * opt_state.direction

    cherries.log_metrics(
        {
            f"{case}/optimizer/fun": line_search.f_alpha,
            f"{case}/optimizer/fun_previous": line_search.f0,
            f"{case}/optimizer/actual_decrease": line_search.f0 - line_search.f_alpha,
            f"{case}/optimizer/alpha": line_search.alpha,
            f"{case}/optimizer/line_search_steps": line_search.step,
            f"{case}/optimizer/line_search_ok": line_search.ok,
            f"{case}/optimizer/grad_norm": convergence.grad_norm,
            f"{case}/optimizer/relative_grad_norm": relative_grad_norm,
            f"{case}/optimizer/direction_norm": jnp.linalg.norm(opt_state.direction),
            f"{case}/optimizer/step_norm": jnp.linalg.norm(step_direction),
            f"{case}/optimizer/step_max_norm": jnp.linalg.norm(
                step_direction, ord=jnp.inf
            ),
            f"{case}/optimizer/hess_damping_factor": hess_damping.factor,
            f"{case}/optimizer/hess_diag_mean": hess_damping.hess_diag_mean,
        },
        step=opt_state.n_steps,
    )


def run_forward_with_progress(forward: Any, case: str) -> Any:
    from liblaf.peach.optim import Result

    problem = forward.problem
    model_state = forward.state
    opt_state = forward.optimizer.init(problem, model_state, forward.free)
    result = Result.UNKNOWN_ERROR

    while True:
        model_state, opt_state = forward.optimizer.step(problem, model_state, opt_state)
        log_optimizer_progress(case, opt_state)
        done, result = forward.optimizer.terminate(problem, model_state, opt_state)
        if bool(np.asarray(done)):
            break

    solution = forward.optimizer.postprocess(problem, model_state, opt_state, result)
    forward.state = model_state
    return solution


def solve(
    base_body: pv.UnstructuredGrid,
    cfg: Config,
    *,
    with_collision: bool,
) -> tuple[pv.UnstructuredGrid, np.ndarray, dict[str, float | str]]:
    from liblaf.apple.common import GLOBAL_POINT_ID
    from liblaf.apple.forward import Forward

    body = base_body.copy(deep=True)
    model = build_model(
        body,
        E=cfg.E,
        nu=cfg.nu,
        stiffness_ratio=cfg.stiffness_ratio,
        force_scale=cfg.force_scale,
        with_collision=with_collision,
    )
    forward = Forward(model)

    initial_energy = float(forward.problem.fun(forward.state))
    case = "self_collision" if with_collision else "no_collision"
    solution = run_forward_with_progress(forward, case)
    ic(solution)
    final_energy = float(forward.problem.fun(forward.state))

    global_ids = body.point_data[GLOBAL_POINT_ID.vtk]
    displacement = np.asarray(forward.state.u)[global_ids]
    deformed_points = body.points + displacement
    result = body.copy(deep=True)
    result.point_data["Displacement"] = displacement
    result.field_data["WithCollision"] = np.asarray([int(with_collision)])
    result.field_data["InitialEnergy"] = np.asarray([initial_energy])
    result.field_data["FinalEnergy"] = np.asarray([final_energy])
    result.field_data["SelfCollisionStiffness"] = np.asarray([0.1 * cfg.E])
    result.field_data["OptimizerResult"] = np.asarray([solution.result.name])

    min_y = float(deformed_points[:, 1].min())
    max_y = float(deformed_points[:, 1].max())
    metrics: dict[str, float | str] = {
        "initial_energy": initial_energy,
        "final_energy": final_energy,
        "min_y": min_y,
        "max_y": max_y,
        "max_displacement": float(np.linalg.norm(displacement, axis=1).max()),
        "optimizer_result": solution.result.name,
    }
    return result, displacement, metrics


def make_comparison(
    base_body: pv.UnstructuredGrid,
    no_collision: np.ndarray,
    collision: np.ndarray,
    cfg: Config,
    no_collision_metrics: dict[str, float | str],
    collision_metrics: dict[str, float | str],
) -> pv.UnstructuredGrid:
    comparison = base_body.copy(deep=True)
    comparison.point_data["DisplacementNoCollision"] = no_collision
    comparison.point_data["DisplacementCollision"] = collision
    comparison.point_data["CollisionDelta"] = collision - no_collision
    comparison.field_data["NoCollisionMinY"] = np.asarray(
        [no_collision_metrics["min_y"]]
    )
    comparison.field_data["CollisionMinY"] = np.asarray([collision_metrics["min_y"]])
    comparison.field_data["NoCollisionMaxY"] = np.asarray(
        [no_collision_metrics["max_y"]]
    )
    comparison.field_data["CollisionMaxY"] = np.asarray([collision_metrics["max_y"]])
    comparison.field_data["NoCollisionMaxDisplacement"] = np.asarray(
        [no_collision_metrics["max_displacement"]]
    )
    comparison.field_data["CollisionMaxDisplacement"] = np.asarray(
        [collision_metrics["max_displacement"]]
    )
    comparison.field_data["SelfCollisionStiffness"] = np.asarray([0.1 * cfg.E])
    comparison.field_data["StiffnessRatio"] = np.asarray([cfg.stiffness_ratio])
    return comparison


def main(cfg: Config) -> None:
    if jax.default_backend() != "gpu":
        msg = "This experiment uses Warp JAX FFI and needs JAX's active backend to be GPU."
        raise RuntimeError(msg)

    wp.config.mode = "release"
    wp.init()

    body = make_body_mesh(cfg)

    no_collision_result, no_collision_u, no_collision_metrics = solve(
        body, cfg, with_collision=False
    )
    melon.save(cfg.output_no_collision, no_collision_result)
    collision_result, collision_u, collision_metrics = solve(
        body, cfg, with_collision=True
    )
    melon.save(cfg.output_collision, collision_result)
    comparison = make_comparison(
        body,
        no_collision_u,
        collision_u,
        cfg,
        no_collision_metrics,
        collision_metrics,
    )
    melon.save(cfg.output_comparison, comparison)

    metrics = {
        "E": cfg.E,
        "nu": cfg.nu,
        "force_scale": cfg.force_scale,
        "stiffness_ratio": cfg.stiffness_ratio,
        "self_collision_stiffness": 0.1 * cfg.E,
        "n_points": float(body.n_points),
        "n_cells": float(body.n_cells),
        "no_collision": no_collision_metrics,
        "collision": collision_metrics,
    }
    cherries.log_metrics(metrics)
    print(f"saved: {cfg.output_no_collision}")
    print(f"saved: {cfg.output_collision}")
    print(f"saved: {cfg.output_comparison}")
    print(
        "max y:",
        f"{no_collision_metrics['max_y']:.6g}",
        "without collision,",
        f"{collision_metrics['max_y']:.6g}",
        "with collision",
    )
    print(
        "max displacement:",
        f"{no_collision_metrics['max_displacement']:.6g}",
        "without collision,",
        f"{collision_metrics['max_displacement']:.6g}",
        "with collision",
    )


if __name__ == "__main__":
    cherries.main(main, profile="debug")
