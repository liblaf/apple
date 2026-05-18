import contextlib
from pathlib import Path
from typing import Any

import attrs
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import warp as wp
from environs import env
from frozendict import frozendict
from icecream import ic

from liblaf import cherries, melon

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_STEM = "21-smas-prestrain-stable-neo-hookean-muscle"
SMAS_FRACTION = "SmasFraction"
FAT_FRACTION = "FatFraction"
SMAS_ACTIVE_TOL = 1.0e-3
METRIC_PLOTS = {
    "objective": ("Objective", "energy", True),
    "actual-decrease": ("Actual Decrease", "energy decrease", True),
    "alpha": ("Line Search Step Size", "alpha", True),
    "line-search-steps": ("Line Search Steps", "steps", False),
    "line-search-ok": ("Line Search Accepted", "accepted", False),
    "grad-norm": ("Gradient Norm", "gradient norm", True),
    "relative-grad-norm": ("Relative Gradient Norm", "relative gradient norm", True),
    "elastic-grad-norm": ("Elastic Potential Gradient Norm", "gradient norm", True),
    "collision-grad-norm": ("Collision Potential Gradient Norm", "gradient norm", True),
    "collision-elastic-grad-ratio": (
        "Collision / Elastic Gradient Norm",
        "ratio",
        True,
    ),
    "direction-norm": ("Search Direction Norm", "direction norm", True),
    "step-norm": ("Step Norm", "step norm", True),
    "step-max-norm": ("Step Max Norm", "step max norm", True),
    "max-displacement": ("Max Displacement", "displacement", True),
    "min-y": ("Minimum Deformed Y", "y", False),
    "max-y": ("Maximum Deformed Y", "y", False),
    "hess-damping-factor": ("Hessian Damping Factor", "factor", True),
    "hess-diag-mean": ("Mean Hessian Diagonal", "mean diagonal", True),
}
type MetricRecord = dict[str, float]


class Config(cherries.BaseConfig):
    output_no_collision: Path = cherries.output(f"{OUTPUT_STEM}-no-collision.vtu")
    output_collision: Path = cherries.output(f"{OUTPUT_STEM}-collision.vtu")
    output_comparison: Path = cherries.output(f"{OUTPUT_STEM}-comparison.vtu")
    output_no_collision_steps: Path = cherries.output(
        f"{OUTPUT_STEM}-no-collision-steps.vtu.series"
    )
    output_collision_steps: Path = cherries.output(
        f"{OUTPUT_STEM}-collision-steps.vtu.series"
    )

    width: float = 1.0
    thickness: float = 0.1
    depth: float = 1.0
    body_y_min: float = 0.0
    lr: float = 0.02
    coarsen: bool = False

    E: float = 1.0
    nu: float = 0.49
    stiffness_ratio: float = 1.0e3
    force_scale: float = 0.3
    collision_stiffness_scale: float = 0.0
    smas_prestrain: tuple[float, float, float] = (1.2, 1.3**-2, 1.2)
    write_step_artifacts: bool = env.bool("WRITE_STEP_ARTIFACTS", True)


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


def make_smas_layer(cfg: Config) -> pv.PolyData:
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


def make_body_mesh(cfg: Config) -> pv.UnstructuredGrid:
    surface = make_body_surface(cfg)
    smas_layer = make_smas_layer(cfg)
    mesh: pv.UnstructuredGrid = melon.tetwild(surface, lr=cfg.lr, coarsen=cfg.coarsen)
    mesh = melon.tet.fix_winding(mesh)

    smas_frac = np.asarray(melon.tet.compute_volume_fraction(mesh, smas_layer))
    mesh.cell_data[SMAS_FRACTION] = smas_frac
    mesh.cell_data[FAT_FRACTION] = np.clip(1.0 - smas_frac, 0.0, 1.0)
    return mesh


def diagonal_activation_from_prestrain(
    prestrain: tuple[float, float, float],
) -> np.ndarray:
    prestrain_array = np.asarray(prestrain, dtype=np.float64)
    activation = np.zeros(6, dtype=np.float64)
    activation[:3] = prestrain_array - 1.0
    return activation


def add_smas_activation(
    mesh: pv.UnstructuredGrid, prestrain: tuple[float, float, float]
) -> np.ndarray:
    from liblaf.apple.common import ACTIVATION, PRESTRAIN

    smas = np.asarray(mesh.cell_data[SMAS_FRACTION]) > SMAS_ACTIVE_TOL
    activation_value = diagonal_activation_from_prestrain(prestrain)

    activation = np.zeros((mesh.n_cells, 6), dtype=np.float64)
    activation[smas] = activation_value
    mesh.cell_data[ACTIVATION] = activation

    prestrain_field = np.ones((mesh.n_cells, 3), dtype=np.float64)
    prestrain_field[smas] = np.asarray(prestrain, dtype=np.float64)
    mesh.cell_data[PRESTRAIN.vtk] = prestrain_field
    mesh.cell_data["SmasActive"] = smas.astype(np.int8)
    return activation_value


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
        np.isclose(x, x.min(), atol=1.0e-2)
        | np.isclose(x, x.max(), atol=1.0e-2)
        | np.isclose(z, z.min(), atol=1.0e-2)
        | np.isclose(z, z.max(), atol=1.0e-2)
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
    collision_stiffness_scale: float,
    smas_prestrain: tuple[float, float, float],
    with_collision: bool,
):
    from liblaf.apple.collision import CollisionBuilder
    from liblaf.apple.forward import ModelBuilder
    from liblaf.apple.warp.fem import StableNeoHookean, StableNeoHookeanMuscle
    from liblaf.apple.warp.potential import ExternalForce

    builder = ModelBuilder()
    builder.add_vertices(body)

    add_body_boundary_conditions(body)
    force_surface = add_bottom_force(body, force_scale)
    builder.add_fixed(body)

    add_material(body, E, nu, np.asarray(body.cell_data[FAT_FRACTION]))
    builder.add_potential(StableNeoHookean.from_pyvista(body, name="fat"))

    activation_value = add_smas_activation(body, smas_prestrain)
    add_material(
        body,
        stiffness_ratio * E,
        nu,
        np.asarray(body.cell_data[SMAS_FRACTION]),
    )
    builder.add_potential(
        StableNeoHookeanMuscle.from_pyvista(body, name="smas_prestrain")
    )

    builder.add_potential(
        attrs.evolve(ExternalForce.from_pyvista(force_surface), name="force")
    )

    if with_collision:
        collision_builder = CollisionBuilder(stiffness=collision_stiffness_scale * E)
        collision_builder.add_tetmesh(body)
        builder.collision = collision_builder
    return builder.finalize(), activation_value


def case_name(*, with_collision: bool) -> str:
    return "collision" if with_collision else "no_collision"


def metric_plot_path(plot_name: str) -> Path:
    return EXPERIMENT_DIR / "fig" / f"{OUTPUT_STEM}-metrics-{plot_name}.png"


def metric_plot_paths() -> dict[str, str]:
    return {
        plot_name: metric_plot_path(plot_name).as_posix() for plot_name in METRIC_PLOTS
    }


def scalar_float(value: Any) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def scalar_bool(value: Any) -> float:
    return float(bool(np.asarray(value).reshape(-1)[0]))


def metric_field_name(metric: str) -> str:
    return "".join(part.title() for part in metric.split("-"))


def make_elastic_warp_model(forward: Any) -> Any:
    from liblaf.apple.warp.model import WarpModel, WarpModelAdapter

    potentials = forward.model.warp_model.__wrapped__.potentials
    elastic_potentials = frozendict(
        {name: potential for name, potential in potentials.items() if name != "force"}
    )
    return WarpModelAdapter(
        WarpModel(elastic_potentials),
        n_points=forward.model.n_points,
    )


def free_grad_norm(forward: Any, grad_full: Any) -> float:
    grad_free = forward.model.dof_map.to_free_grad(grad_full)
    return scalar_float(jnp.linalg.norm(grad_free))


def collect_split_gradient_norms(
    *,
    forward: Any,
    model_state: Any,
    elastic_warp_model: Any,
) -> MetricRecord:
    elastic_grad_norm = free_grad_norm(forward, elastic_warp_model.grad(model_state.u))
    if forward.model.collision is None:
        collision_grad_norm = 0.0
    else:
        collision_grad_norm = free_grad_norm(
            forward, forward.model.collision.grad(model_state.u)
        )

    if elastic_grad_norm > 0.0:
        collision_elastic_ratio = collision_grad_norm / elastic_grad_norm
    elif collision_grad_norm > 0.0:
        collision_elastic_ratio = float("inf")
    else:
        collision_elastic_ratio = np.nan

    return {
        "elastic-grad-norm": elastic_grad_norm,
        "collision-grad-norm": collision_grad_norm,
        "collision-elastic-grad-ratio": collision_elastic_ratio,
    }


def collect_step_record(
    *,
    body: pv.UnstructuredGrid,
    forward: Any,
    model_state: Any,
    opt_state: Any | None,
    elastic_warp_model: Any,
) -> MetricRecord:
    from liblaf.apple.common import GLOBAL_POINT_ID

    global_ids = body.point_data[GLOBAL_POINT_ID.vtk]
    displacement = np.asarray(model_state.u)[global_ids]
    deformed_points = body.points + displacement
    max_displacement = float(np.linalg.norm(displacement, axis=1).max())
    split_gradients = collect_split_gradient_norms(
        forward=forward,
        model_state=model_state,
        elastic_warp_model=elastic_warp_model,
    )

    if opt_state is None:
        grad = forward.problem.grad(model_state)
        grad_norm = scalar_float(jnp.linalg.norm(grad))
        record = {
            "step": 0.0,
            "objective": scalar_float(forward.problem.fun(model_state)),
            "actual-decrease": np.nan,
            "alpha": np.nan,
            "line-search-steps": 0.0,
            "line-search-ok": np.nan,
            "grad-norm": grad_norm,
            "relative-grad-norm": 1.0,
            "direction-norm": np.nan,
            "step-norm": 0.0,
            "step-max-norm": 0.0,
            "max-displacement": max_displacement,
            "min-y": float(deformed_points[:, 1].min()),
            "max-y": float(deformed_points[:, 1].max()),
            "hess-damping-factor": np.nan,
            "hess-diag-mean": np.nan,
        }
        record.update(split_gradients)
        return record

    line_search = opt_state.line_search_state
    convergence = opt_state.convergence_state
    hess_damping = opt_state.hess_damping_state
    step_direction = line_search.alpha * opt_state.direction
    record = {
        "step": scalar_float(opt_state.n_steps),
        "objective": scalar_float(line_search.f_alpha),
        "actual-decrease": scalar_float(line_search.f0 - line_search.f_alpha),
        "alpha": scalar_float(line_search.alpha),
        "line-search-steps": scalar_float(line_search.step),
        "line-search-ok": scalar_bool(line_search.ok),
        "grad-norm": scalar_float(convergence.grad_norm),
        "relative-grad-norm": scalar_float(
            convergence.grad_norm / convergence.grad_norm_first
        ),
        "direction-norm": scalar_float(jnp.linalg.norm(opt_state.direction)),
        "step-norm": scalar_float(jnp.linalg.norm(step_direction)),
        "step-max-norm": scalar_float(jnp.linalg.norm(step_direction, ord=jnp.inf)),
        "max-displacement": max_displacement,
        "min-y": float(deformed_points[:, 1].min()),
        "max-y": float(deformed_points[:, 1].max()),
        "hess-damping-factor": scalar_float(hess_damping.factor),
        "hess-diag-mean": scalar_float(hess_damping.hess_diag_mean),
    }
    record.update(split_gradients)
    return record


def log_step_metrics(case: str, record: MetricRecord) -> None:
    cherries.log_metrics(
        {
            f"{case}/optimizer/{metric.replace('-', '_')}": value
            for metric, value in record.items()
            if metric != "step"
        },
        step=int(record["step"]),
    )


def make_step_mesh(
    *,
    body: pv.UnstructuredGrid,
    forward: Any,
    model_state: Any,
    step_displacement: np.ndarray,
    record: MetricRecord,
    cfg: Config,
    activation_value: np.ndarray,
    with_collision: bool,
) -> pv.UnstructuredGrid:
    from liblaf.apple.common import GLOBAL_POINT_ID

    global_ids = body.point_data[GLOBAL_POINT_ID.vtk]
    displacement = np.asarray(model_state.u)[global_ids]
    step_displacement = step_displacement[global_ids]

    mesh = body.copy(deep=True)
    mesh.point_data["Displacement"] = displacement
    mesh.point_data["Solution"] = displacement
    mesh.point_data["StepDisplacement"] = step_displacement
    mesh.point_data["DeformedPoint"] = mesh.points + displacement
    mesh.point_data["Step"] = np.full(mesh.n_points, record["step"])
    for metric, value in record.items():
        mesh.point_data[metric_field_name(metric)] = np.full(mesh.n_points, value)
        mesh.field_data[metric_field_name(metric)] = np.asarray([value])

    mesh.field_data["WithCollision"] = np.asarray([int(with_collision)])
    mesh.field_data["ForceScale"] = np.asarray([cfg.force_scale])
    mesh.field_data["SelfCollisionStiffness"] = np.asarray(
        [cfg.collision_stiffness_scale * cfg.E]
    )
    mesh.field_data["StiffnessRatio"] = np.asarray([cfg.stiffness_ratio])
    mesh.field_data["SmasPrestrain"] = np.asarray([cfg.smas_prestrain])
    mesh.field_data["SmasActivation"] = activation_value[np.newaxis, :]
    mesh.field_data["NFree"] = np.asarray([forward.model.n_free])
    return mesh


def write_step_mesh(
    *,
    writer: Any,
    body: pv.UnstructuredGrid,
    forward: Any,
    model_state: Any,
    opt_state: Any | None,
    record: MetricRecord,
    cfg: Config,
    activation_value: np.ndarray,
    with_collision: bool,
) -> None:
    if opt_state is None:
        step_displacement = np.zeros_like(np.asarray(model_state.u))
    else:
        step_displacement = np.asarray(
            forward.model.dof_map.to_full_grad(
                opt_state.line_search_state.alpha * opt_state.direction
            )
        )
    writer.append(
        make_step_mesh(
            body=body,
            forward=forward,
            model_state=model_state,
            step_displacement=step_displacement,
            record=record,
            cfg=cfg,
            activation_value=activation_value,
            with_collision=with_collision,
        ),
        time=record["step"],
    )


def run_forward_with_progress(
    forward: Any,
    case: str,
    *,
    body: pv.UnstructuredGrid,
    cfg: Config,
    activation_value: np.ndarray,
    with_collision: bool,
    series_path: Path,
    plot_records_by_case: dict[str, list[MetricRecord]],
) -> tuple[Any, list[MetricRecord]]:
    from liblaf.peach.optim import Result

    problem = forward.problem
    model_state = forward.state
    opt_state = forward.optimizer.init(problem, model_state, forward.free)
    result = Result.UNKNOWN_ERROR
    elastic_warp_model = make_elastic_warp_model(forward)
    records = plot_records_by_case.setdefault(
        case_name(with_collision=with_collision), []
    )
    records.clear()

    writer_context = (
        melon.io.SeriesWriter(series_path, clear=True)
        if cfg.write_step_artifacts
        else contextlib.nullcontext(None)
    )
    with writer_context as writer:
        record = collect_step_record(
            body=body,
            forward=forward,
            model_state=model_state,
            opt_state=None,
            elastic_warp_model=elastic_warp_model,
        )
        records.append(record)
        log_step_metrics(case, record)
        if cfg.write_step_artifacts:
            write_step_mesh(
                writer=writer,
                body=body,
                forward=forward,
                model_state=model_state,
                opt_state=None,
                record=record,
                cfg=cfg,
                activation_value=activation_value,
                with_collision=with_collision,
            )
            plot_metrics(plot_records_by_case)

        while True:
            model_state, opt_state = forward.optimizer.step(
                problem, model_state, opt_state
            )
            record = collect_step_record(
                body=body,
                forward=forward,
                model_state=model_state,
                opt_state=opt_state,
                elastic_warp_model=elastic_warp_model,
            )
            records.append(record)
            log_step_metrics(case, record)
            if cfg.write_step_artifacts:
                write_step_mesh(
                    writer=writer,
                    body=body,
                    forward=forward,
                    model_state=model_state,
                    opt_state=opt_state,
                    record=record,
                    cfg=cfg,
                    activation_value=activation_value,
                    with_collision=with_collision,
                )
                plot_metrics(plot_records_by_case)
            done, result = forward.optimizer.terminate(problem, model_state, opt_state)
            if bool(np.asarray(done)):
                break

    solution = forward.optimizer.postprocess(problem, model_state, opt_state, result)
    forward.state = model_state
    return solution, records


def solve(
    base_body: pv.UnstructuredGrid,
    cfg: Config,
    *,
    with_collision: bool,
    plot_records_by_case: dict[str, list[MetricRecord]],
) -> tuple[pv.UnstructuredGrid, np.ndarray, dict[str, float | str], list[MetricRecord]]:
    from liblaf.apple.common import GLOBAL_POINT_ID
    from liblaf.apple.forward import Forward

    body = base_body.copy(deep=True)
    model, activation_value = build_model(
        body,
        E=cfg.E,
        nu=cfg.nu,
        stiffness_ratio=cfg.stiffness_ratio,
        force_scale=cfg.force_scale,
        collision_stiffness_scale=cfg.collision_stiffness_scale,
        smas_prestrain=cfg.smas_prestrain,
        with_collision=with_collision,
    )
    forward = Forward(model)

    initial_energy = float(forward.problem.fun(forward.state))
    case = "self_collision" if with_collision else "no_collision"
    series_path = (
        cfg.output_collision_steps if with_collision else cfg.output_no_collision_steps
    )
    solution, records = run_forward_with_progress(
        forward,
        case,
        body=body,
        cfg=cfg,
        activation_value=activation_value,
        with_collision=with_collision,
        series_path=series_path,
        plot_records_by_case=plot_records_by_case,
    )
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
    result.field_data["ForceScale"] = np.asarray([cfg.force_scale])
    result.field_data["SelfCollisionStiffness"] = np.asarray(
        [cfg.collision_stiffness_scale * cfg.E]
    )
    result.field_data["SmasPrestrain"] = np.asarray([cfg.smas_prestrain])
    result.field_data["SmasActivation"] = activation_value[np.newaxis, :]
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
    return result, displacement, metrics, records


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
    comparison.field_data["ForceScale"] = np.asarray([cfg.force_scale])
    comparison.field_data["SelfCollisionStiffness"] = np.asarray(
        [cfg.collision_stiffness_scale * cfg.E]
    )
    comparison.field_data["StiffnessRatio"] = np.asarray([cfg.stiffness_ratio])
    comparison.field_data["SmasPrestrain"] = np.asarray([cfg.smas_prestrain])
    comparison.field_data["SmasActivation"] = diagonal_activation_from_prestrain(
        cfg.smas_prestrain
    )[np.newaxis, :]
    return comparison


def metric_values(records: list[MetricRecord], metric: str) -> np.ndarray:
    return np.asarray([record[metric] for record in records], dtype=float)


def positive_for_log(values: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(values) & (values > 0.0), values, np.nan)


def plot_metric_figure(
    *,
    records_by_case: dict[str, list[MetricRecord]],
    plot_name: str,
    title: str,
    ylabel: str,
    log_scale: bool,
) -> None:
    fig, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for name, records in records_by_case.items():
        steps = metric_values(records, "step")
        values = metric_values(records, plot_name)
        if log_scale:
            values = positive_for_log(values)
            axis.semilogy(steps, values, marker=".", label=name.replace("_", " "))
        else:
            axis.plot(steps, values, marker=".", label=name.replace("_", " "))

    axis.set_title(title)
    axis.set_xlabel("optimizer step")
    axis.set_ylabel(ylabel)
    axis.grid(visible=True, alpha=0.25)
    axis.legend()

    output = metric_plot_path(plot_name)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


def plot_metrics(records_by_case: dict[str, list[MetricRecord]]) -> None:
    for plot_name, (title, ylabel, log_scale) in METRIC_PLOTS.items():
        plot_metric_figure(
            records_by_case=records_by_case,
            plot_name=plot_name,
            title=title,
            ylabel=ylabel,
            log_scale=log_scale,
        )


def main(cfg: Config) -> None:
    if jax.default_backend() != "gpu":
        msg = "This experiment uses Warp JAX FFI and needs JAX's active backend to be GPU."
        raise RuntimeError(msg)

    wp.config.mode = "release"
    wp.init()

    body = make_body_mesh(cfg)
    records_by_case: dict[str, list[MetricRecord]] = {}

    no_collision_result, no_collision_u, no_collision_metrics, no_collision_records = (
        solve(
            body,
            cfg,
            with_collision=False,
            plot_records_by_case=records_by_case,
        )
    )
    melon.save(cfg.output_no_collision, no_collision_result)
    collision_result, collision_u, collision_metrics, collision_records = solve(
        body,
        cfg,
        with_collision=True,
        plot_records_by_case=records_by_case,
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
    if cfg.write_step_artifacts:
        plot_metrics(records_by_case)

    metrics = {
        "E": cfg.E,
        "nu": cfg.nu,
        "stiffness_ratio": cfg.stiffness_ratio,
        "force_scale": cfg.force_scale,
        "self_collision_stiffness": cfg.collision_stiffness_scale * cfg.E,
        "write_step_artifacts": cfg.write_step_artifacts,
        "smas_prestrain_x": cfg.smas_prestrain[0],
        "smas_prestrain_y": cfg.smas_prestrain[1],
        "smas_prestrain_z": cfg.smas_prestrain[2],
        "n_points": float(body.n_points),
        "n_cells": float(body.n_cells),
        "smas_active_cells": float(
            np.count_nonzero(
                np.asarray(body.cell_data[SMAS_FRACTION]) > SMAS_ACTIVE_TOL
            )
        ),
        "no_collision": no_collision_metrics,
        "collision": collision_metrics,
        "step_counts": {
            case_name(with_collision=False): len(no_collision_records),
            case_name(with_collision=True): len(collision_records),
        },
        "step_series": (
            {
                case_name(with_collision=False): (
                    cfg.output_no_collision_steps.as_posix()
                ),
                case_name(with_collision=True): cfg.output_collision_steps.as_posix(),
            }
            if cfg.write_step_artifacts
            else {}
        ),
        "metrics_plots": metric_plot_paths() if cfg.write_step_artifacts else {},
    }
    cherries.log_metrics(metrics)
    print(f"saved: {cfg.output_no_collision}")
    print(f"saved: {cfg.output_collision}")
    print(f"saved: {cfg.output_comparison}")
    if cfg.write_step_artifacts:
        print(f"saved: {cfg.output_no_collision_steps}")
        print(f"saved: {cfg.output_collision_steps}")
        print(f"saved figures: {EXPERIMENT_DIR / 'fig'}")
    else:
        print("step artifacts disabled: WRITE_STEP_ARTIFACTS=false")
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
