import csv
import json
import math
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast

import attrs
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pyvista as pv
import warp as wp
from environs import env
from jaxtyping import Array, Float, Integer
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
from liblaf.apple.inverse import AdjointLinearSystem, Inverse, Loss, PointToPointLoss
from liblaf.apple.inverse._inverse import InverseObjective
from liblaf.apple.model import Forward, Free, Model, ModelBuilder, ModelMaterials
from liblaf.apple.optim import PNCG
from liblaf.peach.linalg import LinearSolver
from liblaf.peach.linalg import utils as linalg_utils
from liblaf.peach.optim import Objective, Optax, Optimizer

from liblaf import cherries, jarp, melon
from liblaf.apple.warp import WarpStableNeoHookean, WarpStableNeoHookeanMuscle

type EnergyMaterials = Mapping[str, Array]
type MetricRecord = dict[str, bool | float | int | str]
type Scalar = Float[Array, ""]
type Vector = Float[Array, "points dim"]

plt.switch_backend("Agg")

SUFFIX: str = "-smas46-muscle46"
METRIC_PLOT_NAMES: tuple[str, ...] = (
    "loss",
    "point-to-point-l2",
    "relative-activation",
    "activation-error",
    "forward-grad-norm-absolute",
    "forward-grad-norm-relative",
    "adjoint-relative-residual",
    "inverse-grad-absolute",
)


class Config(cherries.BaseConfig):
    input: Path = cherries.input(f"10-input{SUFFIX}.vtu")
    cases: str = env.str("INVERSE_CASES", "easy,zero")
    write_series: bool = env.bool("WRITE_SERIES", True)
    reuse_target: bool = env.bool("REUSE_TARGET", True)

    arch_height: float = env.float("ARCH_HEIGHT", 2.0)
    easy_initial_activation: float = env.float("EASY_INITIAL_ACTIVATION", 1.98)
    zero_initial_log_activation: float = env.float("ZERO_INITIAL_LOG_ACTIVATION", 0.0)
    target_activation: float = env.float("TARGET_ACTIVATION", 2.0)
    smas_prestrain: float = env.float("SMAS_PRESTRAIN", 1.3)
    stiffness_ratio: float = env.float("STIFFNESS_RATIO", 1.0e2)
    poisson_lambda_ratio: float = env.float("POISSON_LAMBDA_RATIO", 49.0)
    side_tolerance: float = env.float("SIDE_TOLERANCE", 1e-2)

    inverse_lr: float = env.float("INVERSE_LR", 0.05)
    inverse_max_steps: int = env.int("INVERSE_MAX_STEPS", 80)
    inverse_patience: int = env.int("INVERSE_PATIENCE", 8)
    inverse_rtol: float = env.float("INVERSE_RTOL", 1e-8)
    activation_tolerance: float = env.float("ACTIVATION_TOLERANCE", 5e-2)
    log_activation_min: float = env.float("LOG_ACTIVATION_MIN", -2.0)
    log_activation_max: float = env.float("LOG_ACTIVATION_MAX", 2.0)


def diagonal_activation_vector(
    xz_activation: Float[Array, "*batch"],
) -> Float[Array, "*batch 6"]:
    y_activation = jnp.reciprocal(jnp.square(xz_activation))
    zeros = jnp.zeros_like(xz_activation)
    return jnp.stack(
        [
            xz_activation - 1.0,
            y_activation - 1.0,
            xz_activation - 1.0,
            zeros,
            zeros,
            zeros,
        ],
        axis=-1,
    )


def make_smas_activation(smas_prestrain: float) -> np.ndarray:
    return np.asarray(diagonal_activation_vector(jnp.asarray(smas_prestrain)))


def directional_activation_vector(
    x_activation: Float[Array, "*batch"],
    y_activation: Float[Array, "*batch"],
    z_activation: Float[Array, "*batch"],
) -> Float[Array, "*batch 6"]:
    zeros = jnp.zeros_like(x_activation)
    return jnp.stack(
        [
            x_activation - 1.0,
            y_activation - 1.0,
            z_activation - 1.0,
            zeros,
            zeros,
            zeros,
        ],
        axis=-1,
    )


def log_param_to_relative_activation(
    log_relative_activation: Float[Array, "*batch"],
    *,
    log_min: float,
    log_max: float,
) -> Float[Array, "*batch"]:
    return jnp.exp(jnp.clip(log_relative_activation, log_min, log_max))


def make_muscle_x_activation(
    log_relative_activation: Float[Array, "*batch"],
    *,
    smas_prestrain: float,
    log_min: float,
    log_max: float,
) -> Float[Array, "*batch 6"]:
    relative_activation = log_param_to_relative_activation(
        log_relative_activation, log_min=log_min, log_max=log_max
    )
    x_activation = relative_activation * smas_prestrain
    y_activation = jnp.ones_like(x_activation) / smas_prestrain**2
    z_activation = jnp.ones_like(x_activation) * smas_prestrain
    return directional_activation_vector(x_activation, y_activation, z_activation)


def compute_arch_profile(
    mesh: pv.UnstructuredGrid,
    arch_height: float,
    points: np.ndarray | None = None,
) -> np.ndarray:
    if points is None:
        points = mesh.points
    bottom_mask = mesh.points[:, 1] < 5e-3
    bottom_points = mesh.points[bottom_mask]

    x_min, x_max = bottom_points[:, 0].min(), bottom_points[:, 0].max()
    z_min, z_max = bottom_points[:, 2].min(), bottom_points[:, 2].max()
    x_center = 0.5 * (x_min + x_max)
    z_center = 0.5 * (z_min + z_max)
    x_hat = 2.0 * (points[:, 0] - x_center) / (x_max - x_min)
    z_hat = 2.0 * (points[:, 2] - z_center) / (z_max - z_min)
    arch = arch_height * (1.0 - x_hat**2) * (1.0 - z_hat**2)
    return np.clip(arch, 0.0, None)


def compute_box_side_mask(mesh: pv.UnstructuredGrid, tolerance: float) -> np.ndarray:
    points = mesh.points
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    z_min, z_max = points[:, 2].min(), points[:, 2].max()
    return (
        (points[:, 0] <= x_min + tolerance)
        | (points[:, 0] >= x_max - tolerance)
        | (points[:, 2] <= z_min + tolerance)
        | (points[:, 2] >= z_max - tolerance)
    )


def apply_box_arch_dirichlet(
    mesh: pv.UnstructuredGrid, *, arch_height: float, side_tolerance: float
) -> pv.UnstructuredGrid:
    bottom_mask = mesh.points[:, 1] < 5e-3
    side_mask = compute_box_side_mask(mesh, side_tolerance)
    arch = compute_arch_profile(mesh, arch_height, mesh.points[bottom_mask])

    mesh.point_data[DIRICHLET_MASK] = np.zeros((mesh.n_points, 3), dtype=bool)
    mesh.point_data[DIRICHLET_MASK][bottom_mask] = True
    mesh.point_data[DIRICHLET_MASK][side_mask] = True
    mesh.point_data[DIRICHLET_VALUE] = np.zeros(
        (mesh.n_points, 3), dtype=mesh.points.dtype
    )
    mesh.point_data[DIRICHLET_VALUE][bottom_mask, 1] = arch
    mesh.point_data[DIRICHLET_VALUE][side_mask] = 0.0
    mesh.point_data["BottomDirichletMask"] = bottom_mask
    mesh.point_data["SideDirichletMask"] = side_mask
    return mesh


def initialize_parabolic_guess(
    model: Model, mesh: pv.UnstructuredGrid, arch_height: float
) -> None:
    u_full = np.zeros((model.n_points, model.dim), dtype=mesh.points.dtype)
    u_full[mesh.point_data[GLOBAL_POINT_ID], 1] = compute_arch_profile(
        mesh, arch_height
    )
    model.u_free = model.dirichlet.get_free(jnp.asarray(u_full))


def configure_forward_solver(forward: Forward) -> None:
    optimizer = cast("PNCG", forward.optimizer)
    optimizer.convergence = attrs.evolve(
        optimizer.convergence,
        acceptable_relative_gradient_norm=jnp.asarray(1e-3),
        target_relative_gradient_norm=jnp.asarray(1e-5),
    )


def keep_forward_displacement_warm_start(forward: Forward) -> None:
    forward.last_successful_material_values = None


def build_model(
    mesh: pv.UnstructuredGrid,
    *,
    muscle_log_relative_activation: float,
    cfg: Config,
    muscle_requires_grad: bool,
) -> Model:
    builder = ModelBuilder()
    mesh = builder.add_points(mesh)
    mesh = apply_box_arch_dirichlet(
        mesh, arch_height=cfg.arch_height, side_tolerance=cfg.side_tolerance
    )

    muscle_frac = mesh.cell_data[MUSCLE_FRACTION]
    smas_frac = mesh.cell_data[SMAS_FRACTION]
    aponeurosis_frac = np.clip(smas_frac - muscle_frac, 0.0, None)
    fat_frac = np.clip(1.0 - smas_frac, 0.0, None)

    mesh.cell_data[ACTIVATION] = np.zeros((mesh.n_cells, 6))
    mesh.cell_data[ACTIVATION][smas_frac > 1e-3] = make_smas_activation(
        cfg.smas_prestrain
    )
    mesh.cell_data[ACTIVATION][muscle_frac > 1e-3] = np.asarray(
        make_muscle_x_activation(
            jnp.asarray(muscle_log_relative_activation),
            smas_prestrain=cfg.smas_prestrain,
            log_min=cfg.log_activation_min,
            log_max=cfg.log_activation_max,
        )
    )
    builder.add_dirichlet(mesh)

    mesh.cell_data["Fraction"] = fat_frac
    mesh.cell_data[MU] = np.full((mesh.n_cells,), 1.0)
    mesh.cell_data[LAMBDA] = np.full((mesh.n_cells,), cfg.poisson_lambda_ratio)
    builder.add_energy(WarpStableNeoHookean.from_pyvista(mesh))

    mesh.cell_data["Fraction"] = aponeurosis_frac
    mesh.cell_data[MU] = np.full((mesh.n_cells,), cfg.stiffness_ratio)
    mesh.cell_data[LAMBDA] = np.full(
        (mesh.n_cells,), cfg.poisson_lambda_ratio * cfg.stiffness_ratio
    )
    builder.add_energy(WarpStableNeoHookeanMuscle.from_pyvista(mesh))

    mesh.cell_data["Fraction"] = muscle_frac
    mesh.cell_data[MU] = np.full((mesh.n_cells,), cfg.stiffness_ratio)
    mesh.cell_data[LAMBDA] = np.full(
        (mesh.n_cells,), cfg.poisson_lambda_ratio * cfg.stiffness_ratio
    )
    requires_grad = ("activation",) if muscle_requires_grad else ()
    builder.add_energy(
        WarpStableNeoHookeanMuscle.from_pyvista(
            mesh, requires_grad=requires_grad, name="muscle"
        )
    )

    return builder.finalize()


def solve_forward(model: Model) -> Optimizer.Solution:
    forward = Forward(model)
    configure_forward_solver(forward)
    solution = forward.step()
    if not bool(np.asarray(solution.success)):
        message = f"forward solve failed: {solution.result}"
        raise RuntimeError(message)
    return solution


def target_path() -> Path:
    return cherries.output(
        f"31-inverse{SUFFIX}-activation-stable-neo-hookean-target.vtu"
    )


def target_matches_source(
    target: pv.UnstructuredGrid, source: pv.UnstructuredGrid, cfg: Config
) -> bool:
    required = (
        target.n_points == source.n_points,
        target.n_cells == source.n_cells,
        "Solution" in target.point_data,
        "TargetRelativeActivation" in target.cell_data,
        "SideDirichletMask" in target.point_data,
        "SideTolerance" in target.field_data,
        "ActivationModeCode" in target.field_data,
        "TargetActualActivationX" in target.cell_data,
    )
    if not all(required):
        return False
    side_tolerance_matches = np.isclose(
        float(target.field_data["SideTolerance"][0]), cfg.side_tolerance
    )
    activation_matches = np.isclose(
        target.cell_data["TargetRelativeActivation"][0], cfg.target_activation
    )
    mode_matches = int(target.field_data["ActivationModeCode"][0]) == 1
    return bool(
        side_tolerance_matches
        and activation_matches
        and mode_matches
        and np.any(target.point_data["SideDirichletMask"])
        and np.array_equal(
            target.point_data["SideDirichletMask"],
            compute_box_side_mask(source, cfg.side_tolerance),
        )
    )


def make_target_mesh(cfg: Config, source: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    output = target_path()
    if cfg.reuse_target and output.exists():
        target = melon.load_unstructured_grid(output)
        if target_matches_source(target, source, cfg):
            return target

    target = source.copy()
    target_log_activation = math.log(cfg.target_activation)
    model = build_model(
        target,
        muscle_log_relative_activation=target_log_activation,
        cfg=cfg,
        muscle_requires_grad=False,
    )
    initialize_parabolic_guess(model, target, cfg.arch_height)
    solve_forward(model)
    target.point_data["Solution"] = np.asarray(
        model.u_full[target.point_data[GLOBAL_POINT_ID]]
    )
    target.cell_data["TargetRelativeActivation"] = np.full(
        (target.n_cells,), cfg.target_activation
    )
    target.cell_data["TargetActualActivationX"] = np.full(
        (target.n_cells,), cfg.target_activation * cfg.smas_prestrain
    )
    target.cell_data["TargetActualActivationY"] = np.full(
        (target.n_cells,), 1.0 / cfg.smas_prestrain**2
    )
    target.cell_data["TargetActualActivationZ"] = np.full(
        (target.n_cells,), cfg.smas_prestrain
    )
    target.field_data["SideTolerance"] = np.asarray([cfg.side_tolerance])
    target.field_data["ActivationModeCode"] = np.asarray([1])
    melon.save(output, target)
    return target


def make_target_u_full(
    model: Model, mesh: pv.UnstructuredGrid, target: pv.UnstructuredGrid
) -> np.ndarray:
    u_full = np.zeros((model.n_points, model.dim), dtype=np.asarray(model.u_full).dtype)
    u_full[mesh.point_data[GLOBAL_POINT_ID]] = np.asarray(
        target.point_data["Solution"], dtype=u_full.dtype
    )
    return u_full


def initialize_target_guess(
    model: Model, mesh: pv.UnstructuredGrid, target: pv.UnstructuredGrid
) -> Free:
    u_free = model.dirichlet.get_free(
        jnp.asarray(make_target_u_full(model, mesh, target))
    )
    model.u_free = u_free
    return u_free


def log_forward_pass_metrics(
    solution: Optimizer.Solution, *, init_grad: Vector
) -> None:
    cherries.log_metrics(
        {
            "forward": {
                "init_grad_norm": jnp.linalg.norm(init_grad),
                "init_grad_norm_inf": jnp.linalg.norm(init_grad, ord=jnp.inf),
                "final_grad_norm": jnp.linalg.norm(solution.state.grad),
                "final_grad_norm_inf": jnp.linalg.norm(
                    solution.state.grad, ord=jnp.inf
                ),
                "relative_decrease": solution.stats.relative_decrease,
                "result": str(solution.result),
                "success": solution.success,
            }
        }
    )


@jarp.define
class ActivationInverse(Inverse[Scalar]):
    muscle_indices: Integer[Array, " muscle_cells"] = jarp.field()
    full_activation: Float[Array, "cells 6"] = jarp.field()
    smas_prestrain: float = jarp.static(kw_only=True)
    log_activation_min: float = jarp.static(kw_only=True)
    log_activation_max: float = jarp.static(kw_only=True)
    fallback_u_free: Free | None = jarp.array(default=None, kw_only=True)
    last_adjoint_relative_residual: Scalar = jarp.array(default=jnp.inf, kw_only=True)

    def make_materials(self, params: Scalar) -> ModelMaterials:
        muscle_activation = make_muscle_x_activation(
            params,
            smas_prestrain=self.smas_prestrain,
            log_min=self.log_activation_min,
            log_max=self.log_activation_max,
        )
        activation = self.full_activation.at[self.muscle_indices].set(muscle_activation)
        return {"muscle": {"activation": activation}}

    def update(self, materials: ModelMaterials) -> None:
        self.model.update_materials(materials)
        if not self.last_forward_success:
            if self.forward.last_successful_u_free is not None:
                self.model.u_free = self.forward.last_successful_u_free
            elif self.fallback_u_free is not None:
                self.model.u_free = self.fallback_u_free
            else:
                self.model.u_free = jnp.zeros_like(self.model.u_free)
        self.forward.state = self.model.init_state(self.model.u_full)
        init_grad = self.model.grad(self.forward.state)
        solution = self.forward.step()
        keep_forward_displacement_warm_start(self.forward)
        log_forward_pass_metrics(solution, init_grad=init_grad)
        self.last_forward_success = jnp.asarray(solution.success)
        if not bool(np.asarray(solution.success)):
            message = f"forward solve failed during inverse: {solution.result}"
            raise RuntimeError(message)

    def loss_and_grad(
        self, materials: ModelMaterials
    ) -> tuple[Scalar, Vector, dict[str, dict[str, Array]]]:
        loss_value: Scalar = jnp.zeros(())
        dLdu: Vector = jnp.zeros_like(self.model.u_full)
        dLdq: dict[str, dict[str, Array]] = {
            energy_id: {
                mat_name: jnp.zeros_like(mat_value)
                for mat_name, mat_value in energy.items()
            }
            for energy_id, energy in materials.items()
        }
        for loss in self.losses:
            loss_i, (dLdu_i, dLdq_i) = loss.value_and_grad(self.model.u_full, materials)
            cherries.log_metric(loss.name, loss_i)
            loss_value += loss_i
            dLdu += dLdu_i
            dLdq = jax.tree.map(jnp.add, dLdq, dLdq_i)
        cherries.log_metric("loss", loss_value)
        return loss_value, dLdu, dLdq

    def value_and_grad(
        self, materials: ModelMaterials
    ) -> tuple[Scalar, ModelMaterials]:
        loss_value: Scalar
        dLdu: Vector
        dLdq: dict[str, dict[str, Array]]
        loss_value, dLdu, dLdq = self.loss_and_grad(materials)
        p = self.adjoint(dLdu)
        mixed_prod: ModelMaterials = self.model.mixed_derivative_prod(
            self.forward.state, p
        )
        for energy_id, energy in mixed_prod.items():
            for mat_name, value in energy.items():
                dLdq[energy_id][mat_name] += value
        return loss_value, dLdq

    def adjoint(self, dLdu: Vector) -> Vector:
        system = AdjointLinearSystem.new(self.model, self.forward.state, dLdu)
        p_free: Free = (
            self.adjoint_vector
            if self.last_adjoint_success
            else jnp.zeros_like(self.model.u_free)
        )
        solution: LinearSolver.Solution = self.adjoint_solver.solve(system, p_free)
        relative_residual = linalg_utils.relative_residual(
            system.matvec, solution.params, system.b
        )
        cherries.log_metrics(
            {
                "adjoint": {
                    "success": solution.success,
                    "relative_residual": relative_residual,
                }
            }
        )
        self.last_adjoint_success = jnp.asarray(solution.success)
        self.last_adjoint_relative_residual = relative_residual
        self.adjoint_vector = solution.params
        return self.model.dirichlet.to_full(self.adjoint_vector, dirichlet=0.0)

    def solve(
        self, params: Scalar, callback: Optimizer.Callback | None = None
    ) -> Scalar:
        params_flat: Vector
        structure: jarp.Structure[Scalar]
        params_flat, structure = jarp.ravel(params)
        objective = InverseObjective(inverse=self, structure=structure)
        solution: Optimizer.Solution
        solution, _ = self.optimizer.minimize(
            objective, params, params_flat, callback=callback
        )
        return structure.unravel(solution.params)


def build_inverse(
    cfg: Config,
    *,
    mesh: pv.UnstructuredGrid,
    target: pv.UnstructuredGrid,
    forward: Forward,
    fallback_u_free: Free,
) -> ActivationInverse:
    surface_indices: Integer[Array, " surface_points"] = mesh.surface_indices()
    muscle_indices: Integer[Array, " muscle_cells"] = jnp.flatnonzero(
        mesh.cell_data[MUSCLE_FRACTION] > 1e-3
    )
    losses: list[Loss] = [
        PointToPointLoss(
            indices=jnp.asarray(surface_indices),
            target=jnp.asarray(target.point_data["Solution"][surface_indices]),
        )
    ]
    return ActivationInverse(
        forward=forward,
        losses=losses,
        muscle_indices=muscle_indices,
        full_activation=jnp.asarray(mesh.cell_data[ACTIVATION]),
        smas_prestrain=cfg.smas_prestrain,
        log_activation_min=cfg.log_activation_min,
        log_activation_max=cfg.log_activation_max,
        fallback_u_free=fallback_u_free,
        optimizer=Optax(
            optax.adam(cfg.inverse_lr),
            max_steps=jnp.asarray(cfg.inverse_max_steps),
            patience=jnp.asarray(cfg.inverse_patience),
            rtol=jnp.asarray(cfg.inverse_rtol),
        ),
    )


def relative_activation_from_log_param(cfg: Config, log_activation: float) -> float:
    clipped = np.clip(log_activation, cfg.log_activation_min, cfg.log_activation_max)
    return float(np.exp(clipped))


def fill_inverse_outputs(
    mesh: pv.UnstructuredGrid,
    *,
    cfg: Config,
    target: pv.UnstructuredGrid,
    forward: Forward,
    model: Model,
    log_relative_activation: float,
) -> None:
    relative_activation = relative_activation_from_log_param(
        cfg, log_relative_activation
    )
    mesh.point_data["InverseSolution"] = np.asarray(
        forward.u_full[mesh.point_data[GLOBAL_POINT_ID]]
    )
    mesh.point_data["PointToPoint"] = (
        mesh.point_data["InverseSolution"] - target.point_data["Solution"]
    )
    mesh.point_data["PointToPointNorm"] = np.linalg.norm(
        mesh.point_data["PointToPoint"], axis=1
    )
    inverse_activation = cast(
        "wp.array", model.get_energy("muscle").materials.activation
    ).numpy()
    mesh.cell_data["InverseActivation"] = inverse_activation
    mesh.cell_data["ActivationDiff"] = inverse_activation - target.cell_data[ACTIVATION]
    mesh.cell_data["InverseLogRelativeActivation"] = np.full(
        (mesh.n_cells,), log_relative_activation
    )
    mesh.cell_data["InverseRelativeActivation"] = np.full(
        (mesh.n_cells,), relative_activation
    )
    mesh.cell_data["InverseActualActivationX"] = np.full(
        (mesh.n_cells,), relative_activation * cfg.smas_prestrain
    )
    mesh.cell_data["InverseActualActivationY"] = np.full(
        (mesh.n_cells,), 1.0 / cfg.smas_prestrain**2
    )
    mesh.cell_data["InverseActualActivationZ"] = np.full(
        (mesh.n_cells,), cfg.smas_prestrain
    )


def save_summary(path: Path, values: Any) -> None:
    path.write_text(json.dumps(values, indent=2))


def assert_finite_outputs(mesh: pv.UnstructuredGrid) -> None:
    for collection in (mesh.point_data, mesh.cell_data):
        for name, values in collection.items():
            array = np.asarray(values)
            if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
                message = f"non-finite values in output array {name!r}"
                raise FloatingPointError(message)


def output_stem(case_name: str) -> str:
    return f"31-inverse{SUFFIX}-activation-stable-neo-hookean-{case_name}"


def output_series_path(case_name: str) -> Path:
    return cherries.output(f"{output_stem(case_name)}.vtu.series")


def metrics_json_path(case_name: str) -> Path:
    return cherries.output(f"{output_stem(case_name)}-metrics.json")


def metrics_csv_path(case_name: str) -> Path:
    return cherries.output(f"{output_stem(case_name)}-metrics.csv")


def metrics_plot_path(case_name: str, plot_name: str) -> Path:
    return cherries.output(f"{output_stem(case_name)}-metrics-{plot_name}.png")


def metrics_plot_snapshot_path(case_name: str, plot_name: str, step: int) -> Path:
    return cherries.output(
        f"{output_stem(case_name)}-metrics.d/{step:06d}-{plot_name}.png"
    )


def metrics_plot_paths(case_name: str) -> dict[str, str]:
    return {
        plot_name: metrics_plot_path(case_name, plot_name).as_posix()
        for plot_name in METRIC_PLOT_NAMES
    }


def scalar_float(value: Any) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def scalar_int(value: Any) -> int:
    return int(np.asarray(value).reshape(-1)[0])


def outputs_are_finite(mesh: pv.UnstructuredGrid) -> bool:
    for collection in (mesh.point_data, mesh.cell_data):
        for values in collection.values():
            array = np.asarray(values)
            if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
                return False
    return True


def annotate_step_metrics(mesh: pv.UnstructuredGrid, record: MetricRecord) -> None:
    cell_scalars = {
        "InverseStep": record["step"],
        "InverseLoss": record["loss"],
        "InverseGrad": record["inverse_grad"],
        "ForwardGradNorm": record["forward_grad_norm"],
        "AdjointRelativeResidual": record["adjoint_relative_residual"],
        "PointToPointL2": record["point_to_point_l2"],
        "PointToPointInf": record["point_to_point_inf"],
    }
    for name, value in cell_scalars.items():
        mesh.cell_data[name] = np.full((mesh.n_cells,), float(value))


def collect_step_metrics(
    *,
    cfg: Config,
    mesh: pv.UnstructuredGrid,
    forward: Forward,
    inverse: ActivationInverse,
    step: int,
    log_relative_activation: float,
    loss: float,
    inverse_grad: float,
) -> MetricRecord:
    point_to_point = np.asarray(mesh.point_data["PointToPoint"])
    point_to_point_norm = np.asarray(mesh.point_data["PointToPointNorm"])
    activation_diff = np.asarray(mesh.cell_data["ActivationDiff"])
    relative_activation = relative_activation_from_log_param(
        cfg, log_relative_activation
    )
    record: MetricRecord = {
        "step": step,
        "log_relative_activation": log_relative_activation,
        "relative_activation": relative_activation,
        "actual_activation_x": relative_activation * cfg.smas_prestrain,
        "actual_activation_y": 1.0 / cfg.smas_prestrain**2,
        "actual_activation_z": cfg.smas_prestrain,
        "activation_error": relative_activation - cfg.target_activation,
        "loss": loss,
        "point_to_point_l2": float(np.linalg.norm(point_to_point)),
        "point_to_point_rmse": float(np.sqrt(np.mean(point_to_point_norm**2))),
        "point_to_point_inf": float(
            np.linalg.norm(point_to_point.reshape(-1), ord=np.inf)
        ),
        "activation_diff_l2": float(np.linalg.norm(activation_diff)),
        "activation_diff_inf": float(
            np.linalg.norm(activation_diff.reshape(-1), ord=np.inf)
        ),
        "inverse_grad": inverse_grad,
        "inverse_grad_abs": abs(inverse_grad),
        "adjoint_success": bool(np.asarray(inverse.last_adjoint_success)),
        "adjoint_relative_residual": scalar_float(
            inverse.last_adjoint_relative_residual
        ),
        "forward_success": False,
        "forward_result": "not_evaluated",
        "forward_n_steps": 0,
        "forward_grad_norm": float("inf"),
        "forward_relative_grad_norm": float("inf"),
        "forward_objective": float("nan"),
        "forward_best_objective": float("nan"),
        "finite_outputs": outputs_are_finite(mesh),
    }
    if forward.last_stage_result is not None:
        solver_result = forward.last_stage_result.solver_result
        final_iteration = solver_result.final_iteration
        record.update(
            {
                "forward_success": bool(solver_result.success),
                "forward_result": solver_result.status,
                "forward_n_steps": scalar_int(solver_result.n_steps),
                "forward_grad_norm": scalar_float(final_iteration.gradient_norm),
                "forward_relative_grad_norm": scalar_float(
                    final_iteration.relative_gradient_norm
                ),
                "forward_objective": scalar_float(final_iteration.objective_value),
                "forward_best_objective": scalar_float(
                    final_iteration.best_objective_value
                ),
            }
        )
    return record


def write_metrics_table(records: list[MetricRecord], case_name: str) -> None:
    save_summary(metrics_json_path(case_name), records)
    csv_path = metrics_csv_path(case_name)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def metric_values(records: list[MetricRecord], key: str) -> np.ndarray:
    return np.asarray([float(record[key]) for record in records], dtype=float)


def positive_for_log(values: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(values) & (values > 0.0), values, np.nan)


def plot_metric_figure(
    *,
    records: list[MetricRecord],
    case_name: str,
    plot_name: str,
    title: str,
    ylabel: str,
    values: np.ndarray,
    step: int,
    log_scale: bool = True,
    target: float | None = None,
) -> None:
    steps = metric_values(records, "step")
    fig, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    plot_values = positive_for_log(values) if log_scale else values
    if log_scale:
        axis.semilogy(steps, plot_values)
    else:
        axis.plot(steps, plot_values)
    if target is not None:
        axis.axhline(target, color="0.4", linestyle="--")
    axis.set_title(title)
    axis.set_xlabel("inverse step")
    axis.set_ylabel(ylabel)
    axis.grid(visible=True, alpha=0.25)

    output = metrics_plot_path(case_name, plot_name)
    snapshot = metrics_plot_snapshot_path(case_name, plot_name, step)
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    fig.savefig(snapshot)
    plt.close(fig)


def plot_metrics(records: list[MetricRecord], cfg: Config, case_name: str) -> None:
    steps = metric_values(records, "step")
    step = int(steps[-1])
    plot_metric_figure(
        records=records,
        case_name=case_name,
        plot_name="loss",
        title="Loss",
        ylabel="loss",
        values=metric_values(records, "loss"),
        step=step,
    )
    plot_metric_figure(
        records=records,
        case_name=case_name,
        plot_name="point-to-point-l2",
        title="Point-to-Point Error",
        ylabel="L2 error",
        values=metric_values(records, "point_to_point_l2"),
        step=step,
    )
    plot_metric_figure(
        records=records,
        case_name=case_name,
        plot_name="relative-activation",
        title="Relative Activation",
        ylabel="relative activation",
        values=metric_values(records, "relative_activation"),
        step=step,
        log_scale=False,
        target=cfg.target_activation,
    )
    plot_metric_figure(
        records=records,
        case_name=case_name,
        plot_name="activation-error",
        title="Activation Error",
        ylabel="absolute error",
        values=np.abs(metric_values(records, "activation_error")),
        step=step,
    )
    plot_metric_figure(
        records=records,
        case_name=case_name,
        plot_name="forward-grad-norm-absolute",
        title="Forward Grad Norm Absolute",
        ylabel="absolute gradient norm",
        values=metric_values(records, "forward_grad_norm"),
        step=step,
    )
    plot_metric_figure(
        records=records,
        case_name=case_name,
        plot_name="forward-grad-norm-relative",
        title="Forward Grad Norm Relative",
        ylabel="relative gradient norm",
        values=metric_values(records, "forward_relative_grad_norm"),
        step=step,
    )
    plot_metric_figure(
        records=records,
        case_name=case_name,
        plot_name="adjoint-relative-residual",
        title="Adjoint Relative Residual",
        ylabel="relative residual",
        values=metric_values(records, "adjoint_relative_residual"),
        step=step,
    )
    plot_metric_figure(
        records=records,
        case_name=case_name,
        plot_name="inverse-grad-absolute",
        title="Inverse Grad Absolute",
        ylabel="absolute inverse grad",
        values=metric_values(records, "inverse_grad_abs"),
        step=step,
    )


def write_metric_artifacts(
    records: list[MetricRecord], cfg: Config, case_name: str
) -> None:
    write_metrics_table(records, case_name)
    plot_metrics(records, cfg, case_name)


def new_inverse_selection() -> dict[str, Any]:
    return {
        "log_activation": 0.0,
        "loss": float("inf"),
        "u_free": None,
        "step": 0,
        "forward_success": False,
        "forward_result": "not_evaluated",
        "forward_grad_norm": float("inf"),
    }


def update_inverse_selection(
    selection: dict[str, Any],
    *,
    model: Model,
    record: MetricRecord,
) -> None:
    selection["step"] = record["step"]
    selection["log_activation"] = record["log_relative_activation"]
    selection["loss"] = record["loss"]
    selection["u_free"] = np.asarray(model.u_free).copy()
    selection["forward_success"] = record["forward_success"]
    selection["forward_result"] = record["forward_result"]
    selection["forward_grad_norm"] = record["forward_grad_norm"]


def make_inverse_step_callback(
    *,
    cfg: Config,
    mesh: pv.UnstructuredGrid,
    target: pv.UnstructuredGrid,
    forward: Forward,
    model: Model,
    inverse: ActivationInverse,
    case_name: str,
    best_inverse: dict[str, Any],
    best_converged_inverse: dict[str, Any],
    metric_records: list[MetricRecord],
    writer: Any,
) -> Optimizer.Callback:
    def callback(
        _objective: Objective[Any],
        model_state: Any,
        _opt_state: Optimizer.State,
        _opt_stats: Optimizer.Stats,
    ) -> None:
        step = len(metric_records) + 1
        cherries.set_step(step)
        log_activation = float(np.asarray(model_state).reshape(-1)[0])
        loss = float(np.asarray(_opt_state.value))
        inverse_grad = float(np.asarray(_opt_state.grad).reshape(-1)[0])
        fill_inverse_outputs(
            mesh,
            cfg=cfg,
            target=target,
            forward=forward,
            model=model,
            log_relative_activation=log_activation,
        )
        record = collect_step_metrics(
            cfg=cfg,
            mesh=mesh,
            forward=forward,
            inverse=inverse,
            step=step,
            log_relative_activation=log_activation,
            loss=loss,
            inverse_grad=inverse_grad,
        )
        annotate_step_metrics(mesh, record)
        assert_finite_outputs(mesh)
        metric_records.append(record)
        write_metric_artifacts(metric_records, cfg, case_name)
        cherries.log_metrics(
            {
                "inverse": {
                    "log_relative_activation": log_activation,
                    "relative_activation": record["relative_activation"],
                    "loss": loss,
                    "grad": inverse_grad,
                    "point_to_point_l2": record["point_to_point_l2"],
                    "forward_grad_norm": record["forward_grad_norm"],
                    "adjoint_relative_residual": record["adjoint_relative_residual"],
                }
            }
        )
        if np.isfinite(loss) and loss < best_inverse["loss"]:
            update_inverse_selection(best_inverse, model=model, record=record)
        converged = (
            bool(record["forward_success"])
            and bool(record["finite_outputs"])
            and abs(float(record["activation_error"])) <= cfg.activation_tolerance
        )
        if converged and np.isfinite(loss) and loss < best_converged_inverse["loss"]:
            update_inverse_selection(best_converged_inverse, model=model, record=record)
        if writer is not None:
            writer.append(mesh, time=float(step))

    return callback


def solve_case(
    cfg: Config,
    *,
    source: pv.UnstructuredGrid,
    target: pv.UnstructuredGrid,
    case_name: str,
    initial_log_activation: float,
) -> dict[str, Any]:
    mesh = source.copy()
    model = build_model(
        mesh,
        muscle_log_relative_activation=initial_log_activation,
        cfg=cfg,
        muscle_requires_grad=True,
    )
    fallback_u_free = initialize_target_guess(model, mesh, target)

    forward = Forward(model)
    configure_forward_solver(forward)
    forward.last_successful_u_free = fallback_u_free
    inverse = build_inverse(
        cfg,
        mesh=mesh,
        target=target,
        forward=forward,
        fallback_u_free=fallback_u_free,
    )

    params: Scalar = jnp.asarray(initial_log_activation, dtype=model.u_full.dtype)
    best_inverse = new_inverse_selection()
    best_inverse["log_activation"] = float(initial_log_activation)
    best_converged_inverse = new_inverse_selection()
    best_converged_inverse["log_activation"] = float(initial_log_activation)
    metric_records: list[MetricRecord] = []
    writer_cm = (
        melon.io.SeriesWriter(output_series_path(case_name), clear=True)
        if cfg.write_series
        else nullcontext(None)
    )
    with writer_cm as writer:
        params = inverse.solve(
            params,
            make_inverse_step_callback(
                cfg=cfg,
                mesh=mesh,
                target=target,
                forward=forward,
                model=model,
                inverse=inverse,
                case_name=case_name,
                best_inverse=best_inverse,
                best_converged_inverse=best_converged_inverse,
                metric_records=metric_records,
                writer=writer,
            ),
        )

    selected_inverse = (
        best_converged_inverse
        if np.isfinite(best_converged_inverse["loss"])
        else best_inverse
    )
    if np.isfinite(selected_inverse["loss"]):
        params = jnp.asarray(
            selected_inverse["log_activation"], dtype=model.u_full.dtype
        )
        if selected_inverse["u_free"] is not None:
            model.u_free = jnp.asarray(
                selected_inverse["u_free"], dtype=model.u_free.dtype
            )

    final_materials = inverse.make_materials(params)
    forward.update_materials(final_materials)
    forward.state = model.init_state(model.u_full)
    final_forward_success = bool(selected_inverse["forward_success"])
    cherries.log_metrics(
        {
            "forward": {
                "reused_best_inverse_state": selected_inverse["u_free"] is not None,
                "final_grad_norm": selected_inverse["forward_grad_norm"],
                "success": final_forward_success,
            }
        }
    )
    final_log_activation = float(np.asarray(params))
    final_relative_activation = relative_activation_from_log_param(
        cfg, final_log_activation
    )
    fill_inverse_outputs(
        mesh,
        cfg=cfg,
        target=target,
        forward=forward,
        model=model,
        log_relative_activation=final_log_activation,
    )
    assert_finite_outputs(mesh)
    melon.save(cherries.output(f"{output_stem(case_name)}.vtu"), mesh)

    point_to_point = np.asarray(mesh.point_data["PointToPoint"])
    activation_diff = np.asarray(mesh.cell_data["ActivationDiff"])
    summary: dict[str, Any] = {
        "case": case_name,
        "backend": str(jnp.asarray(0.0).device),
        "forward_success": final_forward_success,
        "forward_result": str(selected_inverse["forward_result"]),
        "forward_grad_norm": float(selected_inverse["forward_grad_norm"]),
        "inverse_lr": float(cfg.inverse_lr),
        "best_inverse_step": int(best_inverse["step"]),
        "best_converged_inverse_step": int(best_converged_inverse["step"]),
        "selected_inverse_step": int(selected_inverse["step"]),
        "selected_inverse_kind": (
            "activation_tolerance"
            if np.isfinite(best_converged_inverse["loss"])
            else "minimum_loss"
        ),
        "metrics_json": metrics_json_path(case_name).as_posix(),
        "metrics_csv": metrics_csv_path(case_name).as_posix(),
        "metrics_plots": metrics_plot_paths(case_name),
        "series": output_series_path(case_name).as_posix(),
        "initial_log_relative_activation": float(initial_log_activation),
        "initial_relative_activation": relative_activation_from_log_param(
            cfg, initial_log_activation
        ),
        "target_relative_activation": float(cfg.target_activation),
        "target_log_relative_activation": math.log(cfg.target_activation),
        "solved_log_relative_activation": final_log_activation,
        "solved_relative_activation": final_relative_activation,
        "relative_activation_error": final_relative_activation - cfg.target_activation,
        "best_inverse_loss": float(best_inverse["loss"]),
        "best_converged_inverse_loss": float(best_converged_inverse["loss"]),
        "selected_inverse_loss": float(selected_inverse["loss"]),
        "smas_prestrain": float(cfg.smas_prestrain),
        "target_actual_activation_x": float(cfg.target_activation * cfg.smas_prestrain),
        "target_actual_activation_y": float(1.0 / cfg.smas_prestrain**2),
        "target_actual_activation_z": float(cfg.smas_prestrain),
        "solved_actual_activation_x": float(
            final_relative_activation * cfg.smas_prestrain
        ),
        "solved_actual_activation_y": float(1.0 / cfg.smas_prestrain**2),
        "solved_actual_activation_z": float(cfg.smas_prestrain),
        "point_to_point_l2": float(np.linalg.norm(point_to_point)),
        "point_to_point_inf": float(
            np.linalg.norm(point_to_point.reshape(-1), ord=np.inf)
        ),
        "activation_diff_l2": float(np.linalg.norm(activation_diff)),
        "activation_diff_inf": float(
            np.linalg.norm(activation_diff.reshape(-1), ord=np.inf)
        ),
    }
    save_summary(cherries.output(f"{output_stem(case_name)}.json"), summary)
    if (
        not final_forward_success
        or abs(final_relative_activation - cfg.target_activation)
        > cfg.activation_tolerance
    ):
        message = (
            f"{case_name} inverse did not converge: "
            f"relative activation={final_relative_activation:.6g}"
        )
        raise RuntimeError(message)
    return summary


def parse_cases(cases: str) -> list[str]:
    return [case.strip() for case in cases.split(",") if case.strip()]


def initial_log_activation_for_case(cfg: Config, case_name: str) -> float:
    if case_name == "easy":
        return math.log(cfg.easy_initial_activation)
    if case_name == "zero":
        return cfg.zero_initial_log_activation
    message = f"unknown inverse case {case_name!r}"
    raise ValueError(message)


def run_inverse(cfg: Config) -> None:
    source = melon.load_unstructured_grid(cfg.input)
    target = make_target_mesh(cfg, source)
    assert_finite_outputs(target)

    summaries = [
        solve_case(
            cfg,
            source=source,
            target=target,
            case_name=case_name,
            initial_log_activation=initial_log_activation_for_case(cfg, case_name),
        )
        for case_name in parse_cases(cfg.cases)
    ]
    save_summary(
        cherries.output(f"31-inverse{SUFFIX}-activation-stable-neo-hookean.json"),
        {"cases": summaries},
    )


def main(cfg: Config) -> None:
    wp.init()
    run_inverse(cfg)


if __name__ == "__main__":
    cherries.main(main)
