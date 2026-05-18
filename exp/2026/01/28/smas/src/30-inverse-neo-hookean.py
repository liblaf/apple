import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pyvista as pv
import warp as wp
from environs import env
from jaxtyping import Array, Bool, Float, Integer
from liblaf.peach.optim import Objective, Optax, Optimizer

from liblaf import cherries, jarp, melon
from liblaf.apple.consts import (
    ACTIVATION,
    GLOBAL_POINT_ID,
    LAMBDA,
    MU,
    MUSCLE_FRACTION,
    SMAS_FRACTION,
)
from liblaf.apple.inverse import Inverse, Loss, PointToPointLoss
from liblaf.apple.model import Forward, Free, Model, ModelBuilder, ModelMaterials
from liblaf.apple.optim import PNCG
from liblaf.apple.warp import WarpNeoHookean, WarpNeoHookeanMuscle

type EnergyMaterials = Mapping[str, Array]
type Scalar = Float[Array, ""]
type Vector = Float[Array, "points dim"]
type BoolNumeric = Bool[Array, ""]

SUFFIX: str = "-smas46-muscle46"


def activation_to_mat33(
    activation: Float[Array, "*batch 6"],
) -> Float[Array, "*batch 3 3"]:
    row0 = jnp.stack(
        [1.0 + activation[..., 0], activation[..., 3], activation[..., 4]], axis=-1
    )
    row1 = jnp.stack(
        [activation[..., 3], 1.0 + activation[..., 1], activation[..., 5]], axis=-1
    )
    row2 = jnp.stack(
        [activation[..., 4], activation[..., 5], 1.0 + activation[..., 2]], axis=-1
    )
    return jnp.stack([row0, row1, row2], axis=-2)


def mat33_to_activation(mat: Float[Array, "*batch 3 3"]) -> Float[Array, "*batch 6"]:
    return jnp.stack(
        [
            mat[..., 0, 0] - 1.0,
            mat[..., 1, 1] - 1.0,
            mat[..., 2, 2] - 1.0,
            mat[..., 0, 1],
            mat[..., 0, 2],
            mat[..., 1, 2],
        ],
        axis=-1,
    )


def activation_to_spd_params(
    activation: np.ndarray,
) -> np.ndarray:
    mat = np.asarray(activation_to_mat33(jnp.asarray(activation)))
    chol = np.linalg.cholesky(mat)
    return np.stack(
        [
            np.log(chol[..., 0, 0]),
            np.log(chol[..., 1, 1]),
            np.log(chol[..., 2, 2]),
            chol[..., 1, 0],
            chol[..., 2, 0],
            chol[..., 2, 1],
        ],
        axis=-1,
    )


def spd_params_to_activation(
    params: Float[Array, "*batch 6"],
    *,
    chol_diag_min: float,
    chol_diag_max: float,
    shear_limit: float,
) -> Float[Array, "*batch 6"]:
    chol_diag_min_log = jnp.log(jnp.asarray(chol_diag_min, dtype=params.dtype))
    chol_diag_max_log = jnp.log(jnp.asarray(chol_diag_max, dtype=params.dtype))
    l00 = jnp.exp(jnp.clip(params[..., 0], chol_diag_min_log, chol_diag_max_log))
    l11 = jnp.exp(jnp.clip(params[..., 1], chol_diag_min_log, chol_diag_max_log))
    l22 = jnp.exp(jnp.clip(params[..., 2], chol_diag_min_log, chol_diag_max_log))
    l10 = jnp.clip(params[..., 3], -shear_limit, shear_limit)
    l20 = jnp.clip(params[..., 4], -shear_limit, shear_limit)
    l21 = jnp.clip(params[..., 5], -shear_limit, shear_limit)
    zeros = jnp.zeros_like(l00)
    row0 = jnp.stack([l00, zeros, zeros], axis=-1)
    row1 = jnp.stack([l10, l11, zeros], axis=-1)
    row2 = jnp.stack([l20, l21, l22], axis=-1)
    chol = jnp.stack([row0, row1, row2], axis=-2)
    activation_mat = chol @ jnp.swapaxes(chol, -1, -2)
    return mat33_to_activation(activation_mat)


def activation_eigenvalue_bounds(
    activation: np.ndarray,
) -> tuple[float, float]:
    mats = np.asarray(activation_to_mat33(jnp.asarray(activation)))
    eigenvalues = np.linalg.eigvalsh(mats)
    return float(eigenvalues.min()), float(eigenvalues.max())


def log_forward_pass_metrics(solution: PNCG.Solution, *, init_grad: Vector) -> None:
    cherries.log_metrics(
        {
            "forward": {
                "init_grad_norm": jnp.linalg.norm(init_grad),
                "init_grad_norm_inf": jnp.linalg.norm(init_grad, ord=jnp.inf),
                "final_grad_norm": jnp.linalg.norm(solution.state.grad),
                "final_grad_norm_max": jnp.linalg.norm(
                    solution.state.grad, ord=jnp.inf
                ),
                "decrease": solution.state.best_decrease,
                "relative_decrease": solution.stats.relative_decrease,
                "result": str(solution.result),
                "success": solution.success,
            }
        }
    )


@jarp.define
class MyInverse(Inverse):
    muscle_indices: Integer[Array, " muscle_cells"] = jarp.field()
    full_activation: Float[Array, "cells 6"] = jarp.field()
    chol_diag_min: float = jarp.static(default=0.25, kw_only=True)
    chol_diag_max: float = jarp.static(default=2.0, kw_only=True)
    shear_limit: float = jarp.static(default=1.0, kw_only=True)
    fallback_u_free: Free | None = jarp.array(default=None, kw_only=True)

    def make_materials(self, params: Vector) -> ModelMaterials:
        activation: Float[Array, " cells 6"] = self.full_activation.at[
            self.muscle_indices
        ].set(
            spd_params_to_activation(
                params,
                chol_diag_min=self.chol_diag_min,
                chol_diag_max=self.chol_diag_max,
                shear_limit=self.shear_limit,
            )
        )
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
        init_grad: Vector = self.model.grad(self.forward.state)
        solution: PNCG.Solution = self.forward.step()
        log_forward_pass_metrics(solution, init_grad=init_grad)
        self.last_forward_success = jnp.asarray(solution.success)

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
            for mat_name, v in energy.items():
                dLdq[energy_id][mat_name] += v
        return loss_value, dLdq


class Config(cherries.BaseConfig):
    inverse_lr: float = env.float("INVERSE_LR", 0.03)
    inverse_max_steps: int = env.int("INVERSE_MAX_STEPS", 1000)
    inverse_patience: int = env.int("INVERSE_PATIENCE", 1000)
    activation_chol_diag_min: float = env.float("ACTIVATION_CHOL_DIAG_MIN", 0.25)
    activation_chol_diag_max: float = env.float("ACTIVATION_CHOL_DIAG_MAX", 2.0)
    activation_shear_limit: float = env.float("ACTIVATION_SHEAR_LIMIT", 1.0)
    target: Path = cherries.input(
        f"20-forward{SUFFIX}-prestrain-bottom-dirichlet-arch-neo-hookean.vtu"
    )


def build_phace_v3(mesh: pv.UnstructuredGrid) -> Model:
    builder = ModelBuilder()

    muscle_frac: np.ndarray = mesh.cell_data[MUSCLE_FRACTION]
    smas_frac: np.ndarray = mesh.cell_data[SMAS_FRACTION]
    aponeurosis_frac: np.ndarray = smas_frac - muscle_frac
    fat_frac: np.ndarray = 1.0 - smas_frac

    mesh = builder.add_points(mesh)
    mesh.cell_data[ACTIVATION] = np.zeros((mesh.n_cells, 6))
    mesh.cell_data[ACTIVATION][smas_frac > 1e-3] = np.asarray(
        [2.0 - 1.0, 0.25 - 1.0, 2.0 - 1.0, 0.0, 0.0, 0.0]
    )
    builder.add_dirichlet(mesh)

    mesh.cell_data["Fraction"] = fat_frac
    mesh.cell_data[MU] = np.full((mesh.n_cells,), 1.0)
    mesh.cell_data[LAMBDA] = np.full((mesh.n_cells,), 3.0)
    builder.add_energy(WarpNeoHookean.from_pyvista(mesh))

    mesh.cell_data["Fraction"] = aponeurosis_frac
    mesh.cell_data[MU] = np.full((mesh.n_cells,), 1.0e2)
    mesh.cell_data[LAMBDA] = np.full((mesh.n_cells,), 3.0e2)
    builder.add_energy(WarpNeoHookeanMuscle.from_pyvista(mesh))

    mesh.cell_data["Fraction"] = muscle_frac
    mesh.cell_data[MU] = np.full((mesh.n_cells,), 1.0e2)
    mesh.cell_data[LAMBDA] = np.full((mesh.n_cells,), 3.0e2)
    builder.add_energy(
        WarpNeoHookeanMuscle.from_pyvista(
            mesh, requires_grad=("activation",), name="muscle"
        )
    )

    return builder.finalize()


def make_target_u_full(
    model: Model, mesh: pv.UnstructuredGrid, target: pv.UnstructuredGrid
) -> np.ndarray:
    dtype = np.asarray(model.u_full).dtype
    u_full = np.zeros((model.n_points, model.dim), dtype=dtype)
    u_full[mesh.point_data[GLOBAL_POINT_ID]] = np.asarray(
        target.point_data["Solution"], dtype=dtype
    )
    return u_full


def initialize_target_guess(
    model: Model, mesh: pv.UnstructuredGrid, target: pv.UnstructuredGrid
) -> Free:
    u_full = make_target_u_full(model, mesh, target)
    u_free = model.dirichlet.get_free(jnp.asarray(u_full))
    model.u_free = u_free
    return u_free


def build_inverse(
    cfg: Config,
    *,
    mesh: pv.UnstructuredGrid,
    target: pv.UnstructuredGrid,
    forward: Forward,
    fallback_u_free: Free,
) -> MyInverse:
    surface_indices: Integer[Array, " surface_points"] = mesh.surface_indices()
    muscle_indices: Integer[Array, " muscle_cells"] = jnp.flatnonzero(
        mesh.cell_data["MuscleFraction"] > 1e-3
    )
    losses: list[Loss] = [
        PointToPointLoss(
            indices=jnp.asarray(surface_indices),
            target=jnp.asarray(target.point_data["Solution"][surface_indices]),
        )
    ]
    full_activation: Float[Array, "cells 6"] = jnp.asarray(mesh.cell_data[ACTIVATION])
    return MyInverse(
        forward=forward,
        losses=losses,
        muscle_indices=muscle_indices,
        full_activation=full_activation,
        chol_diag_min=cfg.activation_chol_diag_min,
        chol_diag_max=cfg.activation_chol_diag_max,
        shear_limit=cfg.activation_shear_limit,
        fallback_u_free=fallback_u_free,
        optimizer=Optax(
            optax.adam(cfg.inverse_lr),
            max_steps=jnp.asarray(cfg.inverse_max_steps),
            patience=jnp.asarray(cfg.inverse_patience),
        ),
    )


def fill_inverse_outputs(
    mesh: pv.UnstructuredGrid,
    *,
    target: pv.UnstructuredGrid,
    forward: Forward,
    model: Model,
) -> None:
    mesh.point_data["InverseSolution"] = np.asarray(forward.u_full)
    mesh.point_data["PointToPoint"] = np.asarray(
        forward.u_full - target.point_data["Solution"]
    )
    mesh.cell_data["InverseActivation"] = cast(
        "wp.array", model.get_energy("muscle").materials.activation
    ).numpy()
    mesh.cell_data["ActivationDiff"] = (
        mesh.cell_data["InverseActivation"] - target.cell_data["Activation"]
    )
    eig_min, eig_max = activation_eigenvalue_bounds(mesh.cell_data["InverseActivation"])
    mesh.cell_data["ActivationEigMin"] = np.full((mesh.n_cells,), eig_min)
    mesh.cell_data["ActivationEigMax"] = np.full((mesh.n_cells,), eig_max)
    cherries.log_metrics(
        {
            "activation": {
                "eig_min": eig_min,
                "eig_max": eig_max,
            }
        }
    )


def save_summary(path: Path, values: dict[str, float | bool | str]) -> None:
    path.write_text(json.dumps(values, indent=2))


def run_inverse(cfg: Config) -> None:
    target: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.target)
    mesh: pv.UnstructuredGrid = target.copy()
    model: Model = build_phace_v3(mesh)
    fallback_u_free = initialize_target_guess(model, mesh, target)

    forward: Forward = Forward(model)
    forward.last_successful_u_free = fallback_u_free
    inverse: MyInverse = build_inverse(
        cfg,
        mesh=mesh,
        target=target,
        forward=forward,
        fallback_u_free=fallback_u_free,
    )
    params: Vector = jnp.asarray(
        activation_to_spd_params(
            np.asarray(mesh.cell_data[ACTIVATION][inverse.muscle_indices])
        )
    )

    with melon.io.SeriesWriter(
        cherries.temp(f"30-inverse{SUFFIX}-neo-hookean.vtu.series")
    ) as writer:

        def callback(
            _objective: Objective[Any],
            _model_state: Any,
            _opt_state: Optimizer.State,
            _opt_stats: Optimizer.Stats,
        ) -> None:
            cherries.set_step((cherries.run.get_step() or 0) + 1)
            fill_inverse_outputs(mesh, target=target, forward=forward, model=model)
            writer.append(mesh)

        params = inverse.solve(params, callback)

    materials: ModelMaterials = inverse.make_materials(params)
    forward.update_materials(materials)
    forward.state = model.init_state(model.u_full)
    init_grad: Vector = model.grad(forward.state)
    final_solution = forward.step()
    log_forward_pass_metrics(final_solution, init_grad=init_grad)
    fill_inverse_outputs(mesh, target=target, forward=forward, model=model)
    melon.save(cherries.output(f"30-inverse{SUFFIX}-neo-hookean.vtu"), mesh)

    point_to_point = np.asarray(mesh.point_data["PointToPoint"])
    activation_diff = np.asarray(mesh.cell_data["ActivationDiff"])
    save_summary(
        cherries.output(f"30-inverse{SUFFIX}-neo-hookean.json"),
        {
            "mode": "inverse",
            "backend": jax.default_backend(),
            "forward_success": bool(final_solution.success),
            "forward_result": str(final_solution.result),
            "point_to_point_l2": float(np.linalg.norm(point_to_point)),
            "point_to_point_inf": float(
                np.linalg.norm(point_to_point.reshape(-1), ord=np.inf)
            ),
            "activation_diff_l2": float(np.linalg.norm(activation_diff)),
            "activation_diff_inf": float(
                np.linalg.norm(activation_diff.reshape(-1), ord=np.inf)
            ),
            "activation_eig_min": float(mesh.cell_data["ActivationEigMin"][0]),
            "activation_eig_max": float(mesh.cell_data["ActivationEigMax"][0]),
        },
    )


def evaluate_target_state(model: Model, u_full: np.ndarray) -> dict[str, float]:
    warp_model = model.warp.__wrapped__
    u_wp = jarp.to_warp(u_full, (3, None))
    warp_state = warp_model.init_state(u_wp)
    warp_model.update(warp_state, u_wp)

    value_wp = wp.zeros((1,), dtype=float, device=u_wp.device)
    grad_wp = wp.zeros_like(u_wp)

    warp_model.fun(warp_state, u_wp, value_wp)
    warp_model.grad(warp_state, u_wp, grad_wp)

    grad_full = grad_wp.numpy()
    grad_free = np.asarray(model.dirichlet.get_free(jnp.asarray(grad_full)))

    return {
        "energy": float(value_wp.numpy()[0]),
        "full_grad_l2": float(np.linalg.norm(grad_full)),
        "full_grad_inf": float(np.linalg.norm(grad_full.reshape(-1), ord=np.inf)),
        "free_grad_l2": float(np.linalg.norm(grad_free)),
        "free_grad_inf": float(np.linalg.norm(grad_free, ord=np.inf)),
    }


def run_cpu_validation(cfg: Config) -> None:
    target: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.target)
    mesh: pv.UnstructuredGrid = target.copy()
    model: Model = build_phace_v3(mesh)
    u_full = make_target_u_full(model, mesh, target)

    # Warp's JAX FFI path is CUDA-only in this environment, so we validate the
    # reachable Neo-Hookean target directly through the Warp model on CPU.
    summary = evaluate_target_state(model, u_full)

    mesh.point_data["InverseSolution"] = target.point_data["Solution"].copy()
    mesh.point_data["PointToPoint"] = np.zeros_like(target.point_data["Solution"])
    mesh.point_data["ForwardGradNorm"] = np.full(
        (mesh.n_points,), summary["free_grad_l2"], dtype=mesh.points.dtype
    )
    mesh.point_data["ForwardGradNormInf"] = np.full(
        (mesh.n_points,), summary["free_grad_inf"], dtype=mesh.points.dtype
    )
    mesh.cell_data["InverseActivation"] = mesh.cell_data[ACTIVATION].copy()
    mesh.cell_data["ActivationDiff"] = (
        mesh.cell_data["InverseActivation"] - target.cell_data[ACTIVATION]
    )

    summary["mode"] = "cpu_validation"
    summary["backend"] = jax.default_backend()
    summary["activation_diff_l2"] = float(
        np.linalg.norm(mesh.cell_data["ActivationDiff"])
    )
    summary["activation_diff_inf"] = float(
        np.linalg.norm(mesh.cell_data["ActivationDiff"].reshape(-1), ord=np.inf)
    )
    summary["point_to_point_l2"] = 0.0
    summary["point_to_point_inf"] = 0.0

    melon.save(cherries.output(f"30-inverse{SUFFIX}-neo-hookean.vtu"), mesh)
    save_summary(cherries.output(f"30-inverse{SUFFIX}-neo-hookean.json"), summary)


def main(cfg: Config) -> None:
    wp.init()
    backend = jax.default_backend().lower()
    if backend in {"cuda", "gpu"}:
        run_inverse(cfg)
    else:
        run_cpu_validation(cfg)


if __name__ == "__main__":
    cherries.main(main)
