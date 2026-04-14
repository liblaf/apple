import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import jarp
import jax.numpy as jnp
import numpy as np
import optax
import pyvista as pv
import scipy.sparse.linalg as spla
import warp as wp
from environs import env
from jaxtyping import Array, Bool, Float, Integer
from liblaf.peach.optim import Objective, Optax, Optimizer

from liblaf import cherries, melon
from liblaf.apple.consts import ACTIVATION, LAMBDA, MU, MUSCLE_FRACTION, SMAS_FRACTION
from liblaf.apple.inverse import Inverse, Loss, PointToPointLoss
from liblaf.apple.model import Forward, Full, Model, ModelBuilder, ModelMaterials
from liblaf.apple.optim import PNCG
from liblaf.apple.warp import WarpStableNeoHookean, WarpStableNeoHookeanMuscle

type EnergyMaterials = Mapping[str, Array]
type Scalar = Float[Array, ""]
type Vector = Float[Array, "points dim"]
type BoolNumeric = Bool[Array, ""]

FAILURE_CONTEXT: dict[str, Any] = {}


def _eigsh_values(
    operator: spla.LinearOperator,
    *,
    k: int,
    which: str,
    tol: float,
    maxiter: int,
) -> list[float]:
    try:
        values = spla.eigsh(
            operator,
            k=k,
            which=which,
            tol=tol,
            maxiter=maxiter,
            return_eigenvectors=False,
        )
    except spla.ArpackNoConvergence as exc:
        values = exc.eigenvalues
    values = np.asarray(values, dtype=np.float64)
    values.sort()
    return values.tolist()


def estimate_free_hessian_spectrum(
    model: Model,
    *,
    state: Any,
    k: int,
    tol: float,
    maxiter: int,
) -> dict[str, Any]:
    n_free = int(model.n_free)
    hess_diag_full = np.asarray(model.hess_diag(state), dtype=np.float64)
    hess_diag_free = np.asarray(model.dirichlet.get_free(jnp.asarray(hess_diag_full)))

    summary: dict[str, Any] = {
        "n_free": n_free,
        "hess_diag_min": float(np.min(hess_diag_free)),
        "hess_diag_max": float(np.max(hess_diag_free)),
        "hess_diag_mean": float(np.mean(hess_diag_free)),
        "hess_diag_nonpositive": int(np.count_nonzero(hess_diag_free <= 0.0)),
    }
    if n_free < 3:
        summary["smallest_algebraic"] = []
        summary["largest_algebraic"] = []
        return summary

    k_eff = max(1, min(k, n_free - 2))

    def matvec(v: np.ndarray) -> np.ndarray:
        v_free = jnp.asarray(v, dtype=model.u_full.dtype)
        v_full: Full = model.dirichlet.to_full(v_free, dirichlet=0.0)
        output_full = model.hess_prod(state, v_full)
        output_free = model.dirichlet.get_free(output_full)
        return np.asarray(output_free, dtype=np.float64)

    operator = spla.LinearOperator(
        shape=(n_free, n_free),
        matvec=matvec,
        rmatvec=matvec,
        dtype=np.float64,
    )
    summary["smallest_algebraic"] = _eigsh_values(
        operator, k=k_eff, which="SA", tol=tol, maxiter=maxiter
    )
    summary["largest_algebraic"] = _eigsh_values(
        operator, k=k_eff, which="LA", tol=tol, maxiter=maxiter
    )
    return summary


def dump_failure_artifacts(
    *,
    failure_id: str,
    failure_dir: Path,
    forward: Forward,
    model: Model,
    mesh: pv.UnstructuredGrid,
    target: pv.UnstructuredGrid,
    materials: ModelMaterials,
    solution: Optimizer.Solution,
    eigsh_k: int,
    eigsh_tol: float,
    eigsh_maxiter: int,
) -> None:
    spectrum = estimate_free_hessian_spectrum(
        model,
        state=forward.state,
        k=eigsh_k,
        tol=eigsh_tol,
        maxiter=eigsh_maxiter,
    )
    report = {
        "failure_id": failure_id,
        "forward_result": str(solution.result),
        "forward_success": bool(solution.success),
        "forward_steps": int(np.asarray(solution.state.n_steps)),
        "forward_best_decrease": float(np.asarray(solution.state.best_decrease)),
        "forward_relative_decrease": float(
            np.asarray(solution.stats.relative_decrease)
        ),
        "spectrum": spectrum,
    }
    report_path = failure_dir / f"failure-{failure_id}.json"
    report_path.write_text(json.dumps(report, indent=2))

    np.savez_compressed(
        failure_dir / f"failure-{failure_id}.npz",
        params=np.asarray(model.u_free),
        grad=np.asarray(solution.state.grad),
        hess_diag=np.asarray(solution.state.hess_diag),
        activation=np.asarray(materials["muscle"]["activation"]),
    )

    mesh_out = mesh.copy()
    mesh_out.point_data["InverseSolution"] = np.asarray(forward.u_full)
    mesh_out.point_data["PointToPoint"] = np.asarray(
        forward.u_full - target.point_data["Solution"]
    )
    mesh_out.cell_data["InverseActivation"] = cast(
        "wp.array", model.get_energy("muscle").materials.activation
    ).numpy()
    mesh_out.cell_data["ActivationDiff"] = (
        mesh_out.cell_data["InverseActivation"] - target.cell_data["Activation"]
    )
    mesh_out.point_data["ForwardGradNorm"] = np.full(
        (mesh_out.n_points,), float(np.linalg.norm(np.asarray(solution.state.grad)))
    )
    melon.save(failure_dir / f"failure-{failure_id}.vtu", mesh_out)


@jarp.define
class MyInverse(Inverse):
    full_activation: Float[Array, "cells 6"] = jarp.field()
    muscle_indices: Integer[Array, " muscle_cells"] = jarp.field()
    eigsh_k: int = jarp.static(default=4)
    eigsh_maxiter: int = jarp.static(default=80)
    eigsh_tol: float = jarp.static(default=1e-2)
    forward_failures: int = jarp.field(default=0, kw_only=True)

    def make_materials(self, params: Vector) -> ModelMaterials:
        activation: Float[Array, " cells 6"] = self.full_activation.at[
            self.muscle_indices
        ].set(params)
        return {"muscle": {"activation": activation}}

    def record_forward_failure(
        self,
        *,
        materials: ModelMaterials,
        solution: Optimizer.Solution,
    ) -> None:
        self.forward_failures += 1
        failure_id = f"{self.forward_failures:04d}"
        failure_dir: Path = FAILURE_CONTEXT["failure_dir"]
        mesh: pv.UnstructuredGrid = FAILURE_CONTEXT["mesh"]
        target: pv.UnstructuredGrid = FAILURE_CONTEXT["target"]
        dump_failure_artifacts(
            failure_id=failure_id,
            failure_dir=failure_dir,
            forward=self.forward,
            model=self.model,
            mesh=mesh,
            target=target,
            materials=materials,
            solution=solution,
            eigsh_k=self.eigsh_k,
            eigsh_tol=self.eigsh_tol,
            eigsh_maxiter=self.eigsh_maxiter,
        )

    def update(self, materials: ModelMaterials) -> None:
        self.model.update_materials(materials)
        if not self.last_forward_success:
            self.model.u_free = jnp.zeros_like(self.model.u_free)
        solution: PNCG.Solution = self.forward.step()
        cherries.log_metrics(
            {
                "forward": {
                    "decrease": solution.state.best_decrease,
                    "relative_decrease": solution.stats.relative_decrease,
                    "result": str(solution.result),
                }
            }
        )
        self.last_forward_success = jnp.asarray(solution.success)
        if not bool(np.asarray(solution.success)):
            self.record_forward_failure(materials=materials, solution=solution)


SUFFIX: str = "-smas46-muscle46"


class Config(cherries.BaseConfig):
    eigsh_k: int = env.int("EIGSH_K", 4)
    eigsh_maxiter: int = env.int("EIGSH_MAXITER", 80)
    eigsh_tol: float = env.float("EIGSH_TOL", 1e-2)
    inverse_lr: float = env.float("INVERSE_LR", 0.03)
    inverse_max_steps: int = env.int("INVERSE_MAX_STEPS", 20)
    inverse_patience: int = env.int("INVERSE_PATIENCE", 20)
    target: Path = cherries.input(
        f"20-forward{SUFFIX}-prestrain-bottom-dirichlet-arch-stable-neo-hookean.vtu"
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
    builder.add_energy(WarpStableNeoHookean.from_pyvista(mesh))

    mesh.cell_data["Fraction"] = aponeurosis_frac
    mesh.cell_data[MU] = np.full((mesh.n_cells,), 1.0e2)
    mesh.cell_data[LAMBDA] = np.full((mesh.n_cells,), 3.0e2)
    builder.add_energy(WarpStableNeoHookeanMuscle.from_pyvista(mesh))

    mesh.cell_data["Fraction"] = muscle_frac
    mesh.cell_data[MU] = np.full((mesh.n_cells,), 1.0e2)
    mesh.cell_data[LAMBDA] = np.full((mesh.n_cells,), 3.0e2)
    builder.add_energy(
        WarpStableNeoHookeanMuscle.from_pyvista(
            mesh, requires_grad=("activation",), name="muscle"
        )
    )

    return builder.finalize()


def make_unreachable_target(target: pv.UnstructuredGrid) -> pv.UnstructuredGrid:
    target.point_data["Solution"] = np.zeros_like(target.points)
    target.point_data["Solution"][:, 1] = target.points[:, 1]
    return target


def build_inverse(
    cfg: Config,
    *,
    mesh: pv.UnstructuredGrid,
    target: pv.UnstructuredGrid,
    forward: Forward,
    failure_dir: Path,
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
        eigsh_k=cfg.eigsh_k,
        eigsh_maxiter=cfg.eigsh_maxiter,
        eigsh_tol=cfg.eigsh_tol,
        optimizer=Optax(
            optax.adam(cfg.inverse_lr),
            max_steps=jnp.asarray(cfg.inverse_max_steps),
            patience=jnp.asarray(cfg.inverse_patience),
        ),
    )


def main(cfg: Config) -> None:
    wp.init()
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.target)
    target: pv.UnstructuredGrid = make_unreachable_target(mesh.copy())
    mesh.point_data["Solution"] = target.point_data["Solution"].copy()
    target.cell_data[ACTIVATION] = mesh.cell_data[ACTIVATION].copy()

    failure_dir = cherries.temp(
        "30-inverse-unreachable-smas46-muscle46-stable-neo-hookean-failures"
    )
    failure_dir.mkdir(parents=True, exist_ok=True)
    FAILURE_CONTEXT["failure_dir"] = failure_dir
    FAILURE_CONTEXT["mesh"] = mesh
    FAILURE_CONTEXT["target"] = target

    model: Model = build_phace_v3(mesh)
    forward: Forward = Forward(model)
    forward_optimizer = cast("PNCG", forward.optimizer)
    forward_optimizer.rtol = jnp.asarray(1e-3)
    forward_optimizer.rtol_primary = jnp.asarray(1e-5)

    inverse = build_inverse(
        cfg,
        mesh=mesh,
        target=target,
        forward=forward,
        failure_dir=failure_dir,
    )
    params: Vector = jnp.asarray(mesh.cell_data[ACTIVATION][inverse.muscle_indices])
    with melon.io.SeriesWriter(
        cherries.temp(f"30-inverse-unreachable{SUFFIX}-stable-neo-hookean.vtu.series")
    ) as writer:

        def callback(
            _objective: Objective[Any],
            _model_state: Any,
            _opt_state: Optimizer.State,
            _opt_stats: Optimizer.Stats,
        ) -> None:
            cherries.set_step((cherries.run.get_step() or 0) + 1)
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
            writer.append(mesh)

        params = inverse.solve(params, callback)

    materials: ModelMaterials = inverse.make_materials(params)
    forward.update_materials(materials)
    final_solution = forward.step()
    if not final_solution.success:
        dump_failure_artifacts(
            failure_id="final",
            failure_dir=failure_dir,
            forward=forward,
            model=model,
            mesh=mesh,
            target=target,
            materials=materials,
            solution=final_solution,
            eigsh_k=cfg.eigsh_k,
            eigsh_tol=cfg.eigsh_tol,
            eigsh_maxiter=cfg.eigsh_maxiter,
        )
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
    melon.save(
        cherries.output(f"30-inverse-unreachable{SUFFIX}-stable-neo-hookean.vtu"), mesh
    )


if __name__ == "__main__":
    cherries.main(main)
