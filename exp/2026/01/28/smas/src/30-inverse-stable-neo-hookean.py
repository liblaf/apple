from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import jarp
import jax.numpy as jnp
import numpy as np
import optax
import pyvista as pv
import warp as wp
from environs import env
from jaxtyping import Array, Bool, Float, Integer
from liblaf.peach.optim import Objective, Optax, Optimizer

from liblaf import cherries, melon
from liblaf.apple.consts import ACTIVATION, LAMBDA, MU, MUSCLE_FRACTION, SMAS_FRACTION
from liblaf.apple.inverse import Inverse, Loss, PointToPointLoss
from liblaf.apple.model import Forward, Model, ModelBuilder
from liblaf.apple.optim import PNCG
from liblaf.apple.warp import WarpStableNeoHookean, WarpStableNeoHookeanMuscle

type EnergyMaterials = Mapping[str, Array]
type ModelMaterials = Mapping[str, EnergyMaterials]
type Scalar = Float[Array, ""]
type Vector = Float[Array, "points dim"]
type BoolNumeric = Bool[Array, ""]


@jarp.define
class MyInverse(Inverse):
    muscle_indices: Integer[Array, " muscle_cells"] = jarp.field()
    full_activation: Float[Array, "cells 6"] = jarp.field()

    def make_materials(self, params: Vector) -> ModelMaterials:
        activation: Float[Array, " cells 6"] = self.full_activation.at[
            self.muscle_indices
        ].set(params)
        return {"muscle": {"activation": activation}}


SUFFIX: str = "-smas46-muscle46"


class Config(cherries.BaseConfig):
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


def build_inverse(cfg: Config, mesh: pv.UnstructuredGrid, forward: Forward) -> MyInverse:
    surface_indices: Integer[Array, " surface_points"] = mesh.surface_indices()
    muscle_indices: Integer[Array, " muscle_cells"] = jnp.flatnonzero(
        mesh.cell_data["MuscleFraction"] > 1e-3
    )
    losses: list[Loss] = [
        PointToPointLoss(
            indices=jnp.asarray(surface_indices),
            target=jnp.asarray(mesh.point_data["Solution"][surface_indices]),
        )
    ]
    full_activation: Float[Array, "cells 6"] = jnp.asarray(mesh.cell_data[ACTIVATION])
    return MyInverse(
        forward=forward,
        losses=losses,
        muscle_indices=muscle_indices,
        full_activation=full_activation,
        optimizer=Optax(
            optax.adam(cfg.inverse_lr),
            max_steps=jnp.asarray(cfg.inverse_max_steps),
            patience=jnp.asarray(cfg.inverse_patience),
        ),
    )


def main(cfg: Config) -> None:
    wp.init()
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.target)
    target: pv.UnstructuredGrid = mesh.copy()
    model: Model = build_phace_v3(mesh)
    forward: Forward = Forward(model)
    forward_optimizer = cast("PNCG", forward.optimizer)
    forward_optimizer.rtol = jnp.asarray(1e-5)
    forward_optimizer.rtol_primary = jnp.asarray(1e-6)
    inverse: MyInverse = build_inverse(cfg, mesh, forward)
    params: Vector = jnp.asarray(mesh.cell_data[ACTIVATION][inverse.muscle_indices])
    with melon.io.SeriesWriter(
        cherries.temp(f"30-inverse{SUFFIX}-stable-neo-hookean.vtu.series")
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
    forward.step()
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
    melon.save(cherries.output(f"30-inverse{SUFFIX}-stable-neo-hookean.vtu"), mesh)


if __name__ == "__main__":
    cherries.main(main)
