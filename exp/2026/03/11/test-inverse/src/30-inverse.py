from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import jarp
import jax.numpy as jnp
import numpy as np
import optax
import pyvista as pv
import warp as wp
from jaxtyping import Array, Bool, Float, Integer
from liblaf.peach.optim import Objective, Optax, Optimizer

from liblaf import cherries, melon
from liblaf.apple.consts import ACTIVATION, LAMBDA, MU
from liblaf.apple.inverse import Inverse, Loss, PointToPointLoss, UniformActivationLoss
from liblaf.apple.model import Forward, Model, ModelBuilder
from liblaf.apple.warp import WarpArapMuscle, WarpVolumePreservationDeterminant

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


class Config(cherries.BaseConfig):
    target: Path = cherries.input("20-forward.vtu")


def build_model(mesh: pv.UnstructuredGrid) -> Model:
    builder = ModelBuilder()

    mesh: pv.UnstructuredGrid = builder.add_points(mesh)
    mesh.cell_data[ACTIVATION] = np.zeros((mesh.n_cells, 6))
    # mesh.cell_data[ACTIVATION] = np.tile(
    #     np.asarray([2.0 - 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), (mesh.n_cells, 1)
    # )
    builder.add_dirichlet(mesh)

    mesh.cell_data[MU] = np.full((mesh.n_cells,), 1.0)
    energy_muscle: WarpArapMuscle = WarpArapMuscle.from_pyvista(
        mesh, requires_grad=("activation",), name="muscle"
    )
    builder.add_energy(energy_muscle)

    mesh.cell_data[LAMBDA] = np.full((mesh.n_cells,), 3.0)
    energy_vol: WarpVolumePreservationDeterminant = (
        WarpVolumePreservationDeterminant.from_pyvista(mesh)
    )
    builder.add_energy(energy_vol)

    model: Model = builder.finalize()
    return model


def build_inverse(mesh: pv.UnstructuredGrid, forward: Forward) -> MyInverse:
    surface_indices: Integer[Array, " surface_points"] = mesh.surface_indices()
    muscle_indices: Integer[Array, " muscle_cells"] = jnp.arange(mesh.n_cells)
    # gen unreachable target
    # mesh.point_data["Solution"] = np.zeros_like(mesh.points)
    # mesh.point_data["Solution"][:, 1] = mesh.points[:, 1]
    losses: list[Loss] = [
        PointToPointLoss(
            indices=jnp.asarray(surface_indices),
            target=jnp.asarray(mesh.point_data["Solution"][surface_indices]),
        ),
        UniformActivationLoss(muscle_indices=muscle_indices),
    ]
    full_activation: Float[Array, "cells 6"] = jnp.asarray(mesh.cell_data[ACTIVATION])
    return MyInverse(
        forward=forward,
        losses=losses,
        muscle_indices=muscle_indices,
        full_activation=full_activation,
        optimizer=Optax(optax.sgd(0.3), patience=jnp.asarray(1000)),
    )


def main(cfg: Config) -> None:
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.target)
    model: Model = build_model(mesh)
    forward: Forward = Forward(model)
    inverse: MyInverse = build_inverse(mesh, forward)
    params: Vector = jnp.asarray(mesh.cell_data[ACTIVATION][inverse.muscle_indices])
    with melon.io.SeriesWriter(cherries.temp("30-inverse.vtu.series")) as writer:

        def callback(
            _objective: Objective[Any],
            _model_state: Any,
            _opt_state: Optimizer.State,
            _opt_stats: Optimizer.Stats,
        ) -> None:
            cherries.set_step((cherries.run.get_step() or 0) + 1)
            mesh.point_data["InverseSolution"] = np.asarray(forward.u_full)
            mesh.point_data["PointToPoint"] = np.asarray(
                forward.u_full - mesh.point_data["Solution"]
            )
            mesh.cell_data["InverseActivation"] = cast(
                "wp.array", model.get_energy("muscle").materials.activation
            ).numpy()
            mesh.cell_data["ActivationDiff"] = (
                mesh.cell_data["InverseActivation"] - mesh.cell_data["Activation"]
            )
            writer.append(mesh)

        params = inverse.solve(params, callback)
    materials: ModelMaterials = inverse.make_materials(params)
    forward.update_materials(materials)
    forward.step()
    mesh.point_data["InverseSolution"] = np.asarray(forward.u_full)
    mesh.point_data["PointToPoint"] = np.asarray(
        forward.u_full - mesh.point_data["Solution"]
    )
    mesh.cell_data["InverseActivation"] = cast(
        "wp.array", model.get_energy("muscle").materials.activation
    ).numpy()
    mesh.cell_data["ActivationDiff"] = (
        mesh.cell_data["InverseActivation"] - mesh.cell_data["Activation"]
    )
    melon.save(cherries.output("30-inverse.vtu"), mesh)


if __name__ == "__main__":
    cherries.main(main)
