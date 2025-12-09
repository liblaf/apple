import logging
from collections.abc import Mapping
from pathlib import Path
from typing import override

import jax.numpy as jnp
import numpy as np
import pyvista as pv
from jaxtyping import Array, Float, Integer
from liblaf.peach import tree
from liblaf.peach.constraints import BoundConstraint
from liblaf.peach.optim import ScipyOptimizer

from liblaf import cherries, melon
from liblaf.apple import Forward, Inverse, Model, ModelBuilder
from liblaf.apple.constants import MUSCLE_FRACTION, POINT_ID
from liblaf.apple.warp import Phace

type EnergyParams = Mapping[str, Array]
type ModelParams = Mapping[str, EnergyParams]
type Scalar = Float[Array, ""]


logger: logging.Logger = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    input: Path = cherries.input("10-input.vtu")
    target: Path = cherries.input("10-target.vtu")

    output: Path = cherries.output("20-inverse-no-reg.vtu")


@tree.define
class InverseActivation(Inverse):
    @tree.define
    class Params(Inverse.Params):
        activation: Float[Array, "cells 6"]

    @tree.define
    class Aux(Inverse.Aux):
        point_to_point: Scalar
        sparse: Scalar

    face_idx: Integer[Array, " face"]
    muscle_cell_idx: Integer[Array, " muscle_cells"]
    muscle_volume: Float[Array, " muscle_cells"]
    n_cells: int
    target: Float[Array, "face dim"]

    @property
    def n_muscle_cells(self) -> int:
        return self.muscle_cell_idx.size

    @override
    def loss(self, u: Array, params: ModelParams) -> tuple[Scalar, Aux]:
        point_to_point: Scalar = jnp.sum(jnp.square(u[self.face_idx] - self.target))
        sparse: Scalar = 1e-3 * self.reg_sparse(params)
        return point_to_point + sparse, self.Aux(
            point_to_point=point_to_point, sparse=sparse
        )

    @override
    def make_params(self, params: Params) -> ModelParams:
        activation: Float[Array, "cells 6"] = jnp.zeros((self.n_cells, 6))
        activation = activation.at[self.muscle_cell_idx].set(params.activation)
        return {"elastic": {"activation": activation}}

    def reg_sparse(self, params: ModelParams) -> Scalar:
        activation: Float[Array, "cells 6"] = params["elastic"]["activation"]
        muscle_activation: Float[Array, "muscle_cells 6"] = activation[
            self.muscle_cell_idx
        ]
        return jnp.dot(
            self.muscle_volume, jnp.sum(jnp.square(muscle_activation), axis=-1)
        )


def prepare(mesh: pv.UnstructuredGrid) -> tuple[pv.UnstructuredGrid, InverseActivation]:
    mesh = mesh.compute_cell_sizes(length=False, area=False, volume=True)  # pyright: ignore[reportAssignmentType]
    builder = ModelBuilder()
    mesh = builder.assign_global_ids(mesh)
    builder.add_dirichlet(mesh)

    elastic: Phace = Phace.from_pyvista(
        mesh, id="elastic", requires_grad=["activation"]
    )
    builder.add_energy(elastic)
    model: Model = builder.finalize()

    forward: Forward = Forward(model=model)

    face_idx: Integer[Array, " face"] = jnp.flatnonzero(mesh.point_data["IsFace"])
    muscle_cell_idx: Integer[Array, " muscle_cells"] = jnp.flatnonzero(
        mesh.cell_data[MUSCLE_FRACTION] >= 1e-2
    )
    target: Float[Array, "face dim"] = jnp.asarray(
        mesh.point_data["Expression000"][face_idx]
    )
    muscle_volume: Float[Array, " muscle_cells"] = jnp.asarray(
        mesh.cell_data[MUSCLE_FRACTION][muscle_cell_idx]
    ) * jnp.asarray(mesh.cell_data["Volume"][muscle_cell_idx])

    inverse = InverseActivation(
        forward=forward,
        face_idx=face_idx,
        muscle_cell_idx=muscle_cell_idx,
        muscle_volume=muscle_volume,
        n_cells=mesh.n_cells,
        target=target,
        optimizer=ScipyOptimizer(method="L-BFGS-B", tol=1e-5, options={"maxls": 2}),
    )
    return mesh, inverse


def calc_inverse(
    mesh: pv.UnstructuredGrid,
    target: pv.UnstructuredGrid,
    inverse: InverseActivation,
    idx: str = "000",
) -> pv.UnstructuredGrid:
    inverse.target = jnp.asarray(target.point_data[f"Expression{idx}"])[
        inverse.face_idx
    ]
    params = InverseActivation.Params(activation=jnp.zeros((inverse.n_muscle_cells, 6)))
    bounds = BoundConstraint(
        InverseActivation.Params(
            activation=jnp.full((inverse.n_muscle_cells, 6), -20.0)
        ),
        InverseActivation.Params(
            activation=jnp.full((inverse.n_muscle_cells, 6), 20.0)
        ),
    )

    with melon.SeriesWriter(
        cherries.temp(f"20-inverse-no-reg-{idx}.vtu.series")
    ) as writer:

        def callback(state: ScipyOptimizer.State, stats: ScipyOptimizer.Stats) -> None:
            ic(state, stats)
            cherries.log_metrics({idx: {"loss": state.fun}}, step=stats.n_steps)
            params: ModelParams = inverse.make_params(state.params)
            point_id: Integer[Array, " points"] = jnp.asarray(mesh.point_data[POINT_ID])
            mesh.point_data[f"Solution{idx}"] = inverse.model.u_full[point_id]  # pyright: ignore[reportArgumentType]
            mesh.point_data[f"Residual{idx}"] = np.zeros((mesh.n_points, 3))
            mesh.point_data[f"Residual{idx}"][inverse.face_idx] = (  # pyright: ignore[reportArgumentType]
                inverse.model.u_full[inverse.face_idx] - inverse.target
            )
            mesh.cell_data[f"Activation{idx}"] = params["elastic"]["activation"]  # pyright: ignore[reportArgumentType]
            writer.append(mesh)

        solution: ScipyOptimizer.Solution = inverse.solve(
            params=params, constraints=[bounds], callback=callback
        )

    ic(solution)
    model_params: ModelParams = inverse.make_params(solution.params)
    point_id: Integer[Array, " points"] = jnp.asarray(mesh.point_data[POINT_ID])
    mesh.point_data[f"Solution{idx}"] = inverse.model.u_full[point_id]  # pyright: ignore[reportArgumentType]
    mesh.cell_data[f"Activation{idx}"] = model_params["elastic"]["activation"]  # pyright: ignore[reportArgumentType]
    return mesh


def main(cfg: Config) -> None:
    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.input)
    target: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.target)
    # target.point_data["Expression000"] *= 0.8
    # target.point_data["Expression001"] *= 0.8
    inverse: InverseActivation
    mesh, inverse = prepare(mesh)
    mesh = calc_inverse(mesh, target, inverse, idx="000")
    mesh = calc_inverse(mesh, target, inverse, idx="001")
    melon.save(cfg.output, mesh)


if __name__ == "__main__":
    cherries.main(main)
