import logging
from collections.abc import Mapping
from pathlib import Path
from typing import override

import jax.numpy as jnp
import numpy as np
import optax
import pyvista as pv
from environs import env
from jaxtyping import Array, Bool, Float, Integer
from liblaf.peach import tree
from liblaf.peach.linalg import CompositeSolver
from liblaf.peach.optim import PNCG, Optax, Optimizer

from liblaf import cherries, grapes, melon
from liblaf.apple import Forward, Inverse, Model, ModelBuilder
from liblaf.apple.constants import ACTIVATION, MUSCLE_FRACTION, POINT_ID
from liblaf.apple.warp import Phace

type Vector = Float[Array, " N"]
type EnergyParams = Mapping[str, Array]
type ModelParams = Mapping[str, EnergyParams]
type Full = Float[Array, "points dim"]
type Scalar = Float[Array, ""]


logger: logging.Logger = logging.getLogger(__name__)
SUFFIX: str = env.str("SUFFIX", default="-515k")


class Config(cherries.BaseConfig):
    input: Path = cherries.input(f"10-input{SUFFIX}.vtu")
    initial: Path = cherries.temp(
        "20-inverse-adam-123k.vtu.d/20-inverse-adam-123k_000149.vtu"
    )
    expression: str = "Expression000"


@tree.define
class PhaceInverse(Inverse):
    @override
    @tree.define
    class Aux(Inverse.Aux):
        point_to_point: Scalar
        smooth: Scalar
        sparse: Scalar

    @override
    @tree.define
    class Params(Inverse.Params):
        activation: Float[Array, " active"]

    @tree.define
    class Weights:
        point_to_point: Scalar = tree.array(default=jnp.asarray(1.0))
        smooth: Scalar = tree.array(default=jnp.asarray(1.0))
        sparse: Scalar = tree.array(default=jnp.asarray(1e-6))

    active_cell_id: Integer[Array, " active"]
    active_volume: Float[Array, " cells"]
    face_point_area: Float[Array, " face"]
    face_point_id: Integer[Array, " face"]
    muscle_id_to_cell_neighbors: dict[int, Integer[Array, "2 N"]]
    n_cells: int
    target: Float[Array, "face 3"]
    weights: Weights = tree.field(factory=Weights)

    @property
    def n_active_cells(self) -> int:
        return self.active_cell_id.shape[0]

    @override
    def adjoint(self, u: Full, dLdu: Full) -> Full:
        solution: CompositeSolver.Solution = self.adjoint_inner(u, dLdu)
        cherries.log_metrics(
            {
                "adjoint": {
                    "relative_residual": solution.stats.relative_residual.item(),
                    "success": int(solution.success),
                }
            }
        )
        if not solution.success:
            logger.warning("Adjoint fail: %r", solution)
        logger.info("Adjoint Statistics: %r", solution.stats)
        return self.model.to_full(solution.params, 0.0)

    @override
    def loss(self, u: Full, params: ModelParams) -> tuple[Scalar, Aux]:
        point_to_point: Scalar = self.weights.point_to_point * self.point_to_point(u)
        sparse: Scalar = self.weights.sparse * self.sparse(params)
        smooth: Scalar = self.weights.smooth * self.smooth(params)
        total: Scalar = point_to_point + smooth + sparse
        return total, self.Aux(
            point_to_point=point_to_point, smooth=smooth, sparse=sparse
        )

    @override
    def make_params(self, params: Params) -> ModelParams:
        activation: Float[Array, "cells 6"] = jnp.zeros((self.n_cells, 6))
        activation = activation.at[self.active_cell_id].set(params.activation)
        return {"elastic": {"activation": activation}}

    @override
    def value_and_grad(self, params: Params) -> tuple[Scalar, Params, Aux]:
        value: Scalar
        grad: PhaceInverse.Params
        aux: PhaceInverse.Aux
        value, grad, aux = super().value_and_grad(params)
        cherries.log_metrics(
            {
                "loss": {
                    "total": value.item(),
                    "point_to_point": aux.point_to_point.item(),
                    "smooth": aux.smooth.item(),
                    "sparse": aux.sparse.item(),
                }
            }
        )
        return value, grad, aux

    def point_to_point(self, u: Full) -> Scalar:
        diff: Float[Array, "face 3"] = u[self.face_point_id] - self.target
        diff *= 10.0  # centimeter to millimeter
        return jnp.average(
            jnp.sum(jnp.square(diff), axis=-1), weights=self.face_point_area
        )

    def smooth(self, params: ModelParams) -> Scalar:
        activation: Float[Array, "cells 6"] = params["elastic"]["activation"]
        losses: list[Scalar] = []
        normalizations: list[Scalar] = []
        for cell_neighbors in self.muscle_id_to_cell_neighbors.values():
            diff: Float[Array, "N 6"] = (
                activation[cell_neighbors[:, 0]] - activation[cell_neighbors[:, 1]]
            )
            loss: Scalar
            normalization: Scalar
            loss, normalization = jnp.average(
                jnp.sum(jnp.square(diff), axis=-1), returned=True
            )
            losses.append(loss)
            normalizations.append(normalization)
        return jnp.average(jnp.stack(losses), weights=jnp.stack(normalizations))

    def sparse(self, params: ModelParams) -> Scalar:
        activation: Float[Array, "cells 6"] = params["elastic"]["activation"]
        return jnp.average(
            jnp.sum(jnp.square(activation[self.active_cell_id]), axis=-1),
            weights=self.active_volume[self.active_cell_id],
        )

    @override
    def _forward(self, model_params: ModelParams) -> Full:
        solution: PNCG.Solution = self._forward_inner(model_params)
        logger.info("Forward Statistics: %r", solution.stats)
        cherries.log_metrics(
            {
                "forward": {
                    "relative_decrease": solution.stats.relative_decrease.item(),
                    "success": int(solution.success),
                }
            }
        )
        return self.model.u_full


def get_muscle_id_to_cell_neighbors(
    mesh: pv.UnstructuredGrid,
) -> dict[int, Integer[Array, "N 2"]]:
    unique_muscle_id: Integer[Array, " muscles"] = jnp.unique(
        mesh.cell_data["MuscleId"][mesh.cell_data[MUSCLE_FRACTION] > 1e-3]
    )
    muscle_id_to_cell_neighbors: dict[int, Integer[Array, "N 2"]] = {}
    for muscle_id in unique_muscle_id.tolist():
        if muscle_id < 0:
            continue
        muscle_id_to_cell_neighbors[muscle_id] = jnp.asarray(
            mesh.field_data[f"Muscle{muscle_id}CellNeighbors"]
        )
    return muscle_id_to_cell_neighbors


def prepare(mesh: pv.UnstructuredGrid, expression: str) -> PhaceInverse:
    builder = ModelBuilder()
    mesh = builder.assign_global_ids(mesh)
    builder.add_dirichlet(mesh)
    elastic: Phace = Phace.from_pyvista(
        mesh, requires_grad=["activation"], id="elastic"
    )
    builder.add_energy(elastic)
    model: Model = builder.finalize()
    forward = Forward(model=model)

    mesh = mesh.compute_cell_sizes(length=False, area=False, volume=True)  # pyright: ignore[reportAssignmentType]
    surface: pv.PolyData = mesh.extract_surface()  # pyright: ignore[reportAssignmentType]
    surface = melon.tri.compute_point_area(surface)
    mesh = melon.transfer_tri_point_to_tet(
        surface, mesh, data=["Area"], point_id=POINT_ID
    )
    muscle_fraction: Float[Array, " cells"] = jnp.asarray(
        mesh.cell_data[MUSCLE_FRACTION]
    )
    is_face: Bool[Array, " points"] = jnp.asarray(mesh.point_data["IsFace"])
    active_cell_id: Integer[Array, " active"] = jnp.flatnonzero(muscle_fraction > 1e-3)
    active_volume: Float[Array, " cells"] = jnp.asarray(
        muscle_fraction * mesh.cell_data["Volume"]
    )
    face_point_area: Float[Array, " face"] = jnp.asarray(
        mesh.point_data["Area"][is_face]
    )
    face_point_id: Integer[Array, " face"] = jnp.asarray(
        mesh.point_data[POINT_ID][is_face]
    )
    muscle_id_to_cell_neighbors: dict[int, Integer[Array, "N 2"]] = (
        get_muscle_id_to_cell_neighbors(mesh)
    )
    target: Float[Array, " face 3"] = jnp.asarray(mesh.point_data[expression][is_face])
    inverse = PhaceInverse(
        forward=forward,
        active_cell_id=active_cell_id,
        active_volume=active_volume,
        face_point_area=face_point_area,
        face_point_id=face_point_id,
        muscle_id_to_cell_neighbors=muscle_id_to_cell_neighbors,
        n_cells=mesh.n_cells,
        target=target,
        optimizer=Optax(optax.adam(0.1)),
    )
    return inverse


def main(cfg: Config) -> None:
    grapes.config.pretty.short_arrays_threshold.set(10)

    mesh: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.input)
    initial: pv.UnstructuredGrid = melon.load_unstructured_grid(cfg.initial)
    mesh = melon.transfer_tet_cell(initial, mesh, data=[ACTIVATION])

    inverse: PhaceInverse = prepare(mesh, cfg.expression)
    activation: Float[np.ndarray, "cells 6"] = np.zeros_like(mesh.cell_data[ACTIVATION])
    activation[inverse.active_cell_id] = mesh.cell_data[ACTIVATION][  # pyright: ignore[reportArgumentType]
        inverse.active_cell_id
    ]
    params: PhaceInverse.Params = PhaceInverse.Params(
        activation=mesh.cell_data[ACTIVATION][inverse.active_cell_id]
    )
    n_steps: int = 0

    with melon.SeriesWriter(
        cherries.temp(f"20-inverse-adam-fine{SUFFIX}.vtu.series")
    ) as writer:

        def callback(state: Optimizer.State, stats: Optimizer.Stats) -> None:
            nonlocal n_steps
            ic(stats)
            model_params: ModelParams = inverse.make_params(state.params)
            point_id: Integer[Array, " points"] = jnp.asarray(mesh.point_data[POINT_ID])
            mesh.point_data["Solution"] = inverse.forward.u_full[point_id]  # pyright: ignore[reportArgumentType]
            face_point_to_point: Float[Array, "face 3"] = (
                inverse.forward.u_full[inverse.face_point_id] - inverse.target
            )
            mesh.point_data["PointToPoint"] = np.zeros((mesh.n_points, 3))
            mesh.point_data["PointToPoint"][inverse.face_point_id] = face_point_to_point  # pyright: ignore[reportArgumentType]
            point_to_point_max: Scalar = jnp.max(
                jnp.linalg.norm(face_point_to_point, axis=-1)
            )
            point_to_point_max *= 10.0  # centimeter to millimeter
            cherries.log_metric("point_to_point_max", point_to_point_max.item())
            mesh.cell_data[ACTIVATION] = model_params["elastic"]["activation"]  # pyright: ignore[reportArgumentType]
            writer.append(mesh)
            cherries.set_step(n_steps)
            n_steps += 1

        # inverse.adjoint_solver = inverse.default_adjoint_solver(rtol=1e-3)
        # inverse.optimizer = Optax(
        #     optax.adam(0.1), max_steps=1000, rtol=1e-3, patience=20
        # )
        # inverse.weights.smooth = jnp.asarray(1.0)
        # solution: Optimizer.Solution = inverse.solve(params, callback=callback)
        # ic(solution)
        # params = solution.params

        inverse.adjoint_solver = inverse.default_adjoint_solver(rtol=1e-5)
        inverse.optimizer = Optax(
            optax.adam(0.02), max_steps=1000, rtol=0.0, patience=100
        )
        inverse.weights.smooth = jnp.asarray(1e-3)
        solution: Optimizer.Solution = inverse.solve(params, callback=callback)
        ic(solution)
        params = solution.params


if __name__ == "__main__":
    cherries.main(main)
