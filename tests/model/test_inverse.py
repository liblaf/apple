from collections.abc import Mapping
from typing import override

import jax.numpy as jnp
import numpy as np
import pytest
import pyvista as pv
from jaxtyping import Array, Float, Integer
from liblaf.peach import tree
from liblaf.peach.optim import Optimizer

from liblaf.apple import Forward, Inverse, Model, ModelBuilder, Phace
from liblaf.apple.consts import (
    ACTIVATION,
    DIRICHLET_MASK,
    DIRICHLET_VALUE,
    LAMBDA,
    MU,
    MUSCLE_FRACTION,
    POINT_ID,
)

type EnergyParams = Mapping[str, Array]
type ModelParams = Mapping[str, EnergyParams]


@pytest.fixture(scope="package")
def mesh() -> pv.UnstructuredGrid:
    mesh: pv.UnstructuredGrid = pv.examples.download_tetrahedron()  # pyright: ignore[reportAssignmentType]
    mesh.point_data[DIRICHLET_MASK] = mesh.points[:, 1] < mesh.bounds.y_min + 1e-3 * (
        mesh.bounds.y_max - mesh.bounds.y_min
    )
    mesh.point_data[DIRICHLET_VALUE] = np.zeros((mesh.n_points, 3))
    mesh.cell_data[LAMBDA] = np.full((mesh.n_cells,), 1.0)
    mesh.cell_data[MU] = np.full((mesh.n_cells,), 1.0)
    mesh.cell_data[MUSCLE_FRACTION] = np.full((mesh.n_cells,), 0.5)

    rng: np.random.Generator = np.random.default_rng()
    mesh.cell_data[ACTIVATION] = rng.uniform(-1.0, 1.0, (mesh.n_cells, 6))

    return mesh


@pytest.fixture(scope="package")
def model(mesh: pv.UnstructuredGrid) -> Model:
    builder = ModelBuilder()
    mesh = builder.assign_global_ids(mesh)
    builder.add_dirichlet(mesh)
    elastic: Phace = Phace.from_pyvista(
        mesh, id="elastic", requires_grad=["activation"]
    )
    builder.add_energy(elastic)
    model: Model = builder.finalize()
    return model


def test_inverse(mesh: pv.UnstructuredGrid, model: Model) -> None:
    @tree.define
    class InverseActivation(Inverse):
        surface_idx: Integer[Array, " surface"]
        target: Float[Array, "surface dim"]

        @tree.define
        class Params(Inverse.Params):
            activation: Float[Array, "cells 6"]

        @tree.define
        class Aux(Inverse.Aux):
            pass

        @override
        def loss(
            self, u: Float[Array, "points dim"], params: ModelParams
        ) -> tuple[Float[Array, ""], Aux]:
            loss: Float[Array, ""] = 0.5 * jnp.sum(
                jnp.square(u[self.surface_idx] - self.target)
            )
            return loss, self.Aux()

        @override
        def make_params(self, params: Params) -> ModelParams:  # pyright: ignore[reportIncompatibleMethodOverride]
            return {"elastic": {"activation": params.activation}}

    forward = Forward(model)
    forward.step()

    surface: pv.PolyData = mesh.extract_surface()  # pyright: ignore[reportAssignmentType]
    surface_idx: Integer[Array, " surface"] = jnp.asarray(surface.point_data[POINT_ID])
    target: Float[Array, "surface dim"] = forward.u_full[surface_idx]

    inverse = InverseActivation(forward, surface_idx=surface_idx, target=target)
    params = InverseActivation.Params(activation=jnp.zeros((mesh.n_cells, 6)))
    solution: Optimizer.Solution = inverse.solve(params)
    assert solution.success

    forward.update_params(inverse.make_params(solution.params))
    forward.step()
    np.testing.assert_allclose(
        forward.u_full[surface_idx], target, atol=1e-3 * mesh.length
    )
