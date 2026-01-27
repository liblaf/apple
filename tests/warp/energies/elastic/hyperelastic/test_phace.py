import hypothesis
import numpy as np
import pytest
import pyvista as pv
from liblaf.apple.warp.energies.elastic.hyperelastic import Phace

from liblaf.apple import Model, ModelBuilder
from liblaf.apple.consts import ACTIVATION, LAMBDA, MU, MUSCLE_FRACTION
from liblaf.apple.jax import testing

from . import common


@pytest.fixture(scope="package")
def mesh() -> pv.UnstructuredGrid:
    mesh: pv.UnstructuredGrid = pv.examples.cells.Tetrahedron()  # pyright: ignore[reportAssignmentType]
    rng: np.random.Generator = np.random.default_rng()
    mesh.cell_data[ACTIVATION] = rng.uniform(-1.0, 1.0, (mesh.n_cells, 6))
    mesh.cell_data[LAMBDA] = rng.uniform(0.0, 1.0, (mesh.n_cells,))
    mesh.cell_data[MU] = rng.uniform(0.0, 1.0, (mesh.n_cells,))
    mesh.cell_data[MUSCLE_FRACTION] = rng.uniform(0.0, 1.0, (mesh.n_cells,))
    return mesh


@pytest.fixture(scope="package")
def model(mesh: pv.UnstructuredGrid) -> Model:
    builder = ModelBuilder()

    builder.assign_global_ids(mesh)

    elastic = Phace.from_pyvista(
        mesh,
        clamp_hess_diag=False,
        clamp_hess_quad=False,
        clamp_lambda=False,
        id="elastic",
        requires_grad=["activation", "lambda_", "mu", "muscle_fraction"],
    )
    # register kernels to avoid recompilation
    _ = elastic.fun_kernel
    _ = elastic.hess_diag_kernel
    _ = elastic.hess_prod_kernel
    _ = elastic.hess_quad_kernel
    _ = elastic.value_and_grad_kernel
    _ = elastic.grad_and_hess_diag_kernel
    builder.add_energy(elastic)

    return builder.finalize()


@hypothesis.given(seed=testing.seed())
def test_phace_grad(seed: int, model: Model, mesh: pv.UnstructuredGrid) -> None:
    common.check_grad_direction(seed, model, mesh)


@hypothesis.given(seed=testing.seed())
def test_phace_hess_diag(seed: int, model: Model, mesh: pv.UnstructuredGrid) -> None:
    common.check_hess_diag(seed, model, mesh)


@hypothesis.given(seed=testing.seed())
def test_phace_hess_prod(seed: int, model: Model, mesh: pv.UnstructuredGrid) -> None:
    common.check_hess_prod(seed, model, mesh)


@hypothesis.given(seed=testing.seed())
def test_phace_hess_quad(seed: int, model: Model, mesh: pv.UnstructuredGrid) -> None:
    common.check_hess_quad(seed, model, mesh)


@hypothesis.given(seed=testing.seed())
def test_phace_mixed_derivative_prod_activation(
    seed: int, model: Model, mesh: pv.UnstructuredGrid
) -> None:
    common.check_mixed_derivative_prod(
        seed,
        model,
        mesh,
        param_name="activation",
        param_shape=(mesh.n_cells, 6),
        minval=-1.0,
        maxval=1.0,
    )


@hypothesis.given(seed=testing.seed())
def test_phace_mixed_derivative_prod_lambda(
    seed: int, model: Model, mesh: pv.UnstructuredGrid
) -> None:
    common.check_mixed_derivative_prod(
        seed,
        model,
        mesh,
        param_name="lambda_",
        param_shape=(mesh.n_cells,),
        minval=0.0,
        maxval=1.0,
    )


@hypothesis.given(seed=testing.seed())
def test_phace_mixed_derivative_prod_mu(
    seed: int, model: Model, mesh: pv.UnstructuredGrid
) -> None:
    common.check_mixed_derivative_prod(
        seed,
        model,
        mesh,
        param_name="mu",
        param_shape=(mesh.n_cells,),
        minval=0.0,
        maxval=1.0,
    )


@hypothesis.given(seed=testing.seed())
def test_phace_mixed_derivative_prod_muscle_fraction(
    seed: int, model: Model, mesh: pv.UnstructuredGrid
) -> None:
    common.check_mixed_derivative_prod(
        seed,
        model,
        mesh,
        param_name="muscle_fraction",
        param_shape=(mesh.n_cells,),
        minval=0.0,
        maxval=1.0,
    )


@hypothesis.given(seed=testing.seed())
def test_phace_value_and_grad(
    seed: int, model: Model, mesh: pv.UnstructuredGrid
) -> None:
    common.check_value_and_grad(seed, model, mesh)


@hypothesis.given(seed=testing.seed())
def test_phace_grad_and_hess_diag(
    seed: int, model: Model, mesh: pv.UnstructuredGrid
) -> None:
    common.check_grad_and_hess_diag(seed, model, mesh)
