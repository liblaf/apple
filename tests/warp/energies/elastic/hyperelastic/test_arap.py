from collections.abc import Callable

import hypothesis
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import pyvista as pv
import warp as wp
from jaxtyping import Array, ArrayLike, Float, Key

import liblaf.apple.warp.utils as wpu
from liblaf.apple import Model, ModelBuilder
from liblaf.apple.constants import MU
from liblaf.apple.jax import testing
from liblaf.apple.warp.energies.elastic.hyperelastic import ARAP

type Scalar = Float[Array, ""]
type Full = Float[Array, "points 3"]


@pytest.fixture(scope="package")
def mesh() -> pv.UnstructuredGrid:
    mesh: pv.UnstructuredGrid = pv.examples.cells.Tetrahedron()  # pyright: ignore[reportAssignmentType]
    mesh.cell_data[MU] = np.full((mesh.n_cells,), 1.0)
    return mesh


@pytest.fixture(scope="package")
def model(mesh: pv.UnstructuredGrid) -> Model:
    builder = ModelBuilder()

    builder.assign_global_ids(mesh)

    elastic = ARAP.from_pyvista(
        mesh,
        clamp_hess_diag=False,
        clamp_hess_quad=False,
        clamp_lambda=False,
        id="elastic",
        requires_grad=[MU],
    )
    builder.add_energy(elastic)

    return builder.finalize()


@hypothesis.given(seed=testing.seed())
def test_arap_grad(seed: int, model: Model, mesh: pv.UnstructuredGrid) -> None:
    u: Full = _rand_u(seed, mesh)
    testing.check_grad(model.fun, model.grad, u, rtol=1e-4)


def hess_diag_from_quad(quad: Callable, u: Full) -> Full:
    diag_np: np.ndarray = np.zeros_like(u)
    for ijk in np.ndindex(*u.shape):
        p: Full = jnp.zeros_like(u)
        p = p.at[ijk].set(1.0)
        diag_np[ijk] = quad(u, p)
    diag: Full = jnp.asarray(diag_np)
    return diag


@hypothesis.given(seed=testing.seed())
def test_arap_hess_diag(seed: int, model: Model, mesh: pv.UnstructuredGrid) -> None:
    u: Full = _rand_u(seed, mesh)
    actual: Full = model.hess_diag(u)
    expected: Full = hess_diag_from_quad(model.hess_quad, u)
    np.testing.assert_allclose(actual, expected, rtol=1e-4)


@hypothesis.given(seed=testing.seed())
def test_arap_hess_prod(seed: int, model: Model, mesh: pv.UnstructuredGrid) -> None:
    u: Full = _rand_u(seed, mesh)
    testing.check_jvp(model.grad, model.hess_prod, u, rtol=1e-4)


@hypothesis.given(seed=testing.seed())
def test_arap_hess_quad(seed: int, model: Model, mesh: pv.UnstructuredGrid) -> None:
    u: Full = _rand_u(seed, mesh)
    p: Full = _rand_u(seed + 1, mesh)
    actual: Scalar = model.hess_quad(u, p)
    expected: Scalar = jnp.vdot(p, model.hess_prod(u, p))
    np.testing.assert_allclose(actual, expected, rtol=1e-4)


@hypothesis.given(seed=testing.seed())
def test_arap_mixed_derivative_prod(
    seed: int, model: Model, mesh: pv.UnstructuredGrid
) -> None:
    u: Full = _rand_u(seed, mesh)
    p: Full = _rand_u(seed + 1, mesh)

    def f(q: Array) -> Scalar:
        energy: ARAP = model.warp.energies["elastic"]  # pyright: ignore[reportAssignmentType]
        wp.copy(energy.params.mu, wpu.to_warp(q))
        return jnp.vdot(p, model.grad(u))

    def f_jvp(q: Array, dq: Array) -> Scalar:
        energy: ARAP = model.warp.energies["elastic"]  # pyright: ignore[reportAssignmentType]
        wp.copy(energy.params.mu, wpu.to_warp(q))
        grads: dict[str, dict[str, Array]] = model.mixed_derivative_prod(u, p)
        return jnp.vdot(grads["elastic"][MU], dq)

    key: Key = jax.random.key(seed)
    mu: Float[Array, " cells"] = jax.random.uniform(
        key, (mesh.n_cells,), minval=1e-6, maxval=1.0
    )
    testing.check_jvp(f, f_jvp, mu)


@hypothesis.given(seed=testing.seed())
def test_arap_value_and_grad(
    seed: int, model: Model, mesh: pv.UnstructuredGrid
) -> None:
    u: Full = _rand_u(seed, mesh)
    value_expected: Scalar = model.fun(u)
    grad_expected: Full = model.grad(u)
    value_actual: Scalar
    grad_actual: Full
    value_actual, grad_actual = model.value_and_grad(u)
    np.testing.assert_allclose(value_actual, value_expected, rtol=1e-4)
    np.testing.assert_allclose(grad_actual, grad_expected, rtol=1e-4)


@hypothesis.given(seed=testing.seed())
def test_arap_grad_and_hess_diag(
    seed: int, model: Model, mesh: pv.UnstructuredGrid
) -> None:
    u: Full = _rand_u(seed, mesh)
    grad_expected: Full = model.grad(u)
    hess_diag_expected: Full = model.hess_diag(u)
    grad_actual: Full
    hess_diag_actual: Full
    grad_actual, hess_diag_actual = model.grad_and_hess_diag(u)
    np.testing.assert_allclose(grad_actual, grad_expected, rtol=1e-4)
    np.testing.assert_allclose(hess_diag_actual, hess_diag_expected, rtol=1e-4)


def _rand_u(seed: ArrayLike, mesh: pv.UnstructuredGrid) -> Full:
    key: Key = jax.random.key(seed)
    return jax.random.uniform(
        key, (mesh.n_points, 3), minval=-mesh.length, maxval=mesh.length
    )
