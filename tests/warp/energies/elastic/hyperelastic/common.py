from collections.abc import Callable, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
import pyvista as pv
import warp as wp
from jaxtyping import Array, ArrayLike, Float, Key

import liblaf.apple.warp.utils as wpu
from liblaf.apple import Model
from liblaf.apple.jax import testing
from liblaf.apple.warp.energies.elastic.hyperelastic import Hyperelastic

type EnergyParams = Mapping[str, Array]
type Full = Float[Array, "points 3"]
type ModelParams = Mapping[str, EnergyParams]
type Scalar = Float[Array, ""]


def check_grad(seed: int, model: Model, mesh: pv.UnstructuredGrid) -> None:
    u: Full = rand_u(seed, mesh)
    testing.check_grad(model.fun, model.grad, u, rtol=1e-3)


def check_hess_diag(seed: int, model: Model, mesh: pv.UnstructuredGrid) -> None:
    def hess_diag_from_quad(quad: Callable, u: Full) -> Full:
        diag_np: np.ndarray = np.zeros_like(u)
        for ijk in np.ndindex(*u.shape):
            p: Full = jnp.zeros_like(u)
            p = p.at[ijk].set(1.0)
            diag_np[ijk] = quad(u, p)
        diag: Full = jnp.asarray(diag_np)
        return diag

    u: Full = rand_u(seed, mesh)
    actual: Full = model.hess_diag(u)
    expected: Full = hess_diag_from_quad(model.hess_quad, u)
    np.testing.assert_allclose(actual, expected, rtol=1e-3)


def check_hess_prod(seed: int, model: Model, mesh: pv.UnstructuredGrid) -> None:
    u: Full = rand_u(seed, mesh)
    testing.check_jvp(model.grad, model.hess_prod, u, rtol=1e-3)


def check_hess_quad(seed: int, model: Model, mesh: pv.UnstructuredGrid) -> None:
    u: Full = rand_u(seed, mesh)
    p: Full = rand_u(seed + 1, mesh)
    actual: Scalar = model.hess_quad(u, p)
    expected: Scalar = jnp.vdot(p, model.hess_prod(u, p))
    np.testing.assert_allclose(actual, expected, rtol=1e-3)


def check_mixed_derivative_prod(
    seed: int,
    model: Model,
    mesh: pv.UnstructuredGrid,
    *,
    param_name: str,
    param_shape: Sequence[int],
    minval: float = -1.0,
    maxval: float = 1.0,
) -> None:
    u: Full = rand_u(seed, mesh)
    p: Full = rand_u(seed + 1, mesh)

    def f(q: Array) -> Scalar:
        energy: Hyperelastic = model.warp.energies["elastic"]  # pyright: ignore[reportAssignmentType]
        param: wp.array = getattr(energy.params, param_name)
        wp.copy(param, wpu.to_warp(q, param.dtype))
        return jnp.vdot(p, model.grad(u))

    def f_jvp(q: Array, dq: Array) -> Scalar:
        energy: Hyperelastic = model.warp.energies["elastic"]  # pyright: ignore[reportAssignmentType]
        param: wp.array = getattr(energy.params, param_name)
        wp.copy(param, wpu.to_warp(q, param.dtype))
        grads: ModelParams = model.mixed_derivative_prod(u, p)
        return jnp.vdot(grads["elastic"][param_name], dq)

    key: Key = jax.random.key(seed)
    mu: Float[Array, " ..."] = jax.random.uniform(
        key, param_shape, minval=minval, maxval=maxval
    )
    testing.check_jvp(f, f_jvp, mu, rtol=1e-3)


def check_value_and_grad(seed: int, model: Model, mesh: pv.UnstructuredGrid) -> None:
    u: Full = rand_u(seed, mesh)
    value_expected: Scalar = model.fun(u)
    grad_expected: Full = model.grad(u)
    value_actual: Scalar
    grad_actual: Full
    value_actual, grad_actual = model.value_and_grad(u)
    np.testing.assert_allclose(value_actual, value_expected, rtol=1e-3)
    np.testing.assert_allclose(grad_actual, grad_expected, rtol=1e-3)


def check_grad_and_hess_diag(
    seed: int, model: Model, mesh: pv.UnstructuredGrid
) -> None:
    u: Full = rand_u(seed, mesh)
    grad_expected: Full = model.grad(u)
    hess_diag_expected: Full = model.hess_diag(u)
    grad_actual: Full
    hess_diag_actual: Full
    grad_actual, hess_diag_actual = model.grad_and_hess_diag(u)
    np.testing.assert_allclose(grad_actual, grad_expected, rtol=1e-3)
    np.testing.assert_allclose(hess_diag_actual, hess_diag_expected, rtol=1e-3)


def rand_u(seed: ArrayLike, mesh: pv.UnstructuredGrid) -> Full:
    key: Key = jax.random.key(seed)
    return jax.random.uniform(
        key, (mesh.n_points, 3), minval=-mesh.length, maxval=mesh.length
    )
