from collections.abc import Callable, Mapping
from typing import Any

import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
import scipy
import scipy.differentiate
import scipy.optimize
from jaxtyping import PyTree


def check_jac_autodiff(
    fun: Callable,
    jac: Callable,
    x: PyTree,
    *,
    rtol: float = 1e-8,
    atol: float = 1e-8,
    **kwargs,
) -> None:
    actual = jac(x)
    expected = fun(x)
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, **kwargs)


def check_jac_finite_diff(
    fun: Callable,
    jac: Callable,
    x: PyTree,
    *,
    rtol: float = 1e-8,
    atol: float = 1e-8,
    **kwargs,
) -> None:
    x_flat: jax.Array
    x_unravel: Callable[[jax.Array], PyTree]
    x_flat, x_unravel = jax.flatten_util.ravel_pytree(x)

    def fun_flat_batch(x_flat_batch: jax.Array) -> jax.Array:
        return jnp.apply_along_axis(
            lambda x_flat: fun(x_unravel(x_flat)), axis=0, arr=x_flat_batch
        )

    actual: jax.Array = jac(x)
    result: Mapping[str, Any] = scipy.differentiate.jacobian(fun_flat_batch, x_flat)
    assert result["success"]
    expected: np.ndarray = result["df"]
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, **kwargs)
