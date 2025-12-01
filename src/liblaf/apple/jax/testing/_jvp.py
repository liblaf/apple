from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Key

from ._close import assert_fraction_close


def check_grad(
    fun: Callable[[Array], Array],
    grad: Callable[[Array], Array],
    x: Array,
    *,
    atol: float = 0.0,
    fraction: float = 0.0,
    rtol: float = 1e-7,
) -> None:
    tangent: Array = _rand_like(x)
    actual: Array = jnp.vdot(grad(x), tangent)
    expected: Array = numeric_jvp(fun, x, tangent)
    assert_fraction_close(actual, expected, fraction=fraction, atol=atol, rtol=rtol)


def check_jvp(
    fun: Callable[[Array], Array],
    jvp: Callable[[Array, Array], Array],
    primals: Array,
    *,
    atol: float = 0.0,
    fraction: float = 0.0,
    rtol: float = 1e-7,
) -> None:
    tangent: Array = _rand_like(primals)
    actual: Array = jvp(primals, tangent)
    expected: Array = numeric_jvp(fun, primals, tangent)
    assert_fraction_close(actual, expected, fraction=fraction, atol=atol, rtol=rtol)


def numeric_jvp(
    fun: Callable[[Array], Array],
    primal: ArrayLike,
    tangent: ArrayLike | None = None,
    *,
    eps: float = 1e-4,
) -> Array:
    primal = jnp.asarray(primal)
    if tangent is None:
        key: Key = jax.random.key(0)
        tangent = jax.random.uniform(
            key, primal.shape, primal.dtype, minval=-1.0, maxval=1.0
        )
    else:
        tangent = jnp.asarray(tangent)
    f0: Array = fun(primal - 0.5 * eps * tangent)
    f1: Array = fun(primal + 0.5 * eps * tangent)
    output: Array = (f1 - f0) / eps
    return output


def _rand_like(arr: Array) -> Array:
    key: Key = jax.random.key(0)
    return jax.random.uniform(key, arr.shape, arr.dtype, minval=-1.0, maxval=1.0)
