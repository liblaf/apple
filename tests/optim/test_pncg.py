import jax
import jax.numpy as jnp
import numpy as np
import pytest

from liblaf.apple import optim, utils


@utils.jit
def fun(x: jax.Array) -> jax.Array:
    return jnp.sum(x**2)


@utils.jit
def jac(x: jax.Array) -> jax.Array:
    return 2 * x


@utils.jit
def hess_diag(x: jax.Array) -> jax.Array:
    return jnp.full_like(x, 2.0)


@utils.jit
def hess_quad(_x: jax.Array, p: jax.Array) -> jax.Array:
    return jnp.vdot(p, 2 * p)


@pytest.mark.parametrize("x0", [jnp.ones((10,))])
def test_optimizer_pncg(x0: jax.Array) -> None:
    optimizer = optim.PNCG(maxiter=10**4)
    result: optim.OptimizeResult = optimizer.minimize(
        fun=fun, x0=x0, jac=jac, hess_diag=hess_diag, hess_quad=hess_quad
    )
    np.testing.assert_allclose(result["x"], jnp.zeros_like(x0))
