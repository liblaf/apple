import jax
import jax.numpy as jnp
import pytest
import scipy.optimize
from jaxtyping import Float

from liblaf import apple


def test_pncg(n: int = 7) -> None:
    x0: Float[jax.Array, " N"] = jnp.zeros((n,))
    result: scipy.optimize.OptimizeResult = apple.minimize(
        x0=x0,
        fun=apple.rosen,
        jac=jax.grad(apple.rosen),
        hess=jax.hessian(apple.rosen),
        algo=apple.opt.MinimizePNCG(),
    )
    x: Float[jax.Array, " N"] = jnp.asarray(result["x"])
    # TODO: decrease tolerance
    assert x == pytest.approx(jnp.ones((n,)), abs=2.0)
