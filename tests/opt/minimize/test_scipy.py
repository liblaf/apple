import jax
import jax.numpy as jnp
import pytest
import scipy.optimize
from jaxtyping import Float

from liblaf import apple


def test_trust_constr(n: int = 7) -> None:
    x0: Float[jax.Array, " N"] = jnp.zeros((n,))
    result: scipy.optimize.OptimizeResult = apple.minimize(
        x0=x0,
        fun=apple.rosen,
        algo=apple.opt.MinimizeScipy(
            "trust-constr", options={"disp": True, "verbose": 3}
        ),
    )
    x: Float[jax.Array, " N"] = jnp.asarray(result["x"])
    assert x == pytest.approx(jnp.ones((n,)), abs=2e-5)
