import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.optimize

from liblaf.apple import optim


@pytest.mark.parametrize("x0", [jnp.asarray([1.3, 0.7, 0.8, 1.9, 1.2])])
def test_optimizer_scipy(x0: jax.Array) -> None:
    optimizer = optim.OptimizerScipy()
    result: optim.OptimizeResult = optimizer.minimize(
        fun=scipy.optimize.rosen,
        x0=x0,
        jac=scipy.optimize.rosen_der,
        hessp=scipy.optimize.rosen_hess_prod,
    )
    np.testing.assert_allclose(result["x"], jnp.ones_like(x0))
