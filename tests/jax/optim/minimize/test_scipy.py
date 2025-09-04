import hypothesis
import hypothesis.extra.numpy as hnp
import numpy as np
import scipy.optimize
from jaxtyping import ArrayLike, Float

from liblaf.apple.jax import optim


def rosen(x: ArrayLike) -> float:
    return scipy.optimize.rosen(np.asarray(x))


def rosen_der(x: ArrayLike) -> Float[np.ndarray, " N"]:
    return scipy.optimize.rosen_der(np.asarray(x))


def rosen_hess_prod(x: ArrayLike, p: ArrayLike) -> Float[np.ndarray, " N"]:
    return scipy.optimize.rosen_hess_prod(np.asarray(x), np.asarray(p))


@hypothesis.given(
    hnp.arrays(
        dtype=np.float64,
        shape=hnp.array_shapes(min_dims=1, max_dims=1, min_side=2),
        elements=hnp.from_dtype(np.dtype(np.float16), min_value=0.0, max_value=2.0),
    )
)
@hypothesis.settings(max_examples=10)
def test_minimize_scipy_trust_constr(x0: ArrayLike) -> None:
    optimizer = optim.MinimizerScipy(
        jit=False, method="trust-constr", options={"verbose": 3}
    )
    solution: optim.Solution = optimizer.minimize(
        x0=x0,
        fun=scipy.optimize.rosen,
        jac=scipy.optimize.rosen_der,
        hessp=scipy.optimize.rosen_hess_prod,
    )
    np.testing.assert_allclose(solution["x"], 1.0)


@hypothesis.given(
    hnp.arrays(
        dtype=np.float64,
        shape=hnp.array_shapes(min_dims=1, max_dims=1, min_side=2),
        elements=hnp.from_dtype(np.dtype(np.float16), min_value=0.0, max_value=2.0),
    )
)
@hypothesis.settings(max_examples=10)
def test_minimize_scipy_lbfgs(x0: ArrayLike) -> None:
    optimizer = optim.MinimizerScipy(jit=False, method="L-BFGS-B", options={})
    solution: optim.Solution = optimizer.minimize(
        x0=x0, fun=scipy.optimize.rosen, jac=scipy.optimize.rosen_der
    )
    np.testing.assert_allclose(solution["x"], 1.0, rtol=1e-4)
