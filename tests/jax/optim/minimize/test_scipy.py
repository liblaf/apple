import hypothesis
import hypothesis.extra.numpy as hnp
import numpy as np
from jaxtyping import ArrayLike

from liblaf.apple.jax import optim, testing


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
        fun=testing.rosen,
        jac=testing.rosen_der,
        hessp=testing.rosen_hess_prod,
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
        x0=x0, fun=testing.rosen, jac=testing.rosen_der
    )
    np.testing.assert_allclose(solution["x"], 1.0, rtol=1e-4)
