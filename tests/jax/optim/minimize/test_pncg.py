import equinox as eqx
import hypothesis
import hypothesis.extra.numpy as hnp
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, Float

from liblaf.apple.jax import math, optim

type Vector = Float[Array, " N"]
type Scalar = Float[Array, ""]


def fun(x: Vector) -> Scalar:
    # return testing.rosen(x)
    return jnp.sum(jnp.square(x - 1.0))


def jac_and_hess_diag(
    x: Vector,
) -> tuple[Vector, Vector]:
    jac: Vector = jax.grad(fun)(x)
    hess_diag: Vector = math.hess_diag(fun, x)
    return (jac, hess_diag)


def hess_quad(x: Vector, p: Vector) -> Vector:
    return jnp.vdot(p, math.hess_prod(fun, x, p))


@hypothesis.given(
    hnp.arrays(
        dtype=np.float64,
        shape=hnp.array_shapes(min_dims=1, max_dims=1, min_side=2),
        elements=hnp.from_dtype(np.dtype(np.float16), min_value=0.0, max_value=2.0),
    )
)
@hypothesis.settings(max_examples=10)
def test_pncg(x0: ArrayLike) -> None:
    optimizer = optim.MinimizerPNCG(atol=1e-6, rtol=1e-6)
    solution: optim.Solution = optimizer.minimize(
        x0=x0,
        jac_and_hess_diag=eqx.filter_jit(jac_and_hess_diag),
        hess_quad=eqx.filter_jit(hess_quad),
    )
    np.testing.assert_allclose(solution["x"], 1.0, rtol=1e-3)
