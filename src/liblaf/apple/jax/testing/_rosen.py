import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from liblaf.apple.jax import math

type Scalar = Float[Array, ""]
type Vector = Float[Array, " I"]


def rosen(x: Vector) -> Scalar:
    return jnp.sum(
        100.0 * jnp.square(x[1:] - jnp.square(x[:-1])) + jnp.square(1.0 - x[:-1])
    )


def rosen_der(x: Vector) -> Vector:
    return jax.grad(rosen)(x)


def rosen_hess(x: Vector) -> Float[Array, "I I"]:
    return jax.hessian(rosen)(x)


def rosen_hess_prod(x: Vector, p: Vector) -> Vector:
    return math.hess_prod(rosen, x, p)
