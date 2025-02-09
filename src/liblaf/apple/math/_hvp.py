from collections.abc import Callable
from typing import Literal

import autoregistry
import jax
import jax.numpy as jnp
from jaxtyping import Float, Scalar

type HvpMethod = Literal[
    "forward-over-reverse", "reverse-over-forward", "reverse-over-reverse", "naive"
]
hvp_method_registry = autoregistry.Registry(prefix="hvp_", hyphen=True)


def hvp(
    fun: Callable[[jax.Array], Scalar],
    u: jax.Array,
    v: jax.Array,
    method: HvpMethod = "forward-over-reverse",
) -> jax.Array:
    return hvp_method_registry[method](fun, u, v)


@hvp_method_registry
def hvp_forward_over_reverse(
    fun: Callable[[Float[jax.Array, " N"]], Scalar],
    u: Float[jax.Array, " N"],
    v: Float[jax.Array, " N"],
) -> Float[jax.Array, " N"]:
    return jax.jvp(jax.grad(fun), (u,), (v,))[1]


@hvp_method_registry
def hvp_reverse_over_forward(
    fun: Callable[[Float[jax.Array, " N"]], Scalar],
    u: Float[jax.Array, " N"],
    v: Float[jax.Array, " N"],
) -> Float[jax.Array, " N"]:
    def g(u: Float[jax.Array, " N"]) -> Float[jax.Array, " N"]:
        return jax.jvp(fun, (u,), (v,))[1]

    return jax.grad(g)(u)


@hvp_method_registry
def hvp_reverse_over_reverse(
    fun: Callable[[Float[jax.Array, " N"]], Scalar],
    u: Float[jax.Array, " N"],
    v: Float[jax.Array, " N"],
) -> Float[jax.Array, " N"]:
    return jax.grad(lambda x: jnp.vdot(jax.grad(fun)(x), v))(u)


@hvp_method_registry
def hvp_naive(
    fun: Callable[[Float[jax.Array, " N"]], Scalar],
    u: Float[jax.Array, " N"],
    v: Float[jax.Array, " N"],
) -> Float[jax.Array, " N"]:
    return jnp.tensordot(jax.hessian(fun)(u), v, axes=2)
