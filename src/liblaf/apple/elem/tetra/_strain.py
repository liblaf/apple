import beartype
import einops
import jax
import jax.numpy as jnp
import jaxtyping
from jaxtyping import Float

from liblaf import apple


@jaxtyping.jaxtyped(typechecker=beartype.beartype)
@apple.jit()
def gradient(
    u: Float[jax.Array, "*c a=4 I=3"], dh_dX: Float[jax.Array, "*c a=4 J=3"]
) -> Float[jax.Array, "*c I=3 J=3"]:
    return einops.einsum(u, dh_dX, "... a I, ... a J -> ... I J")


@jaxtyping.jaxtyped(typechecker=beartype.beartype)
@apple.jit()
def deformation_gradient(
    u: Float[jax.Array, "*c a=4 I=3"], dh_dX: Float[jax.Array, "*c a=4 J=3"]
) -> Float[jax.Array, "*c I=3 J=3"]:
    grad_u: Float[jax.Array, "*c I=3 J=3"] = gradient(u, dh_dX)
    F: Float[jax.Array, "*c I=3 J=3"] = grad_u + jnp.identity(3)
    return F


@jaxtyping.jaxtyped(typechecker=beartype.beartype)
@apple.jit()
def deformation_gradient_jvp(
    dh_dX: Float[jax.Array, "*C 4 3"], p: Float[jax.Array, "*C 4 3"]
) -> Float[jax.Array, "*C 3 3"]:
    return einops.einsum(dh_dX, p, "... a I, ... a J -> ... J I")


@jaxtyping.jaxtyped(typechecker=beartype.beartype)
@apple.jit()
def deformation_gradient_vjp(
    dh_dX: Float[jax.Array, "*C 4 3"], p: Float[jax.Array, "*C 3 3"]
) -> Float[jax.Array, "*C 4 3"]:
    return einops.einsum(dh_dX, p, "... a I, ... J I -> ... a J")


@jaxtyping.jaxtyped(typechecker=beartype.beartype)
@apple.jit()
def deformation_gradient_gram(
    dh_dX: Float[jax.Array, "*C 4 3"],
) -> Float[jax.Array, "*C 4 3"]:
    result: Float[jax.Array, "*C 4"] = jnp.sum(dh_dX**2, axis=-1)
    result: Float[jax.Array, "*C 4 3"] = einops.repeat(result, "... a -> ... a 3")
    return result
