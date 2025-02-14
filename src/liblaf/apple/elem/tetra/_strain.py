import beartype
import einops
import jax
import jax.numpy as jnp
import jaxtyping
from jaxtyping import Float


@jaxtyping.jaxtyped(typechecker=beartype.beartype)
def gradient(
    u: Float[jax.Array, "*c a=4 I=3"], dh_dX: Float[jax.Array, "*c a=4 J=3"]
) -> Float[jax.Array, "*c I=3 J=3"]:
    return einops.einsum(u, dh_dX, "... a I, ... a J -> ... I J")


@jaxtyping.jaxtyped(typechecker=beartype.beartype)
def deformation_gradient(
    u: Float[jax.Array, "*c a=4 I=3"], dh_dX: Float[jax.Array, "*c a=4 J=3"]
) -> Float[jax.Array, "*c I=3 J=3"]:
    grad_u: Float[jax.Array, "*c I=3 J=3"] = gradient(u, dh_dX)
    F: Float[jax.Array, "*c I=3 J=3"] = grad_u + jnp.identity(3)
    return F
