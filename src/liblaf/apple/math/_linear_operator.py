import jax
import jax.numpy as jnp
import pylops
from jaxtyping import Float


def diagonal(A: Float[pylops.LinearOperator, "N N"]) -> Float[jax.Array, " N"]:
    vs: Float[jax.Array, "N N"] = jnp.identity(A.shape[0], dtype=A.dtype)

    def comp(v: Float[jax.Array, " N"]) -> Float[jax.Array, ""]:
        return jnp.vdot(v, A @ v)

    return jax.jit(jax.vmap(comp))(vs)
