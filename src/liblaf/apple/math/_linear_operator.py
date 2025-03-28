import jax
import jax.numpy as jnp
import pylops
from jaxtyping import Float


def as_linear_operator(
    A: Float[jax.Array, "M N"],
) -> Float[pylops.LinearOperator, "M N"]:
    return pylops.JaxOperator(pylops.MatrixMult(A))  # pyright: ignore[reportArgumentType]


def diagonal(A: Float[pylops.LinearOperator, "N N"]) -> Float[jax.Array, " N"]:
    vs: Float[jax.Array, "N N"] = jnp.identity(A.shape[0], dtype=A.dtype)  # pyright: ignore[reportArgumentType]

    def comp(v: Float[jax.Array, " N"]) -> Float[jax.Array, ""]:
        return jnp.vdot(v, A @ v)  # pyright: ignore[reportArgumentType]

    return jax.jit(jax.vmap(comp))(vs)
