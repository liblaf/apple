import jax
import jax.flatten_util
import jax.numpy as jnp
from jaxtyping import Float, PyTree


def tree_vdot(a: PyTree, b: PyTree, /, **kwargs) -> Float[jax.Array, ""]:
    a_ravel: Float[jax.Array, " N"]
    a_ravel, _ = jax.flatten_util.ravel_pytree(a)
    b_ravel: Float[jax.Array, " N"]
    b_ravel, _ = jax.flatten_util.ravel_pytree(b)
    return jnp.vdot(a_ravel, b_ravel, **kwargs)
