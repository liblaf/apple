import beartype
import jax
import jax.numpy as jnp
import jaxtyping
from jaxtyping import Float

from liblaf import apple


@jaxtyping.jaxtyped(typechecker=beartype.beartype)
@apple.jit()
def area(I_u: Float[jax.Array, "*C 2 2"]) -> Float[jax.Array, "*C"]:
    return 0.5 * jnp.sqrt(jnp.linalg.det(I_u))
