import beartype
import einops
import jax
import jax.numpy as jnp
import jaxtyping
from jaxtyping import Float

from liblaf.apple import utils


@jaxtyping.jaxtyped(typechecker=beartype.beartype)
@utils.jit
def h() -> Float[jax.Array, "a=4"]:
    return jnp.asarray([0.25, 0.25, 0.25, 0.25])


@jaxtyping.jaxtyped(typechecker=beartype.beartype)
@utils.jit
def dh_dr() -> Float[jax.Array, "a=4 J=3"]:
    return jnp.asarray(
        [
            [-1, -1, -1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ],
        dtype=float,
    )


@jaxtyping.jaxtyped(typechecker=beartype.beartype)
@utils.jit
def dX_dr(points: Float[jax.Array, "*c a=4 I=3"]) -> Float[jax.Array, "*c I=3 J=3"]:
    return einops.einsum(points, dh_dr(), "... a I, a J -> ... I J")


@jaxtyping.jaxtyped(typechecker=beartype.beartype)
@utils.jit
def dr_dX(points: Float[jax.Array, "*c a=4 I=3"]) -> Float[jax.Array, "*c I=3 J=3"]:
    return jnp.linalg.inv(dX_dr(points))


@jaxtyping.jaxtyped(typechecker=beartype.beartype)
@utils.jit
def dV(points: Float[jax.Array, "*c a=4 I=3"]) -> Float[jax.Array, "*c"]:
    return jnp.linalg.det(dX_dr(points)) / 6.0


@jaxtyping.jaxtyped(typechecker=beartype.beartype)
@utils.jit
def dh_dX(points: Float[jax.Array, "*c a=4 I=3"]) -> Float[jax.Array, "*c a=4 J=3"]:
    return einops.einsum(dh_dr(), dr_dX(points), "a I, ... I J -> ... a J")
