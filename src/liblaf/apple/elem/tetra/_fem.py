import beartype
import einops
import jax
import jax.numpy as jnp
import jaxtyping
from jaxtyping import Float

from liblaf import apple


@apple.jit()
@jaxtyping.jaxtyped(typechecker=beartype.beartype)
def h() -> Float[jax.Array, "a=4"]:
    return jnp.asarray([0.25, 0.25, 0.25, 0.25], dtype=float)


@apple.jit()
@jaxtyping.jaxtyped(typechecker=beartype.beartype)
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


@apple.jit()
@jaxtyping.jaxtyped(typechecker=beartype.beartype)
def dX_dr(points: Float[jax.Array, "*c a=4 I=3"]) -> Float[jax.Array, "*c I=3 J=3"]:
    return einops.einsum(points, dh_dr(), "... a I, a J -> ... I J")


@apple.jit()
@jaxtyping.jaxtyped(typechecker=beartype.beartype)
def dr_dX(points: Float[jax.Array, "*c a=4 I=3"]) -> Float[jax.Array, "*c I=3 J=3"]:
    return jnp.linalg.pinv(dX_dr(points))


@apple.jit()
@jaxtyping.jaxtyped(typechecker=beartype.beartype)
def dV(points: Float[jax.Array, "*c a=4 I=3"]) -> Float[jax.Array, "*c"]:
    return jnp.linalg.det(dX_dr(points)) / 6.0


@apple.jit()
@jaxtyping.jaxtyped(typechecker=beartype.beartype)
def dh_dX(points: Float[jax.Array, "*c a=4 I=3"]) -> Float[jax.Array, "*c a=4 J=3"]:
    return einops.einsum(dh_dr(), dr_dX(points), "a I, c I J -> c a J")
