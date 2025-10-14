import einops
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def svd_rv(
    a: Float[Array, "*batch I I"],
) -> tuple[
    Float[Array, "*batch I I"], Float[Array, "*batch I"], Float[Array, "*batch I I"]
]:
    u: Float[Array, "*batch I I"]
    s: Float[Array, "*batch I"]
    vh: Float[Array, "*batch I I"]
    u, s, vh = jnp.linalg.svd(a, full_matrices=False)
    det_u: Float[Array, "*batch"] = jnp.linalg.det(u)
    det_v: Float[Array, "*batch"] = jnp.linalg.det(vh)
    u = u.at[..., :, -1].multiply(jnp.where(det_u < 0, -1, 1)[..., jnp.newaxis])
    vh = vh.at[..., -1, :].multiply(jnp.where(det_v < 0, -1, 1)[..., jnp.newaxis])
    s = s.at[..., -1].multiply(jnp.where(det_u * det_v < 0, -1, 1))
    return u, s, vh


def polar_rv(
    a: Float[Array, "*batch I I"],
) -> tuple[Float[Array, "*batch I I"], Float[Array, "*batch I I"]]:
    u: Float[Array, "*batch I I"]
    s: Float[Array, "*batch I"]
    vh: Float[Array, "*batch I I"]
    u, s, vh = svd_rv(a)
    R: Float[Array, "*batch I I"] = einops.einsum(u, vh, "... i j, ... j k -> ... i k")
    S: Float[Array, "*batch I I"] = einops.einsum(
        vh, s, vh, "... j i, ... j, ... j k -> ... i k"
    )
    R = jax.lax.stop_gradient(R)
    S = jax.lax.stop_gradient(S)
    return R, S
